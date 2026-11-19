import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from iwi_harness import EditingSession
from iwi_harness.targets import tiny_sae, tiny_target


FIT = {"prompts": ["color red", "color blue", "color dax"], "layer": 0,
       "n_features": 8, "top_k": 2, "steps": 2}
SAE_PROBE = {"prompts": ["color dax"], "probe": {"kind": "sae"}}
ADAPTIVE = {"discover_target", "tokenize_target", "fit_sae", "select_sae"}


def test_empty_registry_fit_and_live_intervention(tmp_path):
    session = EditingSession(tiny_target(), artifact_dir=tmp_path)
    available = session.call("discover_target", {})
    assert available["dictionaries"] == [] and available["active_dictionary"] is None
    assert available["sites"]["layer_range"] == [0, 0]
    assert available["sites"]["dictionary_component"] == "residual"
    baseline = {n: t.detach().clone() for n, t in session.store.params.items()}
    fitted = session.call("fit_sae", FIT)
    active = fitted["dictionary_id"]
    assert fitted["active_dictionary"] == active
    assert fitted["dictionary"]["source"] == "fitted"
    assert fitted["validation"]["prompts"] == 1
    assert fitted["fit_budget"]["used_steps"] == 2
    assert fitted["cost"]["forwards"] == 3
    inspection = session.call("inspect_target", SAE_PROBE)
    assert inspection["probe"]["dictionary_id"] == active
    row = inspection["outputs"][0]["observations"][0]
    feature = row["top"][0]["index"]
    experiment = {**SAE_PROBE, "intervention": {"feature": feature, "scale": 0}, "token_ids": [6, 7]}
    assert "baseline_outputs" in session.call("inspect_target", experiment)
    assert all(torch.equal(before, session.store.params[n]) for n, before in baseline.items())
    assert session.submit()["changed_parameters"] == 0
    trace = [json.loads(x) for x in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert trace[0]["active_dictionary"] is None
    assert all(x["active_dictionary"] == active for x in trace[1:])


def test_switch_preloaded_and_fitted_dictionaries_across_weight_edits(tmp_path):
    preloaded = tiny_sae()
    session = EditingSession(tiny_target(), artifact_dir=tmp_path, sae=preloaded, max_fitted_dictionaries=2)
    one = session.call("fit_sae", FIT)["dictionary_id"]
    session.call("write_weights", {"tensor": "embed.weight", "selection": [5], "operation": "scale", "factor": 0.9})
    two = session.call("fit_sae", FIT)["dictionary_id"]
    assert one != two
    state = session.call("discover_target", {})
    assert state["weight_revision"] == 1 and state["active_dictionary"] == two
    descriptors = {r["id"]: r for r in state["dictionaries"]}
    assert descriptors[one]["fitted_weight_revision"] == 0
    assert descriptors[two]["fitted_weight_revision"] == 1
    assert descriptors[preloaded.dictionary_id]["source"] == "preloaded"
    with pytest.raises(ValueError, match="limit"):
        session.call("fit_sae", FIT)
    for key in (one, preloaded.dictionary_id, two):
        assert session.call("select_sae", {"dictionary_id": key})["active_dictionary"] == key
        observed = session.call("inspect_target", SAE_PROBE)
        assert observed["revision"] == 1 and observed["probe"]["dictionary_id"] == key
        assert observed["probe"].get("fitted_weight_revision") == descriptors[key]["fitted_weight_revision"]
    with pytest.raises(ValueError, match="unknown dictionary"):
        session.call("select_sae", {"dictionary_id": "missing"})
    assert session.runner.sae.dictionary_id == two
    assert session.call("select_sae", {"dictionary_id": None})["active_dictionary"] is None
    with pytest.raises(ValueError, match="no SAE"):
        session.call("inspect_target", SAE_PROBE)


def test_fit_step_limit_is_atomic_and_failed_fit_preserves_selection(tmp_path):
    preloaded = tiny_sae()
    session = EditingSession(tiny_target(), artifact_dir=tmp_path, sae=preloaded, max_sae_fit_steps=2)
    with pytest.raises(ValueError, match="distinct"):
        session.call("fit_sae", {**FIT, "validation_prompts": FIT["prompts"]})
    assert session.runner.sae is preloaded
    def attempt(seed):
        try:
            return session.call("fit_sae", {**FIT, "seed": seed})
        except ValueError as exc:
            assert "remaining optimizer steps" in str(exc)
            return None
    with ThreadPoolExecutor(2) as workers:
        results = list(workers.map(attempt, [1, 2]))
    assert sum(x is not None for x in results) == 1
    assert session.call("discover_target", {})["fit_budget"]["remaining_steps"] == 0
    assert session.runner.usage["forwards"] == 3


@pytest.mark.parametrize(("interpretability", "expected"), [(True, 9), (False, 4)])
def test_fixed_interfaces_keep_historical_schema_and_settings(tmp_path, interpretability, expected):
    session = EditingSession(tiny_target(), artifact_dir=tmp_path, interpretability=interpretability,
                             adaptive_interpretability=False)
    assert len(session.tools) == expected
    assert not ADAPTIVE.intersection(t["function"]["name"] for t in session.tools)
    assert "max_sae_fit_steps" not in session.info["limits"]
    assert "adaptive_interpretability" not in session.info
    for name in ADAPTIVE:
        with pytest.raises(ValueError, match="unknown tool"):
            session.call(name, {})


@pytest.mark.parametrize("adaptive", [False, True])
def test_suppression_feedback_checks_exact_activation_outside_top_k(tmp_path, adaptive):
    dictionary = tiny_sae()
    # Feature 1 always fires below feature 0, but its decoder row has zero effect on the target.
    dictionary.w_enc.zero_()
    dictionary.b_enc[:2] = torch.tensor([2.0, 1.0])
    dictionary.w_dec[1].zero_()
    session = EditingSession(tiny_target(), sae=dictionary, artifact_dir=tmp_path,
                             adaptive_interpretability=adaptive)
    for feature, activation in ((1, 1.0), (2, 0.0)):
        result = session.call("inspect_target", {
            "prompts": ["color red", "color blue"], "probe": {"kind": "sae", "positions": [0, -1], "top_k": 1},
            "intervention": {"feature": feature, "scale": 0}, "token_ids": [6, 7],
        })
        observations = [obs for row in result["baseline_outputs"] for obs in row["observations"]]
        assert len(observations) == 4
        assert all(obs["top"] == [{"index": 0, "value": 2.0}] for obs in observations)
        assert result["margin_changes"] == [0.0, 0.0]
        assert result["cost"]["forwards"] == 4 and result["cost"]["sae_rows"] == 8
        if adaptive:
            assert result["suppressed_feature"] == feature
            assert all(obs["pre_suppression_activation"] == activation for obs in observations)
            assert ("suppression_feedback" in result) == (activation == 0)
        else:
            assert all("pre_suppression_activation" not in obs for obs in observations)
            assert "suppressed_feature" not in result and "suppression_feedback" not in result
