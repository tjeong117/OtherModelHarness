import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from iwi_harness.targets import tiny_target


def test_matched_driver_exposes_one_optional_fit_and_grades_only_the_receipt(monkeypatch, tmp_path):
    pytest.importorskip("iwibench")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import astra_sae_ab as trial

    monkeypatch.setattr(trial, "load_gemma_target", lambda **_: tiny_target())
    monkeypatch.setattr(trial, "packet_for", lambda *args: {"objective": "Learn the public alias."})
    monkeypatch.setattr(trial, "AstraResponses", lambda *args, **kwargs: SimpleNamespace(records=[]))
    seen_packets = []
    seen_schemas = []

    def scripted(session, task, complete, **kwargs):
        seen_packets.append((task, kwargs))
        seen_schemas.append(session.tools)
        before = session.call("discover_target", {})
        assert not before["dictionaries"]
        assert before["active_dictionary"] is None
        fitting = {"prompts": ["color red", "color blue", "color dax"], "layer": 0,
                   "n_features": 8, "top_k": 2, "steps": 2}
        if before["fit_budget"]["remaining_dictionaries"]:
            assert session.call("fit_sae", fitting)["method"] == "topk_sae"
        else:
            with pytest.raises(ValueError, match="unknown tool"):
                session.call("fit_sae", fitting)
        return session.submit()

    monkeypatch.setattr(trial, "run_updater", scripted)
    for arm in ("fit", "no-fit"):
        out = tmp_path / arm
        trial.edit(SimpleNamespace(case="alias", seed=1, arm=arm, out=out, model="gpt-6.2-astra",
                                   budget_usd=1.25, max_output_tokens=2048, max_turns=6), "mock")
        report = json.loads((out / "editor-result.json").read_text())
        assert report["status"] == "submitted"
        assert report["session"]["sae"] is None
        assert json.loads((out / "tools.json").read_text()) == seen_schemas[-1]

        def check_grade(path, *, family, seed, expected_sha256):
            assert path == out / "patch.safetensors"
            assert family == "alias_acquisition" and seed == 1
            assert expected_sha256 == report["receipt"]["sha256"]
            return {"status": "scored", "reward": 0}

        monkeypatch.setattr(trial, "grade_patch", check_grade)
        trial.grade(out)
        assert json.loads((out / "grade.json").read_text()) == {"status": "scored", "reward": 0}
    assert seen_packets[0] == seen_packets[1]
    assert [s for s in seen_schemas[0] if s["function"]["name"] != "fit_sae"] == seen_schemas[1]


def test_instructed_followup_changes_only_objective_and_protocol(monkeypatch, tmp_path):
    pytest.importorskip("iwibench")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import astra_sae_ab as trial

    monkeypatch.setattr(trial, "load_gemma_target", lambda **_: tiny_target())
    monkeypatch.setattr(trial, "AstraResponses", lambda *args, **kwargs: SimpleNamespace(records=[]))
    monkeypatch.setattr(trial, "packet_for", lambda *args: {"objective": "Maintain accuracy.", "practice": []})
    packets = []
    def collect(session, task, *args, **kwargs):
        packets.append(json.loads(task))
        return session.submit()
    monkeypatch.setattr(trial, "run_updater", collect)

    common = dict(case="suggestion", seed=2, arm="fit", budget_usd=1.25, model="gpt-6.2-astra",
                  max_output_tokens=2048, max_turns=6)
    for requested in (False, True):
        out = tmp_path / str(requested)
        trial.edit(SimpleNamespace(**common, request_sae_check=requested, out=out), "mock")
        assert json.loads((out / "public-input.json").read_text()) == packets[-1]
    assert packets[0] == {"objective": "Maintain accuracy.", "practice": []}
    assert packets[1] == {**packets[0], "objective": "Maintain accuracy. Fit a small SAE using public prompts "
                         "of your choice. Use a learned feature to check a hypothesis relevant to this edit, "
                         "then make your best weight edit and submit within the same turn budget."}
    original, followup = (json.loads((tmp_path / d / "protocol.json").read_text()) for d in ("False", "True"))
    assert original.pop("protocol") == "runtime-sae-access-v1"
    assert followup.pop("protocol") == "runtime-sae-instructed-v1"
    assert original == followup
    assert (tmp_path / "False/tools.json").read_bytes() == (tmp_path / "True/tools.json").read_bytes()
    with pytest.raises(ValueError, match="requires"):
        trial.edit(SimpleNamespace(**{**common, "arm": "no-fit"}, request_sae_check=True), "mock")


def test_sae_guided_arm_records_the_shared_editing_policy(monkeypatch, tmp_path):
    pytest.importorskip("iwibench")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import astra_sae_ab as trial
    from iwi_harness.updater import sae_guided_objective

    monkeypatch.setattr(trial, "load_gemma_target", lambda **_: tiny_target())
    monkeypatch.setattr(trial, "AstraResponses", lambda *args, **kwargs: SimpleNamespace(records=[]))
    monkeypatch.setattr(trial, "packet_for", lambda *args: {"objective": "Teach the alias."})
    args = SimpleNamespace(case="alias", seed=1, arm="fit", out=tmp_path, model="gpt-6.2-astra",
                           budget_usd=0.9, max_output_tokens=1536, max_turns=5, sae_guided=True)
    expected = {"objective": sae_guided_objective("Teach the alias.", 5)}

    def submit(session, task, *args, **kwargs):
        assert json.loads(task) == expected
        return session.submit()

    monkeypatch.setattr(trial, "run_updater", submit)
    trial.edit(args, "mock")
    assert json.loads((tmp_path / "public-input.json").read_text()) == expected
    assert json.loads((tmp_path / "protocol.json").read_text())["protocol"] == "runtime-sae-guided-v2"
    args.arm = "no-fit"
    with pytest.raises(ValueError, match="requires --arm fit"):
        trial.edit(args, "mock")
