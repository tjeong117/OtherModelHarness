import json

import pytest
import torch
from torch import nn

from iwi_harness.experiments import BehaviorExperiment, behavior_tool, compare_behavior
from iwi_harness.probes import Runner
from iwi_harness.weights import WeightStore


class Block(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x):
        # Mix earlier positions into later logits so a last-position hook would fail equivalence.
        return self.mlp(x).cumsum(dim=1)


class Target(nn.Module):
    model_id, revision, eos_ids, vocab_size = "cpu-test", "1", set(), 3

    def __init__(self, mlp=None):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor([[0., 0.], [2., 1.], [1., 3.]]))
        self.blocks = nn.ModuleList([Block(mlp if mlp is not None else nn.Identity())])

    @property
    def model(self):
        return self

    def weights(self):
        return dict(self.named_parameters(remove_duplicate=False))

    def encode(self, prompt):
        return list(map(int, prompt.split()))

    def decode(self, ids):
        return " ".join(map(str, ids))

    def logits(self, ids):
        x = self.blocks[0](self.embedding(ids))
        return torch.cat([torch.zeros_like(x[..., :1]), x], dim=-1)


def native_target(family):
    pytest.importorskip("transformers")
    if family == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2MLP
        mlp = Qwen2MLP(Qwen2Config(hidden_size=2, intermediate_size=2))
    else:
        from transformers.models.gemma3.modeling_gemma3 import Gemma3TextConfig, Gemma3MLP
        mlp = Gemma3MLP(Gemma3TextConfig(hidden_size=2, intermediate_size=2))
    with torch.no_grad():
        for layer in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            layer.weight.copy_(torch.eye(2))
    return Target(mlp)


CASES = [{"prompt": "1 1", "slice": "improve", "answers": ["2 2"]},
         {"prompt": "2", "slice": "preserve", "answers": ["2 2"]}]


def test_no_internals_or_truncated_display_leak(tmp_path):
    target = Target()
    runner = Runner(target, WeightStore(target))
    tool = BehaviorExperiment(runner, artifact_dir=tmp_path)
    result = tool.run(CASES, 2, [1, 2])
    assert [r["correct"] for r in result["baseline"]] == [False, True]
    assert result["cost"] == {"forwards": 4, "forward_tokens": 8, "probe_rows": 0, "sae_rows": 0}
    assert all(r["truncated"] for r in result["baseline"])
    assert json.loads((tmp_path / result["artifact"]).read_text())["result"]["baseline"] == result["baseline"]
    target.decode = lambda ids: "a" * 161
    row = tool.evaluate([{"prompt": "1", "slice": "improve", "answers": ["a" * 160]}], 1)[0]
    assert len(row["text"]) == 160 and row["display_truncated"] and not row["correct"]
    assert "intervention" not in behavior_tool(False)[1]["properties"]
    with pytest.raises(ValueError, match="disabled"):
        tool.run(CASES, intervention={"layer": 0, "scale": 0})
    previous = runner.usage.copy()
    with pytest.raises(ValueError, match="verified"):
        BehaviorExperiment(runner, internal=True).run(CASES, intervention={"layer": 0, "scale": 0})
    assert runner.usage == previous


@pytest.mark.parametrize("family", ["qwen2", "gemma3"])
def test_native_public_outcomes_match_real_weight_edit_all_positions(family):
    target = native_target(family)
    runner = Runner(target, WeightStore(target))
    tool = BehaviorExperiment(runner, internal=True)
    result = tool.run(CASES, 2, [1, 2], {"layer": 0, "scale": 0})
    evidence = result["intervention"]
    assert evidence["unit"] == 0 and evidence["selection"] == "first_prompt_last_token_max_abs"
    assert result["identity"]["matches"]
    assert result["cost"] == {"forwards": 12, "forward_tokens": 24, "probe_rows": 16, "sae_rows": 0}
    assert result["comparison"]["improve"]["fixed"] == [0]
    assert result["comparison"]["improve"]["mean_margin_change"] < 0
    assert result["comparison"]["preserve"]["retained_correct"] == 1
    assert result["comparison"]["preserve"]["regressed"] == []
    assert runner.store.revision == 0 and tool.evaluate(CASES, 2, [1, 2]) == result["baseline"]
    runner.store.write(**evidence["write"])
    actual = tool.evaluate(CASES, 2, [1, 2])
    for temporary, persistent in zip(result["after"], actual, strict=True):
        assert temporary["generation_id"] == persistent["generation_id"]
        assert temporary["margin"] == pytest.approx(persistent["margin"], abs=1e-6)
    inverses = compare_behavior(actual, result["baseline"])
    assert inverses["improve"]["regressed"] == [0]
    assert inverses["improve"]["accuracy_delta"] == -1
    with pytest.raises(ValueError, match="case or slice"):
        compare_behavior(actual[::-1], result["baseline"])


def test_failed_forward_detaches_native_hook_and_keeps_metered_work():
    target = native_target("qwen2")
    runner = Runner(target, WeightStore(target), max_forwards=9)
    tool = BehaviorExperiment(runner, internal=True)
    with pytest.raises(ValueError, match="budget"):
        tool.run(CASES, 2, intervention={"layer": 0, "unit": 1, "scale": -1})
    assert runner.usage["forwards"] == 9 and runner.usage["probe_rows"] == 10
    assert not target.blocks[0].mlp.down_proj._forward_pre_hooks
    assert runner.store.revision == 0


def test_sae_intervention_uses_active_dictionary_and_all_positions():
    class Dict:
        model_id, target_revision, revision = "cpu-test", "1", "sae-1"
        n_features, layer = 2, 0

        def encode(self, h):
            return h * 2

        def decode(self, z):
            return z / 2 + 1  # Nonzero reconstruction error must survive a no-op.

    target = Target()
    runner = Runner(target, WeightStore(target), Dict())
    result = BehaviorExperiment(runner, internal=True).run(
        CASES, 2, [1, 2], {"kind": "sae", "feature": 0, "scale": 0})
    assert result["identity"]["matches"] and result["intervention"]["dictionary_revision"] == "sae-1"
    assert result["cost"]["probe_rows"] == result["cost"]["sae_rows"] == 16
    assert "write" not in result["intervention"]
    assert result["comparison"]["improve"]["fixed"] == [0]
    assert result["after"][0]["first_activity"] == 8


def test_eos_terminates_generation_without_padding_and_invalid_input_is_free():
    target = Target()
    target.eos_ids = {1}
    runner = Runner(target, WeightStore(target))
    tool = BehaviorExperiment(runner)
    for bad in ({"cases": CASES, "token_ids": [7, 2]}, {"cases": CASES, "max_new_tokens": 0},
                {"cases": [{"prompt": "1", "slice": "test", "answers": ["a"]}]}):
        with pytest.raises(ValueError):
            tool.run(**bad)
    assert runner.usage["forwards"] == 0
    assert not tool.evaluate(CASES, max_new_tokens=2)[0]["truncated"]
    assert runner.usage["forwards"] == 3


def test_failed_identity_skips_treatment_and_withholds_column_recipe():
    target = native_target("qwen2")
    forwards = 0
    raw = target.logits

    def unstable(ids):
        nonlocal forwards
        forwards += 1
        logits = raw(ids)
        if forwards == 2:
            logits[..., 0] = 1e6
        return logits

    target.logits = unstable
    runner = Runner(target, WeightStore(target))
    result = BehaviorExperiment(runner, internal=True).run(
        CASES[:1], 1, intervention={"layer": 0, "scale": -1})
    assert result["identity"] == {"matches": False, "mismatches": [0]}
    assert "write" not in result["intervention"] and "after" not in result
    assert result["cost"]["forwards"] == 2 and not target.blocks[0].mlp.down_proj._forward_pre_hooks


def test_activity_matched_control_measures_effect_and_displacement(tmp_path):
    target = native_target("qwen2")
    runner = Runner(target, WeightStore(target))
    result = BehaviorExperiment(runner, artifact_dir=tmp_path, internal=True).run(
        CASES, 2, [1, 2], {"layer": 0, "scale": 0, "matched_control": True})
    match = result["matched_control"]
    assert match["status"] == "matched" and match["alternative"]["unit"] == 1
    assert match["positions"] == match["coactive_positions"] == 3
    assert match["selected"]["active_fraction"] == match["alternative"]["active_fraction"] == 1
    assert match["selected_displacement"]["applied_positions"] == 8
    assert match["alternative_displacement"]["gated_input_delta_l2"] > 0
    d = match["selected_displacement"]
    assert d["gated_input_delta_l2"] == d["projected_delta_l2_before_output_rounding"]  # unit-norm columns
    assert match["comparison"]["preserve"]["regressed"] == [1]
    assert result["comparison"]["improve"]["fixed"] == [0]
    assert result["cost"]["forwards"] == 16 and result["cost"]["probe_rows"] == 24
    record = json.loads((tmp_path / result["artifact"]).read_text())
    assert len(record["matched_control_rows"]) == 2


def test_public_calibration_match_ties_inactivity_and_disjoint_positions():
    from iwi_harness.experiments import _ActivityMatch

    def choose(rows, unit):
        calibration = _ActivityMatch()
        calibration.observe(torch.tensor(rows, dtype=torch.float32), unit)
        return calibration.select(unit)

    nearest = choose([[1,2,1,1], [0,0,0,0]], 3)
    assert nearest["alternative"]["unit"] == 0 and nearest["rms_difference"] == 0
    assert nearest["selected"]["rms"] == pytest.approx(2**-.5)
    assert nearest["selected"]["active_fraction"] == .5
    assert choose([[0,1],[0,2]], 0)["reason"].startswith("selected unit inactive")
    assert choose([[1,0],[2,0]], 0)["reason"].startswith("all alternative units inactive")
    assert choose([[1,0],[0,1]], 0)["reason"].startswith("closest unit active on disjoint")


def test_baseline_cache_invalidation_and_fresh_preview_measurements():
    from iwi_harness.preview import WeightPreview

    target = Target()
    store = WeightStore(target)
    runner = Runner(target, store)
    tool = BehaviorExperiment(runner)
    first = tool.run(CASES, 2)
    first["baseline"][0]["text"] = "caller mutation"
    repeated = tool.run(CASES, 2)
    assert repeated["baseline_cache_hit"] and repeated["cost"]["forwards"] == 0
    assert repeated["baseline"][0]["text"] == "1 1"
    # A staged branch can reach the exact revision that a different, later direct write reaches.
    preview = WeightPreview(store)
    edit = {"tensor": "embedding.weight", "operation": "fill", "selection": [1, 0], "value": 0.}
    before = runner.usage["forwards"]
    measured = preview.preview([edit], lambda: tool.evaluate(CASES, 2))
    assert runner.usage["forwards"] - before == 8
    assert measured["after"][0]["text"] != measured["before"][0]["text"]
    assert tool.run(CASES, 2)["baseline_cache_hit"]
    # Even direct Python callers that use run() as a preview callback cannot poison subsequent reads.
    preview.preview([edit], lambda: tool.run(CASES, 2))
    store.write(**{**edit, "value": 3.})
    live = tool.run(CASES, 2)
    assert not live["baseline_cache_hit"] and live["baseline"][0]["text"] == "1 1"
    assert not tool.run(CASES, 1)["baseline_cache_hit"]
    assert not tool.run(CASES, 2, [1,2])["baseline_cache_hit"]


def test_native_discovery_checks_writable_dense_site():
    from iwi_harness.experiments import supported_native_layers

    assert supported_native_layers(Target()) == []
    target = native_target("gemma3")
    assert supported_native_layers(target) == [0]
    target.duplicate_weight = target.blocks[0].mlp.down_proj.weight
    assert supported_native_layers(target) == []
