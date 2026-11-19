import pytest
import torch
from torch import nn

from iwi_harness.probes import Runner


class Block(nn.Module):
    def forward(self, h):
        return h, "keep the original auxiliary output"


class Target(nn.Module):
    model_id, revision, eos_ids = "test", "v1", set()

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor([[0., 0.], [4., 2.], [1., 3.]]))
        self.blocks = nn.ModuleList([Block()])

    @property
    def model(self):
        return self

    def encode(self, prompt):
        return [0] + [int(x) for x in prompt.split()]

    def decode(self, tokens):
        return " ".join(map(str, tokens))

    def logits(self, ids):
        result, extra = self.blocks[0](self.embed(ids))
        assert extra == "keep the original auxiliary output"
        return torch.cat((torch.zeros_like(result[..., :1]), result), dim=-1)


class Dictionary:
    model_id, target_revision, revision = "test", "v1", "test-dictionary"
    layer, n_features = 0, 2

    def encode(self, h):
        return h * 2

    def decode(self, z):
        # The reconstruction has an error. Suppression must preserve the original residual.
        return z / 2 + 7


class Store:
    revision = 0

    def __init__(self):
        self.handles = {}

    def capture(self, v):
        handle = str(len(self.handles))
        self.handles[handle] = v.clone()
        return handle

    def resolve(self, handle):
        return self.handles[handle]


@pytest.fixture
def runner():
    return Runner(Target(), Store(), Dictionary())


SAE_PROBE = {"layer": 0, "kind": "sae"}
SUPPRESS = {"feature": 0, "scale": 0}


def test_paired_suppression_preserves_unedited_residual_and_cleans_up(runner):
    result = runner.inspect(["1"], SAE_PROBE, intervention=SUPPRESS, token_ids=[1, 2])
    original, edited = result["baseline_outputs"][0], result["outputs"][0]
    assert original["token_id"] == 1 and edited["token_id"] == 2
    assert edited["predicted_next_token_id"] == 2
    assert [s["logit"] for s in edited["scores"]] == [0, 2]
    assert result["margin_changes"] == [-4]
    assert result["cost"] == {"forwards": 2, "forward_tokens": 4, "probe_rows": 2, "sae_rows": 2}
    obs = original["observations"][0]
    assert (obs["position"], obs["token_id"], obs["step"]) == (1, 1, 0)
    assert edited["observations"][0]["input_token_id"] == 1
    torch.testing.assert_close(runner.store.resolve(obs["handle"]), torch.tensor([8., 4.]))
    assert runner.inspect(["1"], SAE_PROBE)["outputs"][0]["token_id"] == 1
    assert not runner.target.blocks[0]._forward_hooks


def test_generation_captures_selected_generated_positions_and_clamps_a_feature(runner):
    probe = {**SAE_PROBE, "phase": "each_step"}
    result = runner.run(["1"], 3, probe, {"feature": 1, "value": 18})
    edited = result["outputs"][0]
    assert result["baseline_outputs"][0]["token_ids"] == [1, 1, 1]
    assert edited["token_ids"] == [2, 2, 2]
    assert [(r["step"], r["position"], r["token_id"]) for r in edited["observations"]] == [
        (0, 1, 1), (1, 2, 2), (2, 3, 2)]
    assert result["cost"]["forward_tokens"] == 2 * (2 + 3 + 4)
    # Earlier intervened positions remain patched during full-prefix recomputation.
    assert result["cost"]["sae_rows"] == 3 + (1 + 2 + 3)
    prefill = runner.run(["1"], 3, SAE_PROBE)
    assert len(prefill["outputs"][0]["observations"]) == 1
    assert prefill["cost"]["sae_rows"] == 1


def test_contrast_compares_corresponding_positions_across_different_prefix_lengths(runner):
    probe = {"layer": 0, "positions": [-1]}
    result = runner.contrast(["1", "2 1"], ["1 2", "2"], probe)
    torch.testing.assert_close(runner.store.resolve(result["direction"]), torch.tensor([-3., 1.]))
    assert result["ranking"][0] == {"index": 0, "value": -3}
    assert result["pair_count"] == 2 and result["cost"]["forward_tokens"] == 10
    assert result["pairs"][0]["left"] == [{"position": 1, "token_id": 1, "input_token_id": 1}]
    assert result["pairs"][0]["right"] == [{"position": 2, "token_id": 2, "input_token_id": 2}]


def test_failed_hook_always_removed_and_failed_forwards_charged(runner, monkeypatch):
    def fail(_):
        raise RuntimeError("test error")
    monkeypatch.setattr(runner.sae, "encode", fail)
    with pytest.raises(RuntimeError, match="test error"):
        runner.run(["1"], probe=SAE_PROBE)
    assert runner.usage["forwards"] == 1 and not runner.target.blocks[0]._forward_hooks
    assert runner.run(["1"])["outputs"][0]["token_ids"] == [1]


def test_budgets_cover_baseline_and_rerun_and_validation_precedes_forward():
    runner = Runner(Target(), Store(), Dictionary(), max_forward_tokens=3)
    with pytest.raises(ValueError, match="distinct"):
        runner.inspect(["1"], {**SAE_PROBE, "positions": [-1, 1]})
    assert runner.usage["forwards"] == 0
    with pytest.raises(ValueError, match="budget exhausted"):
        runner.inspect(["1"], SAE_PROBE, intervention=SUPPRESS)
    assert runner.usage["forwards"] == 1 and runner.usage["forward_tokens"] == 2
    assert not runner.target.blocks[0]._forward_hooks


def test_dictionary_must_match_target_revision():
    wrong = Dictionary()
    wrong.target_revision = "different checkpoint"
    with pytest.raises(ValueError, match="checkpoint"):
        Runner(Target(), Store(), wrong)


def test_sae_uses_loaded_layer_by_default_and_reports_its_actual_layer(runner):
    assert runner.inspect(["1"], {"kind": "sae"})["probe"]["layer"] == 0
    with pytest.raises(ValueError, match="loaded SAE is at layer 0"):
        runner.inspect(["1"], {"kind": "sae", "layer": 4})
    with pytest.raises(ValueError, match="layer must be"):
        runner.inspect(["1"], {"kind": "residual"})
    without_dictionary = Runner(Target(), Store())
    with pytest.raises(ValueError, match="no SAE loaded"):
        without_dictionary.feature_examples(["1"], 0)


def test_feature_examples_reports_public_frequency_and_position_specific_margin_changes(runner):
    result = runner.feature_examples(["1", "2", "0"], 0, top_k=2, token_ids=[1, 2])
    assert result["margin_token_ids"] == [1, 2]
    assert result["evaluated_tokens"] == 6 and result["active_tokens"] == 2
    assert result["evaluated_prompts"] == 3 and result["active_prompts"] == 2
    assert result["activation_frequency"] == pytest.approx(1 / 3)
    first, second = result["examples"]
    assert (first["prompt_index"], first["position"], first["input_token_id"], first["activation"]) == (0, 1, 1, 8)
    assert first["snippet"] == {"before": "0", "marked_token": "1", "after": ""}
    assert first["suppression"] == {"baseline_margin": 2, "suppressed_margin": -2, "margin_change": -4}
    assert second["suppression"]["margin_change"] == -1
    assert result["probe"]["positions"] == "all"
    assert result["cost"] == {"forwards": 5, "forward_tokens": 10, "probe_rows": 8, "sae_rows": 8}
    assert not runner.store.handles and not runner.target.blocks[0]._forward_hooks


def test_feature_examples_restricts_positions_and_stably_ranks_distinct_prompts(runner):
    result = runner.feature_examples(["1 0", "1 2", "2"], 0, positions=[-1], top_k=3)
    assert result["probe"]["positions"] == [-1]
    assert result["activation_frequency"] == pytest.approx(2 / 3)
    assert [(e["prompt_index"], e["position"], e["activation"]) for e in result["examples"]] == [
        (1, 2, 2), (2, 1, 2), (0, 2, 0)]
    assert all("suppression" not in e for e in result["examples"])
    assert result["cost"] == {"forwards": 3, "forward_tokens": 8, "probe_rows": 3, "sae_rows": 3}


def test_feature_examples_validates_scan_bounds_before_running(runner):
    with pytest.raises(ValueError, match="512"):
        runner.feature_examples([" ".join(["1"] * 512)], 0)
    with pytest.raises(ValueError, match="two token_ids"):
        runner.feature_examples(["1"], 0, token_ids=[1])
    with pytest.raises(ValueError, match="distinct"):
        runner.feature_examples(["1"], 0, positions=[1, -1])
    assert runner.usage["forwards"] == 0
