from types import SimpleNamespace

import pytest
import torch

from iwi_harness.causal import CausalTools
from iwi_harness.probes import Runner
from iwi_harness.targets import GemmaTarget, HFCausalTarget, discover_sites, tiny_target


SITES = [{"layer": 0, "component": component} for component in ("attention", "mlp", "residual")]


def make_tools(target=None, **limits):
    return CausalTools(Runner(target or tiny_target(), SimpleNamespace(revision=7), **limits))


def test_last_residual_transplant_reproduces_source_and_linear_attribution():
    tools = make_tools()
    args = dict(source_prompt="color check red", target_prompt="check blue", sites=SITES,
                token_ids=[6, 7], source_position=3, target_position=-1, top_k=1)
    exact = tools.patch_activations(**args)
    assert exact["source"] == {"position": 3, "token_id": 3, "input_token_id": 3, "margin": 2.5}
    assert exact["baseline"] == {"position": 2, "token_id": 4, "input_token_id": 4, "margin": -2.5}
    assert exact["revision"] == 7 and exact["evaluated_sites"] == 3
    assert exact["ranking"] == [dict(layer=0, component="residual", patched_margin=2.5, margin_change=5.0)]
    assert exact["cost"] == dict(forwards=5, forward_tokens=16, probe_rows=6, sae_rows=0)
    with torch.inference_mode():
        estimate = tools.rank_attributions(**args)
    assert estimate["ranking"] == [dict(layer=0, component="residual", estimated_margin_change=5.0)]
    assert estimate["cost"] == dict(forwards=2, forward_tokens=7, backwards=1, backward_tokens=3,
                                    probe_rows=6, sae_rows=0)
    # Reversing token order negates both the metric and the predicted change.
    args["token_ids"] = [7, 6]
    assert tools.rank_attributions(**args)["ranking"][0]["estimated_margin_change"] == -5
    assert all(p.grad is None and not p.requires_grad for p in tools.runner.target.model.parameters())


def test_component_effects_are_independent_and_graph_paths_remain_connected():
    tools = make_tools()
    block = tools.runner.target.blocks[0]
    with torch.no_grad():
        # Give each branch a nonzero contribution to the answer margin.
        block.attn_out.weight[0, 0] = 0.5
        block.down.weight[1, 0] = 0.5
        block.up.weight[0, 0] = 1.0
    args = ("red", "blue", [6, 7])
    joint = tools.rank_attributions(*args, SITES)
    for item in joint["ranking"]:
        site = {"layer": 0, "component": item["component"]}
        single = tools.rank_attributions(*args, [site])["ranking"][0]
        assert single["estimated_margin_change"] == pytest.approx(item["estimated_margin_change"])
        assert abs(single["estimated_margin_change"]) > 0.01
        if item["component"] in ("mlp", "residual"):
            # Both sites are linear through the final tiny readout.
            patch = tools.patch_activations(*args, [site])["ranking"][0]
            assert patch["margin_change"] == pytest.approx(item["estimated_margin_change"], abs=1e-6)
    noop = tools.patch_activations("red", "red", [6, 7], SITES)
    assert all(r["margin_change"] == 0 for r in noop["ranking"])


def test_nonlinear_attention_attribution_matches_local_difference_at_target():
    tools = make_tools()
    target = tools.runner.target
    block = target.blocks[0]
    with torch.no_grad():
        block.attn_out.weight[0, 0] = 0.5
        block.up.weight[0, 0] = -1.0
        block.down.weight[1, 0] = 0.5
    source = torch.tensor([target.encode("red")])
    destination = torch.tensor([target.encode("blue")])
    natural = []
    with block.attn_out.register_forward_hook(lambda _, args, h: natural.append(h[0, -1].clone())):
        target.logits(source)
        target.logits(destination)

    def measure(alpha):
        def interpolate(_, args, output):
            result = output.clone()
            result[0, -1] = natural[1] + alpha * (natural[0] - natural[1])
            return result

        with block.attn_out.register_forward_hook(interpolate):
            scores = target.logits(destination)[0, -1]
        return (scores[6] - scores[7]).item()

    epsilon = 1e-3
    numerical_derivative = (measure(epsilon) - measure(-epsilon)) / (2 * epsilon)
    args = ("red", "blue", [6, 7], [{"layer": 0, "component": "attention"}])
    estimate = tools.rank_attributions(*args)["ranking"][0]["estimated_margin_change"]
    exact = tools.patch_activations(*args)["ranking"][0]["margin_change"]
    assert estimate == pytest.approx(numerical_derivative, rel=1e-4)
    assert exact == pytest.approx(measure(1.0) - measure(0.0), rel=1e-6)
    assert abs(exact - estimate) > 1.0


@pytest.mark.parametrize(("overrides", "message"), [
    ({"sites": SITES * 2}, "duplicate"), ({"source_position": 5}, "positions"),
    ({"target_position": True}, "target_position"), ({"sites": [{"layer": 1, "component": "mlp"}]}, "layer"),
    ({"token_ids": [6, 6]}, "duplicate"), ({"token_ids": [6, 999]}, "token_ids"),
    ({"token_ids": [6]}, "two distinct"), ({"target_prompt": "blue " * 520}, "512"),
])
def test_invalid_request_is_rejected_before_compute(overrides, message):
    tools = make_tools()
    args = dict(source_prompt="red", target_prompt="blue", token_ids=[6, 7], sites=SITES)
    with pytest.raises(ValueError, match=message):
        tools.rank_attributions(**(args | overrides))
    assert tools.runner.usage["forwards"] == 0


def test_forward_budget_applies_to_whole_call_and_hooks_clean_up_after_failure(monkeypatch):
    tools = make_tools(max_forward_tokens=4)
    args = ("red", "blue", [6, 7], SITES)
    with pytest.raises(ValueError, match="budget"):
        tools.patch_activations(*args)
    assert tools.runner.usage["forwards"] == 0
    tools.rank_attributions(*args)
    with pytest.raises(ValueError, match="budget"):
        tools.rank_attributions(*args)
    assert tools.runner.usage["backwards"] == 1
    tools = make_tools()
    target = tools.runner.target
    original = target.model.forward

    def fail_after_forward(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("downstream failure")

    monkeypatch.setattr(target.model, "forward", fail_after_forward)
    with pytest.raises(RuntimeError, match="downstream failure"):
        tools.patch_activations(*args)
    assert tools.runner.usage["forwards"] == 1
    assert all(not module._forward_hooks for module in target.model.modules())


def test_gemma_native_layers_preserve_branch_gradients_and_final_residual_margin():
    hf = pytest.importorskip("transformers")
    config = hf.Gemma3Config(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=128,
                         sliding_window=16, pad_token_id=0, bos_token_id=1, eos_token_id=2),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                           image_size=8, patch_size=4),
        mm_tokens_per_image=4, pad_token_id=0, eos_token_id=[2, 3], attn_implementation="eager",
    )
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1] + list(map(int, messages[0]["content"].split()))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(761)
        target = GemmaTarget(hf.Gemma3ForConditionalGeneration(config), Tokenizer())
    tools = make_tools(target)
    sites = SITES + [{"layer": 1, "component": "residual"}]
    args = ("7 4 3", "5 8", [6, 7])
    estimates = tools.rank_attributions(*args, sites)
    exact = tools.patch_activations(*args, sites)
    for item in estimates["ranking"]:
        site = {"layer": item["layer"], "component": item["component"]}
        assert abs(item["estimated_margin_change"]) > 1e-8
        single = tools.rank_attributions(*args, [site])["ranking"][0]
        assert single["estimated_margin_change"] == pytest.approx(item["estimated_margin_change"], abs=1e-6)
    last_patch = next(item for item in exact["ranking"] if item["layer"] == 1)
    assert last_patch["patched_margin"] == pytest.approx(exact["source"]["margin"], abs=1e-6)
    assert exact["baseline"]["margin"] == pytest.approx(estimates["baseline"]["margin"])
    assert all(p.grad is None and not p.requires_grad for p in target.model.parameters())


@pytest.mark.parametrize("family", ("qwen2", "llama", "gpt2"))
def test_hf_decoders_expose_real_causal_sites_and_differentiable_margin(family):
    hf = pytest.importorskip("transformers")
    if family == "gpt2":
        config = hf.GPT2Config(vocab_size=32, n_embd=16, n_layer=2, n_head=2, n_positions=64)
    else:
        config = hf.AutoConfig.for_model(family, vocab_size=32, hidden_size=16, intermediate_size=32,
                                         num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=2,
                                         max_position_embeddings=64)
    config._attn_implementation = "eager"
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            return [1] + list(map(int, text.split())) if add_special_tokens else list(map(int, text.split()))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(176)
        target = HFCausalTarget(hf.AutoModelForCausalLM.from_config(config), Tokenizer(),
                                model_id=f"fixture/{family}", revision="v1")
    assert discover_sites(target)["components"] == ["residual", "attention", "mlp"]
    tools = make_tools(target)
    args = ("4 7", "6 8", [3, 9], [*SITES, {"layer": 1, "component": "residual"}])
    with torch.inference_mode():
        approx = tools.rank_attributions(*args)
    exact = tools.patch_activations(*args)
    assert all(torch.isfinite(torch.tensor(r["estimated_margin_change"])) for r in approx["ranking"])
    assert all(r["estimated_margin_change"] != 0 for r in approx["ranking"])
    last = next(r for r in exact["ranking"] if r["layer"] == 1)
    assert last["patched_margin"] == pytest.approx(exact["source"]["margin"], abs=1e-6)
    assert all(p.grad is None for p in target.model.parameters())
    # An unfamiliar family with a recognized ordered block list still exposes residuals.
    target.model.config.model_type = "custom-text"
    residual_target = HFCausalTarget(target.model, Tokenizer(), model_id="custom", revision="v1")
    assert discover_sites(residual_target)["components"] == ["residual"]
    with pytest.raises(ValueError, match="Available components: residual"):
        make_tools(residual_target).patch_activations("4", "5", [3, 9], SITES)
