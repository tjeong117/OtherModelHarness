import pytest
import torch

from iwi_harness.dictionaries import fit_dictionary
from iwi_harness.probes import Runner
from iwi_harness.targets import tiny_target
from iwi_harness.weights import WeightStore


PROMPTS = ["color red", "color blue", "check blue", "check red", "dax red"]
VALIDATION = ["blue red", "red blue"]


def make_runner(**limits):
    target = tiny_target()
    return Runner(target, WeightStore(target), **limits)


def test_fit_learns_sparse_dictionary_without_modifying_target_or_random_state():
    runner = make_runner()
    weights = {k: w.detach().clone() for k, w in runner.target.weights().items()}
    state = torch.random.get_rng_state()
    with torch.inference_mode():
        dictionary, report = fit_dictionary(runner, PROMPTS, 0, n_features=8, top_k=2, steps=32,
                                            seed=42, validation_prompts=VALIDATION)
    assert torch.equal(torch.random.get_rng_state(), state)
    for name, weight in runner.target.weights().items():
        assert torch.equal(weights[name], weight) and weight.grad is None
    assert runner.store.revision == report["source_revision"] == 0
    assert not any(p.requires_grad or p.grad is not None for p in dictionary.parameters())
    assert report["training"]["reconstruction_mse"] < report["training"]["initial_reconstruction_mse"]
    assert report["validation"]["prompts"] == 2
    assert report["validation"]["rows"] == 6
    assert 0 < report["validation"]["mean_active_features"] <= 2
    assert report["validation"]["fraction_squared_error_reduced"] > 0
    assert report["cost"] == {"forwards": 7, "forward_tokens": 21, "probe_rows": 21, "sae_rows": 0,
                              "sae_fit_steps": 32, "sae_fit_rows": 32 * 15}
    runner.sae = dictionary
    probe = {"kind": "sae", "top_k": 2}
    source = runner.inspect(["red"], probe)
    feature = source["outputs"][0]["observations"][0]["top"][0]["index"]
    result = runner.inspect(["red"], probe, {"feature": feature, "scale": 0}, token_ids=[6, 7])
    assert result["probe"]["dictionary_id"] == dictionary.dictionary_id
    assert result["probe"]["sae_revision"] == dictionary.revision
    assert len(result["margin_changes"]) == 1 and report["dictionary_id"].startswith("session/topk-")


def test_fit_reproducible_and_default_split_uses_last_prompts():
    runner = make_runner()
    args = dict(prompts=PROMPTS, layer=0, positions=[-1], n_features=6, top_k=2, steps=3)
    first, a = fit_dictionary(runner, **args)
    again, b = fit_dictionary(runner, **args)
    assert first.dictionary_id == again.dictionary_id and first.revision == again.revision
    assert a == b and a["training"]["rows"] == 4 and a["validation"]["rows"] == 1
    torch.testing.assert_close(first.decoder.norm(dim=1), torch.ones(6))


def test_invalid_fits_do_not_consume_forwards_or_install_a_dictionary():
    runner = make_runner(max_forward_tokens=3)
    with pytest.raises(ValueError, match="distinct tokenized"):
        fit_dictionary(runner, ["red", "blue"], 0, validation_prompts=["red"])
    with pytest.raises(ValueError, match="budget"):
        fit_dictionary(runner, PROMPTS, 0)
    with pytest.raises(ValueError, match="1,024"):
        fit_dictionary(runner, ["red " * 1024, "blue"], 0)
    assert runner.usage["forwards"] == 0 and runner.sae is None
