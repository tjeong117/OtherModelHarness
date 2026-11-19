"""Exercise the GPU-check protocol offline with the deterministic target."""

from iwi_harness import gpu_check
from iwi_harness.targets import tiny_sae, tiny_target


def test_smoke_checks_native_logits_temporary_intervention_and_replay(tmp_path, monkeypatch):
    for attr, value in {
        "EMBEDDING": "embed.weight", "SOURCE_ID": 3, "ALIAS_ID": 5,
        "ALIAS_PROMPT": "color dax", "SOURCE_PROMPT": "color red",
    }.items():
        monkeypatch.setattr(gpu_check, attr, value)
    discovery = gpu_check._discover(tiny_target(), tiny_sae(), tmp_path)
    assert discovery["pristine"]["prompts"]["color dax"]["text"] == "no"
    assert discovery["edited"]["prompts"]["color dax"]["text"] == "yes"
    assert discovery["identity"]["changed_logits"] == 0
    assert discovery["suppression"]["changed_logits"] > 0
    assert discovery["suppression"]["baseline_logit_sha256"] == discovery["cleanup_logit_sha256"]
    artifact = tmp_path / "patch.safetensors"
    replay = gpu_check._replay(tiny_target(), artifact, discovery["receipt"]["sha256"])
    assert replay["measurement"] == discovery["edited"]
    assert replay["target_usage"]["sae_rows"] == 0
