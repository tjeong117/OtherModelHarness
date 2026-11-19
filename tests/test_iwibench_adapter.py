import hashlib
import json
import os
from pathlib import Path
import re

import pytest
import torch
from safetensors.torch import save_file

from iwi_harness import iwibench_adapter as adapter
from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION
from iwi_harness.weights import NATIVE_FORMAT


FAMILY = "resistance_to_false_suggestions"


def _install_cpu_fixture():
    """Replace only checkpoint loading and decoding inside a real worker process."""
    import iwibench.runtime as runtime
    from iwibench.families import PRIVATE_GROUPS, episode

    class TinyWorkerTarget(runtime.NativeTarget):
        def __init__(self, model_id, revision, device, cache_dir):
            assert (model_id, revision, device) == (GEMMA_ID, GEMMA_REVISION, "cpu")
            self.model_id, self.revision = model_id, revision
            self.model = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
            self.model.weight.data.zero_()
            self.weight_revision = 0
            ep = episode(FAMILY, 0, model_id=model_id)
            self.items = {item["prompt"]: item for group in PRIVATE_GROUPS for item in ep[group]}
            with open(os.environ["IWI_TEST_WORKER_PIDS"], "a") as log:
                print(os.getpid(), file=log)

        def generate(self, prompts, max_new_tokens, seeds):
            completions = []
            for prompt in prompts:
                item = self.items[prompt]
                answer = item["answers"][0]
                if (self.model.weight.item() == 0 and item.get("suggestion_correct") is False):
                    answer = f"agree; {item['suggestion']}"
                completions.append({"text": answer, "input_tokens": 16, "output_tokens": 2})
            return completions

    runtime.NativeTarget = TinyWorkerTarget


def _native_patch(path):
    save_file({"weight": torch.ones(1, 1, dtype=torch.bfloat16)}, str(path), metadata={
        "model_id": GEMMA_ID, "revision": GEMMA_REVISION, "format": NATIVE_FORMAT,
    })
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_task_projection(monkeypatch):
    pytest.importorskip("iwibench")
    monkeypatch.setattr(adapter, "_cached_tokenizer", lambda: None)
    ep = adapter._episode(FAMILY, 0, "development")
    packet = adapter.public_task(FAMILY, 0)
    assert packet["objective"] == ep["objective"]
    assert packet["practice"] == ep["public_practice"]
    assert packet["token_ids"] == {}
    assert set(packet) == {"id", "model_id", "checkpoint_revision", "family", "family_version",
                           "objective", "practice", "token_ids"}
    assert all(item["prompt"] not in json.dumps(packet) for item in ep["targets"])


def test_alias_context_uses_upstream_native_offsets_and_hides_private_occurrences(monkeypatch):
    pytest.importorskip("iwibench")
    monkeypatch.setattr(adapter, "_cached_tokenizer", lambda: None)
    with pytest.raises(RuntimeError, match="pinned Gemma tokenizer"):
        adapter.public_task("alias_acquisition", 3)

    class Tokenizer:
        all_special_ids = [0]

        def __call__(self, text, **kwargs):
            words = list(re.finditer(r"\w+|[^\w\s]", text))
            return {"input_ids": [self.encode(w[0])[0] for w in words],
                    "offset_mapping": [w.span() for w in words]}

        def encode(self, text, **kwargs):
            return [1 + sum(map(ord, text))]

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            rendered = "user: " + messages[0]["content"] + " assistant:"
            return self(rendered)["input_ids"] if tokenize else rendered

    tokenizer = Tokenizer()
    monkeypatch.setattr(adapter, "_cached_tokenizer", lambda: tokenizer)
    ep = adapter._episode("alias_acquisition", 3, "development")
    packet = adapter.public_task("alias_acquisition", 3)
    word = ep["parameters"]["alias"]
    alias = packet["token_ids"]["alias_context"][word]
    assert alias == {"token_ids": tokenizer.encode(word), "standalone_token_ids": tokenizer.encode(word),
                     "leading_space_token_ids": tokenizer.encode(" " + word)}
    assert "target:0" not in json.dumps(packet)


def test_false_suggestions_native_replay_in_two_cpu_workers(tmp_path, monkeypatch):
    pytest.importorskip("iwibench")
    patch = tmp_path / "submitted.safetensors"
    digest = _native_patch(patch)
    # Each child runs upstream's unmodified CLI, generates in batches, and applies
    # the actual native patch. Its one BF16 parameter determines the fixture output.
    (tmp_path / "sitecustomize.py").write_text(
        f"import runpy\nrunpy.run_path({str(Path(__file__).resolve())!r})['_install_cpu_fixture']()\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(tmp_path), str(Path(adapter.__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]
    ))
    pids = tmp_path / "pids.txt"
    monkeypatch.setenv("IWI_TEST_WORKER_PIDS", str(pids))
    run_worker = adapter._run_worker

    def swap_submission_during_baseline(request):
        if request["patch_path"] is None:
            patch.write_bytes(b"Superseded by another trial")
        return run_worker(request)

    monkeypatch.setattr(adapter, "_run_worker", swap_submission_during_baseline)
    result = adapter.grade_patch(patch, family=FAMILY, seed=0, device="cpu", expected_sha256=digest)
    assert len(set(pids.read_text().splitlines())) == 2
    metrics = result["metrics"]
    assert result["status"] == "scored" and result["reward"] == 1.0
    assert result["patch_sha256"] == digest
    assert metrics["target_success_base"] == 0.5 and metrics["target_success_edited"] == 1.0
    assert metrics["groups"]["targets"]["count"] == 64
    assert metrics["control_retention"] == 1.0 and metrics["admitted"] is False


def test_ineligible_and_failed_workers_have_distinct_outcomes(tmp_path, monkeypatch):
    pytest.importorskip("iwibench")
    patch = tmp_path / "submitted.safetensors"
    _native_patch(patch)
    calls = []

    def incapable_worker(request):
        calls.append(request)
        return {"model_id": GEMMA_ID, "revision": GEMMA_REVISION, "patch_metadata": None,
                "generations": [{"text": "", "input_tokens": 8, "output_tokens": 0}] * len(request["prompts"])}

    monkeypatch.setattr(adapter, "_run_worker", incapable_worker)
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.grade_patch(patch, family=FAMILY, seed=0, expected_sha256="0" * 64)
    assert not calls
    result = adapter.grade_patch(patch, family=FAMILY, seed=0)
    assert result["status"] == "ineligible" and result["reward"] is None and result["reasons"]
    assert len(calls) == 1 and calls[0]["patch_path"] is None
    monkeypatch.setattr(adapter, "_run_worker", lambda request: {**incapable_worker(request), "generations": []})
    with pytest.raises(ValueError, match="Incomplete"):
        adapter.grade_patch(patch, family=FAMILY, seed=0)


def test_only_supported_development_families():
    with pytest.raises(ValueError, match="development"):
        adapter.public_task(FAMILY, 0, split="test")
    with pytest.raises(ValueError, match="Supported"):
        adapter.public_task("unknown", 0)
