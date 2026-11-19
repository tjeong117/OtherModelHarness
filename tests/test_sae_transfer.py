import hashlib
import json
from pathlib import Path

import pytest


def test_pristine_only_selection_and_patch_replay_resume(monkeypatch, tmp_path):
    pytest.importorskip("iwibench")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import grade_sae_transfer as study

    def episode(family, seed, split):
        prompt = "source public" if seed in (13, 2) else f"private prompt {seed}"
        return {**{g: [] for g in study.PRIVATE_GROUPS}, "targets": [{"id": "q", "prompt": prompt}],
                "general_controls": [{"id": "c", "prompt": "shared control"}],
                "public_practice": [{"prompt": "source public"}], "seed": seed, "id": f"id/{seed}", "test_seed": seed}

    workers = []
    def completions(items, req):
        workers.append((req["seeds"][0], req.get("patch_sha256")))
        return {i["id"]: ("patched" if req["patch_path"] else "pristine") for i in items}

    def score(ep, base, edited):
        if ep["seed"] == 12:
            raise study.CalibrationError("pristine is ineligible")
        success = int(edited["q"] == "patched")
        return {"reward": success, "controls_retained": 1, "control_retention_denominator": 1,
                "groups": {"targets": dict(count=1, base_successes=0, edited_successes=success, retained_successes=0)}}

    monkeypatch.setattr(study, "_completions", completions)
    monkeypatch.setattr(study, "_episode", episode)
    monkeypatch.setattr(study, "score_episode", score)
    work, out = tmp_path / "cache", tmp_path / "public.json"
    study.evaluate({}, work, out, "cpu")
    assert workers == [(12, None), (13, None), (14, None)]

    arm = tmp_path / "arm"
    arm.mkdir()
    patch = b"test patch"
    (arm / "patch.safetensors").write_bytes(patch)
    sha = hashlib.sha256(patch).hexdigest()
    study.save(arm / "receipt.json", dict(sha256=sha, bytes=len(patch), format=study.NATIVE_FORMAT))
    study.save(arm / "protocol.json", dict(family=study.FAMILY, seed=2, iwibench_revision=study.IWIBENCH_REVISION,
                                           target=study.GEMMA_ID, target_revision=study.GEMMA_REVISION))
    args = {"request": {"texts": ["<start_of_turn>user\nprivate prompt 14<end_of_turn>"]}}
    event = {"type": "updater", "response": {"tool_calls": [{"function": {"arguments": json.dumps(args)}}]}}
    (arm / "trajectory.jsonl").write_text(json.dumps(event) + "\n")
    summary = study.evaluate({"candidate": arm}, work, out, "cpu")
    assert workers[3:] == [(13, sha), (14, sha)]
    assert summary["pooled_episode_item_counts"]["candidate"]["edited_correct"] == 2
    overlap = summary["prompt_overlaps"]
    assert overlap["across_transfer_seeds"][0]["all_private"]["shared"] == 1
    assert overlap["against_source"]["13"]["source_public_practice"]["targets"]["shared"] == 1
    assert overlap["against_source"]["14"]["candidate_public_tool_texts"]["targets"]["shared"] == 1
    assert "private prompt 14" not in out.read_text() and "shared control" not in out.read_text()
    study.evaluate({"candidate": arm}, work, out, "cpu")
    assert len(workers) == 5
    (arm / "patch.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid patch receipt"):
        study.evaluate({"candidate": arm}, work, out, "cpu")
