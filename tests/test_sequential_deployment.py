import importlib
import json
from pathlib import Path

import torch

from iwi_harness.targets import tiny_target
from iwi_harness.weights import WeightStore


def test_model_selected_sequence_carries_only_submitted_weights(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    driver = importlib.import_module("run_qwen_memory")
    source = tiny_target()
    lessons = [{"alias": alias, "country": "red"} for alias in ("first", "second", "third")]
    enrollment = tmp_path / "enrollment.json"
    enrollment.write_text(json.dumps({"model": source.model_id, "revision": source.revision,
                                      "sequences": [{"id": "sequence-1", "associations": lessons}]}))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"protocol": "test", "target": source.model_id,
                                 "target_revision": source.revision,
                                 "enrollment_sha256": driver.digest(enrollment),
                                 "target_limits": {}, "editor_decoding": {"max_turns": 6}}))
    monkeypatch.setattr(driver, "load_gemma_target", lambda **_: tiny_target())
    monkeypatch.setattr(type(source), "tokenizer", None, raising=False)
    monkeypatch.setattr(driver, "public_packet", lambda binding, *_: {
        "binding": binding, "public_cases": [{"prompt": "color blue", "answers": ["yes"], "slice": "preserve"}]})

    class Editor:
        def __init__(self, policy):
            pass

        def episode(self, *args):
            return None, [{"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}]

    monkeypatch.setattr(driver, "QwenEditor", Editor)

    def act(session, objective, *_, **kwargs):
        stage = json.loads(objective)["binding"]["alias"]
        row = {"first": 0, "second": 1, "third": 2}[stage]
        session.call("write_weights", {"operation": "fill", "tensor": "embed.weight", "selection": [row],
                                       "value": row + 5})
        if stage == "second":
            raise driver.NoSubmission("did not submit")
        session.submit()

    monkeypatch.setattr(driver, "run_updater", act)
    out = tmp_path / "live"
    driver.run(policy, enrollment, out)
    snapshots = out / "snapshots" / "sequence-1"
    _, first = driver.load_snapshot(snapshots / "stage-1")
    _, second = driver.load_snapshot(snapshots / "stage-2")
    patch, third = driver.load_snapshot(snapshots / "stage-3")
    assert second["disposition"] == "carried_forward"
    assert second["sha256"] == first["sha256"] == third["previous_sha256"]
    assert not (out / "episodes/sequence-1/stage-2/receipt.json").exists()
    replay = tiny_target()
    WeightStore(replay).apply(patch, expected_sha256=third["sha256"])
    matrix = replay.weights()["embed.weight"]
    assert torch.all(matrix[0] == 5) and torch.all(matrix[2] == 7)
    assert torch.equal(matrix[1], source.weights()["embed.weight"][1])
    def reject_repeated_stage(*_):
        raise AssertionError("Resuming repeated an already completed lesson")

    monkeypatch.setattr(driver, "run_stage", reject_repeated_stage)
    driver.run(policy, enrollment, out)
