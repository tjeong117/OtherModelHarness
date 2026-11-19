import importlib.util
import json
from pathlib import Path
import sys

import torch

from iwi_harness.session import EditingSession
from iwi_harness.targets import TinyTarget
from iwi_harness.weights import WeightStore


def test_replay_preserves_tensor_term_and_later_write(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/ablate_residual_writes.py"
    spec = importlib.util.spec_from_file_location("ablate_residual_writes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_gemma_target", lambda **_: TinyTarget())
    episode, out = tmp_path / "episode", tmp_path / "ablation"
    session = EditingSession(TinyTarget(), artifact_dir=episode)
    name = "embed.weight"
    session.call("write_weights", {"tensor": name, "selection": [3], "operation": "scale", "factor": 1.25})
    captured = session.call("inspect_target", {
        "prompts": ["red", "blue"], "probe": {"kind": "residual", "layer": 0, "positions": [-1]}})
    a, b = (obs["observations"][0]["handle"] for obs in captured["outputs"])
    session.call("write_weights", {
        "tensor": name, "selection": [7], "operation": "add",
        "value": {"op": "add",
                  "left": {"op": "scale", "left": {"tensor": name, "selection": [3]}, "factor": 0.5},
                  "right": {"op": "scale", "factor": 0.25,
                            "left": {"op": "subtract", "left": {"handle": a}, "right": {"handle": b}}}}})
    session.call("write_weights", {"tensor": name, "selection": [7], "operation": "scale", "factor": 2})
    session.submit()

    monkeypatch.setattr(sys, "argv", [str(path), "--episode", str(episode), "--out", str(out)])
    module.main()
    report = json.loads((out / "replay.json").read_text())
    assert report["hash_matches_original"] and report["replay_matches_original"]
    assert report["later_write_calls_replayed"] == [4]
    assert report["other_edited_rows_verified_equal"] == 1
    assert report["final_row_difference"]["l2"] == 2 * report["immediate_row_difference"]["l2"] > 0

    target = TinyTarget()
    baseline = target.model.embed.weight.detach().clone()
    WeightStore(target).apply(out / "ablated.safetensors")
    assert torch.equal(target.model.embed.weight[3], 1.25 * baseline[3])
    assert torch.equal(target.model.embed.weight[7], 2 * (baseline[7] + 0.5 * 1.25 * baseline[3]))
