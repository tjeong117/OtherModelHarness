"""Trusted CPU fixture orchestration, including its independent terminal grader."""
import json
from pathlib import Path
import subprocess
import sys

from safetensors import SafetensorError

from .probes import Runner
from .session import EditingSession
from .targets import tiny_sae, tiny_target
from .weights import WeightStore


def _answers(runner, prompts):
    return [o["text"] for o in runner.run(prompts)["outputs"]]


def verify_tiny(patch, expected_sha256=None):
    target = tiny_target()
    store = WeightStore(target)
    runner = Runner(target, store)
    queries = ["check dax", "check red", "check blue"]
    base = _answers(runner, queries)
    try:
        receipt = store.apply(patch, expected_sha256=expected_sha256)
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError, SafetensorError) as exc:
        return {"reward": 0.0, "status": "invalid_patch", "error": str(exc)[:240]}
    edited = _answers(runner, queries)
    gain = int(edited[0] == "yes") - int(base[0] == "yes")
    retained = sum(before == after for before, after in zip(base[1:], edited[1:]))
    retention = retained / (len(queries) - 1)
    return dict(status="scored", base=base, edited=edited, target_gain=gain, retained=retained,
                controls=len(queries)-1, retention=retention, reward=max(0, gain) * retention,
                patch_sha256=receipt["sha256"])


def run_demo(out, *, token_counter=None, literal=False):
    session = EditingSession(tiny_target(), sae=tiny_sae(), artifact_dir=out, token_counter=token_counter)
    probe = {"kind": "sae", "layer": 0, "top_k": 2}
    initial = session.call("run_target", {"prompts": ["color dax"]})
    contrast = session.call("contrast_target", {"left": ["color blue"], "right": ["color red"],
                                                 "probe": probe})
    measured = session.call("inspect_target", {"prompts": ["color red"], "probe": probe,
                                               "intervention": {"feature": 0, "scale": 0},
                                               "token_ids": [6, 7]})
    source = {"tensor": "embed.weight", "selection": [3]}
    data = session.call("read_weights", {**source, "mode": "values" if literal else "ref"})
    destination = {"tensor": "embed.weight", "selection": [5]}
    if literal:
        edit = dict(operation="replace", value={"values": data["values"]})
    else:
        edit = dict(operation="copy", source={"handle": data["handle"]})
    session.call("write_weights", {**destination, **edit})
    edited = session.call("run_target", {"prompts": ["color dax"]})
    receipt = session.submit()
    result = subprocess.run(
        [sys.executable, "-m", "iwi_harness", "verify-tiny", "--patch", str(Path(out)/"patch.safetensors"),
         "--sha256", receipt["sha256"]], check=True, capture_output=True, text=True)
    report = dict(updater="scripted", mode="literal" if literal else "compact",
                  public_before=initial["outputs"][0]["text"], public_after=edited["outputs"][0]["text"],
                  contrast=contrast["ranking"], feature_0_margin_change=measured["margin_changes"][0],
                  target_usage=session.runner.usage, receipt=receipt, fresh_verifier=json.loads(result.stdout))
    with (Path(out) / "demo.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    return report
