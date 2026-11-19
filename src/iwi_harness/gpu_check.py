"""Pinned Gemma Scope intervention and IWIBench development alias replay smoke."""

import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

import torch

from .probes import Runner
from .session import EditingSession
from .targets import GEMMA_REVISION, SCOPE_REVISION, load_gemma_sae, load_gemma_target
from .weights import WeightStore

# IWIBench alias_acquisition development seed 3, using its public practice prompt.
EMBEDDING = "model.language_model.embed_tokens.weight"
SOURCE_ID, ALIAS_ID = 26365, 122076
ALIAS_PROMPT = "Count the flat faces of a saffron. Return only the integer."
SOURCE_PROMPT = "Count the flat faces of a cube. Return only the integer."


def _digest(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


@contextmanager
def _next_logits(target):
    """Trusted instrumentation captures only the native final-position logits."""
    rows = []

    def capture(module, args, result):
        logits = result if isinstance(result, torch.Tensor) else result.logits
        rows.append(logits[0, -1].detach().cpu().contiguous())

    with target.model.register_forward_hook(capture):
        yield rows


def _row_hashes(target):
    weights = target.weights()[EMBEDDING]
    return {"source": _digest(weights[SOURCE_ID]), "alias": _digest(weights[ALIAS_ID])}


def _measure(target, run):
    prompts = {}
    for prompt in (ALIAS_PROMPT, SOURCE_PROMPT):
        with _next_logits(target) as rows:
            output = run({"prompts": [prompt], "max_new_tokens": 4})["outputs"][0]
        if len(rows) != len(output["token_ids"]):
            raise RuntimeError("Expected one native logit vector for each generated token")
        scores, ids = rows[0].float().topk(2)
        prompts[prompt] = {
            "token_ids": output["token_ids"], "text": output["text"],
            "first_top_two": [{"token_id": i, "logit": x} for i, x in zip(ids.tolist(), scores.tolist())],
            "logit_sha256_by_step": [_digest(row) for row in rows],
        }
    return {"prompts": prompts, "embedding_row_sha256": _row_hashes(target)}


def _inspect_change(session, arguments, feature, scale):
    target = session.runner.target
    with _next_logits(target) as rows:
        out = session.call("inspect_target", {**arguments, "intervention": {"feature": feature, "scale": scale}})
    if len(rows) != 2:
        raise RuntimeError("Expected baseline and intervention forwards")
    baseline, changed = rows
    delta = changed.float() - baseline.float()
    return {
        "scale": scale,
        "baseline_logit_sha256": _digest(baseline), "intervention_logit_sha256": _digest(changed),
        "changed_logits": int(delta.count_nonzero()), "vocab_size": delta.numel(),
        "max_abs_logit_change": float(delta.abs().max()), "logit_delta_l2": float(delta.norm()),
        "baseline_scores": out["baseline_outputs"][0]["scores"],
        "intervention_scores": out["outputs"][0]["scores"], "margin_change": out["margin_changes"][0],
        "cost": out["cost"],
    }


def _discover(target, sae, out):
    if SOURCE_ID not in target.encode(SOURCE_PROMPT) or ALIAS_ID not in target.encode(ALIAS_PROMPT):
        raise ValueError("The pinned public prompts must contain the expected embedding tokens")
    session = EditingSession(target, artifact_dir=out, sae=sae)
    run = lambda args: session.call("run_target", args)
    pristine = _measure(target, run)
    natural = pristine["prompts"][ALIAS_PROMPT]
    probe = {"kind": "sae", "layer": sae.layer, "positions": [-1]}
    contrast = session.call("contrast_target", {"left": [SOURCE_PROMPT], "right": [ALIAS_PROMPT], "probe": probe})
    args = {"prompts": [ALIAS_PROMPT], "probe": probe, "token_ids": [t["token_id"] for t in natural["first_top_two"]]}
    observed = session.call("inspect_target", args)["outputs"][0]["observations"][0]
    feature = observed["top"][0]
    if feature["value"] <= 0:
        raise RuntimeError("No active SAE feature at the public alias prompt's prediction position")
    identity = _inspect_change(session, args, feature["index"], 1)
    suppressed = _inspect_change(session, args, feature["index"], 0)
    baseline_digest = natural["logit_sha256_by_step"][0]
    if (identity["baseline_logit_sha256"] != baseline_digest
            or identity["intervention_logit_sha256"] != baseline_digest
            or suppressed["baseline_logit_sha256"] != baseline_digest):
        raise RuntimeError("Unmodified and identity-intervention native logits differ")
    with _next_logits(target) as rows:
        run({"prompts": [ALIAS_PROMPT], "max_new_tokens": 1})
    if _digest(rows[0]) != baseline_digest or target.blocks[sae.layer]._forward_hooks:
        raise RuntimeError("The temporary intervention persisted after its tool call")

    write = session.call("write_weights", {
        "tensor": EMBEDDING, "selection": [ALIAS_ID], "operation": "copy",
        "source": {"tensor": EMBEDDING, "selection": [SOURCE_ID]},
    })
    if write["changed_parameters"] == 0:
        raise RuntimeError("The requested alias row copy was already present")
    edited = _measure(target, run)
    hashes = edited["embedding_row_sha256"]
    if hashes["source"] != hashes["alias"] or hashes["source"] != pristine["embedding_row_sha256"]["source"]:
        raise RuntimeError("The edited embedding row does not match its pristine source")
    receipt = session.submit()
    return {
        "pristine": pristine, "edited": edited, "write": write, "receipt": receipt,
        "sae": session.info["sae"], "target_id": target.model_id, "target_revision": target.revision,
        "contrast": {"ranking": contrast["ranking"], "sign": contrast["sign"], "cost": contrast["cost"]},
        "feature": feature, "identity": identity, "suppression": suppressed,
        "cleanup_logit_sha256": baseline_digest, "discovery_target_usage": session.runner.usage.copy(),
    }


def _replay(target, path, digest):
    store = WeightStore(target)
    applied = store.apply(path, expected_sha256=digest)
    runner = Runner(target, store)
    measurement = _measure(target, lambda args: runner.run(**args))
    return {"pid": os.getpid(), "applied": applied, "measurement": measurement, "target_usage": runner.usage}


def _peak(device):
    return torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None


def gpu_smoke(out, *, local_files_only=False, device="cuda") -> dict:
    """Create the numeric patch and demand exact native-logit replay in a new interpreter."""
    out = Path(out).resolve()
    start = perf_counter()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    discovery = _discover(load_gemma_target(device=device, local_files_only=local_files_only),
                          load_gemma_sae(device=device, local_files_only=local_files_only), out)
    peak = _peak(device)
    # EditingSession contains bound tool methods. Release their cycle before the second load.
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    # The trusted replay receives only the submitted numeric artifact and its expected digest.
    with tempfile.TemporaryDirectory(prefix="iwi-clean-replay-") as stage:
        patch = Path(stage) / "patch.safetensors"
        shutil.copyfile(out / patch.name, patch)
        command = [sys.executable, "-m", "iwi_harness.gpu_check", "replay",
                   str(patch), "--sha256", discovery["receipt"]["sha256"], "--device", device]
        replay = json.loads(subprocess.check_output(command, text=True))
    if replay["measurement"] != discovery["edited"]:
        raise RuntimeError("Fresh replay differs in edited slice, greedy tokens, or native logits")
    report = {
        "experiment": "scripted IWIBench development seed 3 alias copy and Gemma Scope 2 suppression",
        "target_revision": GEMMA_REVISION, "dictionary_revision": SCOPE_REVISION,
        "versions": {package: version(package) for package in ("torch", "transformers", "huggingface-hub", "safetensors")},
        "device": device, "gpu_name": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        "elapsed_seconds": perf_counter() - start, "discovery_pid": os.getpid(),
        "discovery_peak_allocated_bytes": peak, "discovery": discovery, "replay": replay,
        "exact_replay": True,
    }
    with (out / "gpu-smoke.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    discover = commands.add_parser("smoke")
    discover.add_argument("out")
    discover.add_argument("--local-files-only", action="store_true")
    discover.add_argument("--device", default="cuda")
    replay = commands.add_parser("replay")
    replay.add_argument("patch")
    replay.add_argument("--sha256", required=True)
    replay.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase == "replay":
        target = load_gemma_target(device=args.device, local_files_only=True)
        report = _replay(target, args.patch, args.sha256)
        report["peak_allocated_bytes"] = _peak(args.device)
    else:
        report = gpu_smoke(args.out, device=args.device, local_files_only=args.local_files_only)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
