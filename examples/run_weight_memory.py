"""Produce sequential enrollment patches and evaluate fresh, chat-isolated memory reads."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from iwi_harness.probes import Runner
from iwi_harness.targets import load_gemma_target
from iwi_harness.weights import WeightStore
from weight_memory_cases import DESIGN, cases, contextual_row, correct, fetch_note, load


EMBEDDING = "model.language_model.embed_tokens.weight"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    staging.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def gpu_lock(path):
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def fresh(design):
    start = time.monotonic()
    target = load_gemma_target(local_files_only=True)
    if (target.model_id, target.revision) != (design["target"], design["revision"]):
        raise ValueError("model identity differs from frozen design")
    torch.cuda.synchronize()
    return target, time.monotonic() - start


def write_sequence(args, design, sequence, fingerprint):
    out = args.out / sequence["id"]
    final = out / "enrollment-results.json"
    if final.exists():
        record = load(final)
        if record["design_sha256"] != fingerprint:
            raise ValueError("enrollment design changed")
        for row in record["stages"]:
            if sha(out / f'stage-{row["stage"]}/patch.safetensors') != row["receipt"]["sha256"]:
                raise ValueError("enrollment patch digest changed")
        return
    target, load_seconds = fresh(design)
    store = WeightStore(target)
    stages = []
    for stage, binding in enumerate(sequence["associations"], 1):
        started = time.monotonic()
        dest = out / f"stage-{stage}"
        dest.mkdir(parents=True, exist_ok=True)
        alias_id = contextual_row(target.tokenizer, binding["alias"])
        country_id = contextual_row(target.tokenizer, binding["country"])
        if alias_id == country_id:
            raise ValueError("alias and country rows coincide")
        write = dict(tensor=EMBEDDING, selection=[alias_id], operation="copy",
                     source=dict(tensor=EMBEDDING, selection=[country_id]))
        edited = store.write(**write)
        receipt = store.export(dest / "patch.safetensors")
        receipt.pop("path")
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        save(dest / "receipt.json", receipt)
        def measured_payload(obj):
            wire = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            return len(target.tokenizer.encode(wire, add_special_tokens=False))
        stages.append(dict(stage=stage, alias=binding["alias"], country=binding["country"], write=write,
                           write_result=edited, receipt=receipt, enrollment_seconds=elapsed,
                           tool_payload_tokens_gemma_tokenizer=measured_payload(write) + measured_payload(edited),
                           note_storage_tokens=len(target.encode_text(design["note_template"].format(**binding).strip())),
                           llm_editor_input_tokens=0, llm_editor_output_tokens=0, target_forwards_for_edit=0))
        print(sequence["id"], "enroll", stage, receipt["sha256"][:10], flush=True)
    save(final, dict(design_sha256=fingerprint, model_load_seconds=load_seconds, stages=stages))


def score_queries(design, sequence, stage, arm, target, store, cache):
    runner = Runner(target, store, max_forwards=65536, max_forward_tokens=16_777_216)
    rows = []
    active = sequence["associations"][:stage]
    for item in cases(design, sequence, stage):
        rowid = f'{sequence["id"]}/{stage}/{item["id"]}'
        note, retrieved = ("", None)
        if arm == "retrieval":
            note, retrieved = fetch_note(item["prompt"], active, design["note_template"])
        prompt = note + item["prompt"]
        prefix_tokens = len(target.encode(prompt))
        if prompt in cache:
            observed, origin = cache[prompt]
        else:
            started = time.monotonic()
            result = runner.run([prompt], max_new_tokens=design["max_new_tokens"])
            torch.cuda.synchronize()
            output = result["outputs"][0]
            observed = dict(text=output["text"], output_ids=output["token_ids"],
                            output_tokens=len(output["token_ids"]), input_tokens=prefix_tokens,
                            forwards=result["cost"]["forwards"], forward_tokens=result["cost"]["forward_tokens"],
                            generation_seconds=time.monotonic() - started)
            cache[prompt] = (observed, rowid)
            origin = rowid
        rows.append(dict(**item, sequence=sequence["id"], stage=stage, row_id=rowid, arm=arm,
                         retrieved_alias=retrieved, note=note,
                         note_added_prompt_tokens=prefix_tokens - len(target.encode(item["prompt"])),
                         **observed, physical_origin=origin, correct=correct(observed["text"], item["answers"])))
    return rows


def results_path(args, sequence, stage):
    return args.out / args.arm / sequence["id"] / f"stage-{stage}.json"


def patch_path(args, sequence, stage):
    return args.patch_root / sequence["id"] / f"stage-{stage}"


def verify_output(record, sequence, stage, arm, fingerprint, expected_sha):
    if (record["design_sha256"], record["sequence"], record["stage"], record["arm"],
        record["patch_sha256"]) != (fingerprint, sequence["id"], stage, arm, expected_sha):
        raise ValueError("existing result is from a different design or patch")


def evaluate_on_target(args, design, seq, stage, fingerprint, target, store, cache, patch_sha, calibration):
    path = results_path(args, seq, stage)
    if path.exists():
        verify_output(load(path), seq, stage, args.arm, fingerprint, patch_sha)
        return
    rows = score_queries(design, seq, stage, args.arm, target, store, cache)
    save(path, dict(design_sha256=fingerprint, sequence=seq["id"], stage=stage, arm=args.arm,
                    checkpoint=target.revision, patch_sha256=patch_sha, **calibration, rows=rows))
    print(args.arm, seq["id"], stage, "completed", len(rows), flush=True)


def evaluate_patched(args, design, seq, stage, fingerprint):
    artifact = patch_path(args, seq, stage)
    receipt = load(artifact / "receipt.json")
    digest = receipt["sha256"]
    if sha(artifact / "patch.safetensors") != digest:
        raise ValueError("patch does not match receipt")
    dest = results_path(args, seq, stage)
    if dest.exists():
        verify_output(load(dest), seq, stage, args.arm, fingerprint, digest)
        return
    target, load_seconds = fresh(design)
    store = WeightStore(target)
    start = time.monotonic()
    applied = store.apply(artifact / "patch.safetensors", expected_sha256=digest)
    torch.cuda.synchronize()
    apply_seconds = time.monotonic() - start
    evaluate_on_target(args, design, seq, stage, fingerprint, target, store, {}, digest,
                       dict(checkpoint_load_seconds=load_seconds, patch_load_seconds=apply_seconds,
                            replay=applied))


def evaluate_reference(args, design, seq, fingerprint):
    target, load_seconds = fresh(design)
    store = WeightStore(target)
    # A reference target is immutable. Logical costs still include every scheduled read.
    cache = {}
    for stage in range(1, len(seq["associations"]) + 1):
        evaluate_on_target(args, design, seq, stage, fingerprint, target, store, cache, None,
                           dict(checkpoint_load_seconds=load_seconds if stage == 1 else 0, patch_load_seconds=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["enroll", "evaluate"])
    p.add_argument("--design", type=Path, default=DESIGN)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=["pristine", "retrieval", "edited", "qwen"])
    p.add_argument("--patch-root", type=Path)
    p.add_argument("--wait-for-patches", type=int, metavar="SECONDS", default=0,
                   help="wait outside the GPU lock for each externally produced stage receipt")
    p.add_argument("--gpu-lock", type=Path)
    args = p.parse_args()
    if args.action == "evaluate" and (not args.arm or args.arm in ("edited", "qwen") and not args.patch_root):
        p.error("supply --arm, and --patch-root for a patched arm")
    design, fingerprint = load(args.design), sha(args.design)
    for seq in design["sequences"]:
        if args.action == "enroll":
            jobs = [(write_sequence, (args, design, seq, fingerprint))]
        elif args.arm in ("edited", "qwen"):
            jobs = [(evaluate_patched, (args, design, seq, stage, fingerprint))
                    for stage in range(1, len(seq["associations"]) + 1)]
        else:
            jobs = [(evaluate_reference, (args, design, seq, fingerprint))]
        for run, params in jobs:
            if run is evaluate_patched and args.wait_for_patches:
                deadline = time.monotonic() + args.wait_for_patches
                directory = patch_path(args, seq, params[3])
                while not all((directory / name).is_file() for name in ("receipt.json", "patch.safetensors")):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"still waiting for {directory}")
                    time.sleep(3)
            with gpu_lock(args.gpu_lock):
                run(*params)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
