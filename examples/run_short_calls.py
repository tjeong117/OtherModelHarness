"""Run the frozen paired short-call IWIBench development experiment with local Qwen."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from run_preview_benchmark import QwenEditor, public_packet, save


def episode(design, assignment, row, editor, destination):
    import torch
    from iwi_harness.session import EditingSession, encode_json
    from iwi_harness.targets import load_gemma_target
    from iwi_harness.updater import run_updater, NoSubmission

    started = time.monotonic()
    target = load_gemma_target(local_files_only=True)
    packet = public_packet(row, target.tokenizer, design)
    defaults = packet["public_cases"] if assignment["arm"] == "short" else None
    session = EditingSession(
        target,
        artifact_dir=destination,
        public_cases=defaults,
        experiment_tools_only=True,
        detailed_writes=True,
        behavior_checks=True,
        interpretability=False,
        internal_measurements=False,
        edit_preview=True,
        **design["target_limits"],
    )
    save(destination / "public-input.json", packet)
    save(destination / "tools.json", session.tools)
    complete, turns = editor.episode(assignment, destination)
    reason = None
    try:
        run_updater(
            session,
            encode_json(packet),
            complete,
            max_turns=design["editor_decoding"]["max_turns"],
            opening_manifest=design["opening_manifest"],
        )
        status = "submitted"
    except NoSubmission as exc:
        status, reason = "no_submission", str(exc)
    usage = {k: sum(t[k] for t in turns) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    result = dict(
        **assignment,
        status=status,
        reason=reason,
        receipt=session.receipt,
        turns=len(turns),
        editor_usage=usage,
        target_usage=session.runner.usage,
        elapsed_seconds=round(time.monotonic() - started, 3),
        host_api_usd=0,
    )
    save(destination / "editor-result.json", result)
    del target, session, complete
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    design = json.loads(args.design.read_text())
    raw = (args.design.parent / "public-rows.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != design["public_rows_sha256"]:
        raise ValueError("public input mismatch")
    rows = json.loads(raw)
    dest = args.out / "trials"
    dest.mkdir(parents=True, exist_ok=True)
    editor = QwenEditor(design)
    for assignment in design["attempts"]:
        location = dest / assignment["slug"]
        if (location / "editor-result.json").is_file():
            continue
        if location.exists():
            raise RuntimeError(
                f"Interrupted attempt at {location}. Inspect and archive it before reusing the assigned seed."
            )
        print(f"Start {assignment['order']} {location.name}", flush=True)
        result = episode(design, assignment, rows[assignment["task_id"]], editor, location)
        print(
            f"Finished {assignment['order']} {result['status']} in {result['elapsed_seconds']}s", flush=True
        )


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
