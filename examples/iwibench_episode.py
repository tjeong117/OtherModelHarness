"""Run one extended-tool development episode with a local or hosted updater."""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path

import torch

from iwi_harness import EditingSession
from iwi_harness.iwibench_adapter import grade_patch, public_task
from iwi_harness.targets import load_gemma_sae, load_gemma_target
from iwi_harness.updater import NoSubmission, chat_completions, run_updater


@contextmanager
def endpoint(args):
    if args.local_qwen:
        from qwen_http_trial import MODEL, local_qwen_endpoint
        with local_qwen_endpoint(args.out) as (url, _, _turns):
            yield chat_completions(url, MODEL)
    else:
        yield chat_completions(args.base_url, args.model, os.getenv("UPDATER_API_KEY"))


def run(args):
    task = public_task(args.family, args.seed)
    target = load_gemma_target(local_files_only=True)
    sae = None if args.compact_only else load_gemma_sae(local_files_only=True)
    session = EditingSession(target, artifact_dir=args.out, sae=sae, interpretability=not args.compact_only,
                             detailed_writes=args.detailed_writes)
    # Only this public packet and session tool responses reach the updater.
    with endpoint(args) as complete:
        try:
            receipt = run_updater(session, json.dumps(task), complete, max_turns=args.max_turns, guide=args.guide)
        except NoSubmission as exc:
            receipt = None
            result = {"episode_id": task["id"], "status": "no_submission", "reward": 0.0,
                      "reason": str(exc), "metrics": None}
    del session, target, sae
    gc.collect()
    torch.cuda.empty_cache()
    if receipt is not None:
        result = grade_patch(args.out / "patch.safetensors", family=args.family, seed=args.seed,
                             expected_sha256=receipt["sha256"])
    (args.out / "grade.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["alias_acquisition", "resistance_to_false_suggestions"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    host = parser.add_mutually_exclusive_group(required=True)
    host.add_argument("--local-qwen", action="store_true", help="use the cached Qwen2.5-7B-Instruct on the same GPU")
    host.add_argument("--base-url")
    parser.add_argument("--model", help="required for an external endpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compact-only", action="store_true")
    parser.add_argument("--detailed-writes", action="store_true", help="expose the full write validation schema")
    parser.add_argument("--guide", action="store_true",
                        help="show a four-tensor manifest preview, a slice example and a last-turn reminder")
    parser.add_argument("--max-turns", type=int)
    args = parser.parse_args()
    if args.base_url and not args.model or args.local_qwen and args.model:
        parser.error("--model belongs to --base-url and must be provided with it")
    if args.max_turns is None:
        args.max_turns = 10 if args.local_qwen else 32
    if args.max_turns < 1 or args.local_qwen and args.max_turns > 10:
        parser.error("turn count must be positive; the local server supports up to 10 turns")
    run(args)
