"""Calibrate and independently grade numerical patches from the short-call experiment."""

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from grade_qualified_suggestion import audited_outputs, generate
from iwibench.qualified_catalog import qualified_episode
from iwibench.scoring import CalibrationError, content_checker_scope, score_episode
from iwibench.terra_semantics import JudgedChecker
from iwibench.variants import fingerprint
from iwi_harness.iwibench_adapter import _alias_tokens, _cached_tokenizer
from qualified_eval import assert_deterministic, hash_file, no_remote_verdict, verify_install
from run_preview_benchmark import save


@contextmanager
def aux_gpu(path):
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def run(args):
    verify_install(args.catalog)
    spec = json.loads(args.design.read_text())
    tasks = {}
    for task in spec["tasks"]:
        ep = qualified_episode(task["family"], task["variant_index"], spec["target"])
        if ep["id"] != task["id"] or fingerprint(ep) != task["episode_sha256"]:
            raise ValueError("regenerated episode differs from frozen assignment")
        assert_deterministic(ep)
        tasks[task["id"]] = task, ep
    work, calibration = args.out / "private-cache", args.out / "calibration"
    calibration.mkdir(parents=True, exist_ok=True)
    work.mkdir(mode=0o700, exist_ok=True)
    if args.action == "calibrate":
        with aux_gpu(args.out / "aux.lock"):
            generate(list(tasks.values()), work, "pristine", None, None)
        for task, ep in tasks.values():
            base = audited_outputs(work, task, ep, None)
            try:
                metrics = score(ep, base, base)
                record = dict(status="eligible", metrics=metrics)
            except CalibrationError as exc:
                record = dict(status="ineligible", reason=str(exc))
            save(calibration / (task["episode_sha256"] + ".json"), {**task, **record})
        return
    for assignment in spec["attempts"]:
        dest = args.out / "trials" / assignment["slug"]
        if (dest / "grade.json").exists() or not (dest / "editor-result.json").exists():
            continue
        row, ep = tasks[assignment["task_id"]]
        calibration_status = json.loads((calibration / (row["episode_sha256"] + ".json")).read_text())[
            "status"
        ]
        result = json.loads((dest / "editor-result.json").read_text())
        if calibration_status != "eligible":
            grade = dict(status="ineligible", reward=None, metrics=None)
        elif result["status"] == "no_submission":
            grade = dict(status="no_submission", reward=0.0, metrics=None)
        elif result["status"] == "submitted":
            receipt = result["receipt"]
            patch = dest / "patch.safetensors"
            sha = receipt["sha256"]
            if hash_file(patch) != sha:
                raise ValueError("patch hash differs from receipt")
            with aux_gpu(args.out / "aux.lock"):
                generate([(row, ep)], work, assignment["slug"], patch, sha)
            pristine = audited_outputs(work, row, ep, None)
            edited = audited_outputs(work, row, ep, sha)
            metrics = score(ep, pristine, edited)
            grade = dict(status="scored", reward=metrics["reward"], metrics=metrics, patch_sha256=sha)
        else:
            raise ValueError("incomplete editor outcome")
        save(dest / "grade.json", {**assignment, **grade})
        print(assignment["slug"], grade["status"], grade["reward"], flush=True)


def score(ep, base, edited):
    tokens = _alias_tokens(ep, _cached_tokenizer()) if ep["family"] == "alias_acquisition" else {}
    with content_checker_scope(JudgedChecker(no_remote_verdict)):
        return score_episode(ep, base, edited, tokenization=tokens)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("calibrate", "grade"))
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
