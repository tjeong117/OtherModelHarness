"""Measure pristine eligibility of every frozen task, independently of submissions."""
import argparse
import json
from pathlib import Path

from grade_qualified_suggestion import audited_outputs, generate
from iwibench.scoring import CalibrationError, content_checker_scope, score_episode
from iwibench.terra_semantics import JudgedChecker
from iwi_harness.iwibench_adapter import _alias_tokens, _cached_tokenizer
from qualified_eval import assert_deterministic, no_remote_verdict, qualified_task
from run_preview_benchmark import save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=Path("evidence/paired-preview-benchmark/design.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    work = args.out / "private-cache"
    dest = args.out / "calibration"
    dest.mkdir(parents=True, exist_ok=True)
    done = 0
    for task in json.loads(args.design.read_text())["tasks"]:
        result_path = dest / (task["episode_sha256"] + ".json")
        if result_path.exists():
            continue
        row, ep = qualified_task(args.catalog, task["family"], task["variant_index"])
        assert_deterministic(ep)
        generate([(row, ep)], work, "pristine", None, None)
        base = audited_outputs(work, row, ep, None)
        tokens = _alias_tokens(ep, _cached_tokenizer()) if task["family"] == "alias_acquisition" else {}
        with content_checker_scope(JudgedChecker(no_remote_verdict)):
            try:
                metrics = score_episode(ep, base, base, tokenization=tokens)
                result = dict(status="eligible", pristine_reward=metrics["reward"], metrics=metrics)
            except CalibrationError as exc:
                result = dict(status="ineligible", pristine_reward=None, reason=str(exc))
        save(result_path, {**task,**result})
        done += 1
        if args.limit and done >= args.limit:
            break
