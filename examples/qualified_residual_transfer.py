"""Replay both recorded residual-derived edits on every qualified false-suggestion test task.

Prepend the pinned qualified package's src directory to PYTHONPATH. Use a private
--work directory for the fresh target generations. Run --generate once per variant
and for pristine, then use --out to write a shareable aggregate report.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from grade_qualified_suggestion import audited_outputs, generate, load_tasks
from grade_residual_ablation import transitions
from grade_sae_transfer import FAMILY, grade_counts, prompt_hashes, requested_prompts, save
from qualified_eval import REVISION, no_remote_verdict
from iwi_harness.iwibench_adapter import IWIBENCH_REVISION
from iwi_harness.weights import NATIVE_FORMAT
from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION
from iwibench.config import Limits
from iwibench.families import episode
from iwibench.scoring import content_checker_scope, score_episode
from iwibench.terra_semantics import JudgedChecker
from iwibench.variants import fingerprint

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "evidence/astra-residual-ablation"
SEEDS = (2, 5)
VARIANTS = ("full", "ablated")

# These full-episode fingerprints were also checked in the historical IWIBench checkout.
SOURCE_EPISODE_SHA = {
    2: "dcb7725d4c7a45976783e84b7a34d0e991a513c0a9ccb02394aa409c01ee978e",
    5: "bcd32c9ab3b7cab6f01f4002c123757938ddb7d34994421bd2952c831129df3a",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_pair(seed):
    folder = SOURCES / f"suggestion-{seed}"
    full = folder / "patch.safetensors"
    full_sha = digest(full)
    protocol = json.loads((folder / "protocol.json").read_text())
    receipt = json.loads((folder / "receipt.json").read_text())
    ablated = folder / "ablated.safetensors"
    ablated_sha = digest(ablated)
    replay = json.loads((folder / "replay.json").read_text())
    study = json.loads((folder / "comparison.json").read_text())
    if not (
        (protocol["family"], protocol["seed"], protocol["iwibench_revision"])
        == (FAMILY, seed, IWIBENCH_REVISION)
        and (protocol["session"]["model_id"], protocol["session"]["revision"]) == (GEMMA_ID, GEMMA_REVISION)
        and (receipt["sha256"], receipt["format"], receipt["bytes"])
        == (full_sha, NATIVE_FORMAT, full.stat().st_size)
        and replay["trace_sha256"] == digest(folder / "trajectory.jsonl")
        and replay["original_sha256"] == replay["replayed_sha256"] == study["original_sha256"] == full_sha
        and replay["ablated_sha256"] == study["ablated_sha256"] == ablated_sha != full_sha
        and replay["replay_matches_original"]
        and replay["hash_matches_original"]
        and study["seed"] == seed
        and study["original_grade_reproduced"]
    ):
        raise ValueError(f"Source seed {seed} does not match the recorded one-term ablation")
    return {"full": (full, full_sha), "ablated": (ablated, ablated_sha)}


def source_overlap(seed, target_texts):
    folder = SOURCES / f"suggestion-{seed}"
    source = episode(FAMILY, seed, "development", model_id=GEMMA_ID)
    if fingerprint(source) != SOURCE_EPISODE_SHA[seed]:
        raise ValueError(f"Historical source episode {seed} failed verification")
    private = prompt_hashes(source["targets"])
    practice = prompt_hashes(source["public_practice"])
    calls = requested_prompts(folder)
    return {
        "episode_sha256": SOURCE_EPISODE_SHA[seed],
        "source_private_target_strings": len(private),
        "source_practice_strings": len(practice),
        "source_tool_query_strings": len(calls),
        "test_targets_matching_source_private_targets": len(target_texts & private),
        "test_targets_matching_source_practice": len(target_texts & practice),
        "test_targets_matching_source_tool_queries": len(target_texts & calls),
    }


def sum_counts(records):
    return {k: sum(r[k] for r in records) for k in records[0]}


def pooled_grade(records):
    return {
        **sum_counts([{k: v for k, v in r.items() if k != "reward"} for r in records]),
        "mean_reward": mean(r["reward"] for r in records),
    }


def measure_pair(work, row, ep, base, pair):
    outputs = {arm: audited_outputs(work, row, ep, sha) for arm, (_, sha) in pair.items()}
    grades = {arm: grade_counts(score_episode(ep, base, texts)) for arm, texts in outputs.items()}
    targets = ep["targets"]
    controls = [
        item for group in ("nearby_controls", "general_controls", "override_controls") for item in ep[group]
    ]
    groups = {
        "targets": targets,
        "true_suggestion_targets": [i for i in targets if i["suggestion_correct"]],
        "false_suggestion_targets": [i for i in targets if not i["suggestion_correct"]],
        "controls": controls,
    }
    paired = {key: transitions(items, outputs["full"], outputs["ablated"]) for key, items in groups.items()}
    t = paired["targets"]
    for arm in VARIANTS:
        if t["both_correct"] + t[arm + "_only_correct"] != grades[arm]["edited_correct"]:
            raise ValueError("Paired item counts disagree with the official scorer")
    return {"arms": grades, "paired": paired}


def report(tasks, work, pairs):
    target_hashes = [prompt_hashes(ep["targets"]) for _, ep in tasks]
    target_texts = set().union(*target_hashes)
    rows = []
    for row, ep in tasks:
        base = audited_outputs(work, row, ep, None)
        with content_checker_scope(JudgedChecker(no_remote_verdict)):
            pristine = grade_counts(score_episode(ep, base, base))
            results = {seed: measure_pair(work, row, ep, base, pairs[seed]) for seed in SEEDS}
        rows.append(
            {
                "id": row["id"],
                "variant_index": row["variant_index"],
                "episode_sha256": row["episode_sha256"],
                "pristine": pristine,
                "sources": results,
            }
        )
    pooled = {}
    for seed in SEEDS:
        records = [row["sources"][seed] for row in rows]
        rewards = [r["arms"]["full"]["reward"] - r["arms"]["ablated"]["reward"] for r in records]
        pooled[seed] = {
            "arms": {arm: pooled_grade([r["arms"][arm] for r in records]) for arm in VARIANTS},
            "paired": {k: sum_counts([r["paired"][k] for r in records]) for k in records[0]["paired"]},
            "full_minus_ablated_reward": {
                "mean": mean(rewards),
                "higher_tasks": sum(x > 0 for x in rewards),
                "tied_tasks": sum(x == 0 for x in rewards),
                "lower_tasks": sum(x < 0 for x in rewards),
            },
        }
    return {
        "selection": "Both recorded full-tool development edits with a residual operand, source seeds 2 and 5, and all 15 qualified test tasks in the same family. Both original and ablated patches are frozen.",
        "model_id": GEMMA_ID,
        "checkpoint": GEMMA_REVISION,
        "source_iwibench_revision": IWIBENCH_REVISION,
        "catalog_revision": REVISION,
        "split": "test",
        "family": FAMILY,
        "task_count": len(rows),
        "unique_target_prompts": len(target_texts),
        "recurring_target_prompt_occurrences": sum(len(ep["targets"]) for _, ep in tasks)
        - len(target_texts),
        "batch_size": 8,
        "max_new_tokens": Limits().output_tokens,
        "new_editor_calls": 0,
        "remote_judge_calls": 0,
        "patch_hashes": {
            seed: {arm: sha for arm, (_, sha) in pair.items()} for seed, pair in pairs.items()
        },
        "overlap": {seed: source_overlap(seed, target_texts) for seed in SEEDS},
        "paired_counts_definition": "Compared on identical items and generation seeds. Full-only correct counts fixes from including the residual term; ablated-only correct counts regressions. Full and ablated use identical recorded weight operations apart from that term.",
        "pristine": pooled_grade([row["pristine"] for row in rows]),
        "pooled_by_source": pooled,
        "tasks": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--work",
        type=Path,
        required=True,
        help="Private generation cache shared with the pristine verifier",
    )
    tasks = parser.add_mutually_exclusive_group(required=True)
    labels = {f"{seed}-{arm}": (seed, arm) for seed in SEEDS for arm in VARIANTS}
    tasks.add_argument("--generate", choices=["pristine", *labels])
    tasks.add_argument("--out", type=Path)
    args = parser.parse_args()
    family = load_tasks(args.catalog)
    pairs = {seed: source_pair(seed) for seed in SEEDS}
    if args.generate:
        path, sha = (None, None)
        if args.generate != "pristine":
            seed, arm = labels[args.generate]
            path, sha = pairs[seed][arm]
        generate(family, args.work, args.generate, path, sha)
    else:
        save(args.out, report(family, args.work, pairs))


if __name__ == "__main__":
    main()
