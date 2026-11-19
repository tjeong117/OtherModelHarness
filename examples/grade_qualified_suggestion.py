"""Replay three immutable seed-2 patches across the qualified suggestion test family.

Provision the pinned Hub snapshot's src/iwibench, JSON data and manifest.json, then
prepend its src to PYTHONPATH. Each --generate invocation is a fresh GPU process.
The private work directory is restartable; --out writes only aggregate metrics.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

from iwibench.config import Limits
from iwibench.families import episode
from iwibench.qualified_catalog import catalog as qualified_catalog
from iwibench.task_partitions import meaningful_digest
from iwibench.runtime import NativeTarget
from iwibench.scoring import content_checker_scope, score_episode
from iwibench.terra_semantics import JudgedChecker
from iwibench.variants import fingerprint

from grade_sae_transfer import FAMILY, grade_counts, patch_info, private_items, requested_prompts, save
from qualified_eval import (
    REVISION,
    assert_deterministic,
    no_remote_verdict,
    qualified_rows,
    regenerate_row,
    verify_install,
)
from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION

ROOT = Path(__file__).resolve().parents[1]
EPISODE_DIRS = {
    "no-fit": ROOT / "evidence/astra-sae-ab-v1/suggestion-2/no-fit",
    "native": ROOT / "evidence/astra-sae-guided-v2/native-seed-2",
    "sae-guided": ROOT / "evidence/astra-sae-guided-v2/suggestion-2",
}


def load_tasks(catalog):
    tasks = []
    for row in qualified_rows(verify_install(catalog), FAMILY):
        ep = regenerate_row(row)
        assert_deterministic(ep)
        tasks.append((row, ep))
    if len(tasks) != 15 or len({row["episode_sha256"] for row, _ in tasks}) != 15:
        raise ValueError("expected the full 15-task test family")
    return tasks


def path_for(work, row, patch_hash):
    return work / (row["episode_sha256"] + "-" + (patch_hash or "pristine") + ".json")


def generate(tasks, work, label, patch, patch_hash):
    work.mkdir(parents=True, mode=0o700, exist_ok=True)
    missing = [(row, ep) for row, ep in tasks if not path_for(work, row, patch_hash).exists()]
    if not missing:
        return
    target = NativeTarget(GEMMA_ID, GEMMA_REVISION, device="cuda")
    if patch is not None:
        meta = target.apply_patch(str(patch), expected_sha256=patch_hash)
        if (meta.get("model_id"), meta.get("revision")) != (GEMMA_ID, GEMMA_REVISION):
            raise ValueError("patch metadata differs from target")
    for row, ep in missing:
        items = private_items(ep)
        if len(items) != len({item["id"] for item in items}):
            raise ValueError("non-unique item id")
        outputs = {}
        for i in range(0, len(items), 8):
            batch = items[i : i + 8]
            seeds = [ep["test_seed"] + j for j in range(i, i + len(batch))]
            generated = target.generate([x["prompt"] for x in batch], Limits().output_tokens, seeds)
            for item, g in zip(batch, generated, strict=True):
                if not isinstance(g["text"], str) or not 0 <= g["output_tokens"] <= Limits().output_tokens:
                    raise ValueError("invalid native generation")
                outputs[item["id"]] = g["text"]
        save(
            path_for(work, row, patch_hash),
            dict(
                episode_sha256=row["episode_sha256"],
                patch=patch_hash,
                outputs=outputs,
                output_sha256=fingerprint(outputs),
            ),
        )
        print(f"{label}: finished catalogue index {row['variant_index']}", flush=True)


def audited_outputs(work, row, ep, sha):
    record = json.loads(path_for(work, row, sha).read_text())
    outputs = record["outputs"]
    if (
        record["episode_sha256"] != row["episode_sha256"]
        or record["patch"] != sha
        or record["output_sha256"] != fingerprint(outputs)
        or set(outputs) != {item["id"] for item in private_items(ep)}
    ):
        raise ValueError("private cache fields or hash mismatch")
    return outputs


def report(tasks, work, patches):
    totals, per_task = {}, []
    rules = Counter()
    source_text = {arm: requested_prompts(directory) for arm, directory in EPISODE_DIRS.items()}
    source = episode(FAMILY, 2, "development", model_id=GEMMA_ID)
    practice = {hashlib.sha256(i["prompt"].encode()).hexdigest() for i in source["public_practice"]}
    for observed in source_text.values():
        observed.update(practice)
    source_targets = {hashlib.sha256(i["prompt"].encode()).hexdigest() for i in source["targets"]}
    source_parameter_hash = fingerprint(source["parameters"])
    source_meaningful_hash = meaningful_digest(source)
    source_targets_reused = 0
    source_tasks_reused = 0
    admission = {
        e["index"]: e["baseline_admission"] for e in qualified_catalog()["entries"] if e["family"] == FAMILY
    }
    baseline_control_disagreements = []
    targets_seen = set()
    overlap = Counter()
    reuse = 0
    for row, ep in tasks:
        base = audited_outputs(work, row, ep, None)
        target_hashes = {hashlib.sha256(i["prompt"].encode()).hexdigest() for i in ep["targets"]}
        source_targets_reused += len(target_hashes & source_targets)
        source_tasks_reused += meaningful_digest(ep) == source_meaningful_hash
        reuse += len(target_hashes & targets_seen)
        targets_seen.update(target_hashes)
        for name, texts in source_text.items():
            overlap[name] += len(target_hashes & texts)
        rules.update(i["rule"] for i in private_items(ep))
        arms = {}
        for name, (_, sha) in patches.items():
            edited = audited_outputs(work, row, ep, sha) if sha else base
            with content_checker_scope(JudgedChecker(no_remote_verdict)):
                arms[name] = grade_counts(score_episode(ep, base, edited))
        if admission[row["variant_index"]]["control_denominator"] != arms["pristine"]["controls_total"]:
            baseline_control_disagreements.append(
                dict(
                    variant_index=row["variant_index"],
                    published=admission[row["variant_index"]]["control_denominator"],
                    fresh=arms["pristine"]["controls_total"],
                )
            )
        per_task.append(
            dict(
                id=row["id"],
                variant_index=row["variant_index"],
                episode_sha256=row["episode_sha256"],
                arms=arms,
            )
        )
    for name in patches:
        records = [task["arms"][name] for task in per_task]
        totals[name] = {k: sum(r[k] for r in records) for k in records[0] if k != "reward"}
        totals[name]["mean_native_reward"] = mean(r["reward"] for r in records)
    reward_pairs = {}
    for name in ("native", "no-fit"):
        deltas = [task["arms"]["sae-guided"]["reward"] - task["arms"][name]["reward"] for task in per_task]
        reward_pairs[name] = dict(
            higher=sum(d > 0 for d in deltas),
            equal=sum(d == 0 for d in deltas),
            lower=sum(d < 0 for d in deltas),
            mean_difference=mean(deltas),
        )
    return dict(
        catalog_revision=REVISION,
        split="environments/test",
        family=FAMILY,
        source_seed=2,
        evaluation="Fixed patch transfer with shared pristine generations; zero new editor calls.",
        model_id=GEMMA_ID,
        checkpoint=GEMMA_REVISION,
        batch_size=8,
        max_new_tokens=Limits().output_tokens,
        rule_counts=dict(rules),
        remote_judge_calls=0,
        native_patch_hashes={name: sha for name, (_, sha) in patches.items() if sha},
        source_tool_and_practice_to_test_target_exact_text_matches=dict(overlap),
        source_private_target_exact_text_matches=source_targets_reused,
        source_meaningful_task_digest=source_meaningful_hash,
        source_meaningful_task_matches=source_tasks_reused,
        source_parameter_digest=source_parameter_hash,
        source_parameter_matches=sum(r["parameter_sha256"] == source_parameter_hash for r, _ in tasks),
        qualified_baseline_control_denominator_mismatches=baseline_control_disagreements,
        unique_test_target_prompt_strings=len(targets_seen),
        recurring_target_prompt_strings=reuse,
        guided_minus_comparator_task_rewards=reward_pairs,
        tasks=per_task,
        totals=totals,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", choices=["pristine", *EPISODE_DIRS])
    group.add_argument("--out", type=Path)
    args = p.parse_args()
    tasks = load_tasks(args.catalog.resolve())
    patches = {"pristine": (None, None), **{k: patch_info(v) for k, v in EPISODE_DIRS.items()}}
    if args.generate:
        generate(tasks, args.work, args.generate, *patches[args.generate])
    else:
        save(args.out, report(tasks, args.work, patches))


if __name__ == "__main__":
    main()
