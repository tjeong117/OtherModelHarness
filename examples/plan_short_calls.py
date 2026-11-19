"""Freeze a sampled paired comparison on the qualified IWIBench training partition."""

import argparse
import hashlib
import json
from pathlib import Path

from plan_preview_benchmark import FAMILIES
from qualified_eval import REVISION, verify_install
from run_preview_benchmark import save

PROTOCOL = "iwi-public-case-defaults-dev-v1"
GUIDANCE = (
    "Edit the target to accomplish the objective and preserve other behavior. You have six model turns "
    "including submission. Use test_behavior early. When public cases are optional, omit them to run "
    "the supplied cases; otherwise pass the supplied cases explicitly. Answer checks work without "
    "token_ids. Use preview_weights to measure a proposed write before accepting it. A preview alone "
    "does not change the submitted weights. Commit a useful preview or use write_weights directly. "
    "Batch calls when appropriate and finish with submit_patch({})."
)


def rank(value):
    return hashlib.sha256((PROTOCOL + "|" + value).encode()).hexdigest()


def build(catalog, out):
    from iwibench.qualified_catalog import qualified_episode
    from iwibench.task_partitions import partitions, meaningful_digest
    from iwibench.variants import fingerprint
    from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION

    catalog = verify_install(catalog)
    metadata = json.loads((catalog / "data/qualified_catalog.json").read_text())
    admitted = {
        (e["family"], e["index"])
        for e in metadata["entries"]
        if e["baseline_admission"]["status"] == "eligible"
    }
    entries = partitions()["entries"]
    test_digests = {e["meaningful_sha256"] for e in entries if e["dataset_split"] == "test"}
    tasks, public = [], {}
    for family in FAMILIES:
        choices = [
            r
            for r in entries
            if r["family"] == family
            and r["dataset_split"] == "train"
            and not r["historical_equivalent_archive_episode_ids"]
            and r["meaningful_sha256"] not in test_digests
            and (family, r["variant_index"]) in admitted
        ]
        for row in sorted(choices, key=lambda r: rank(r["id"]))[:2]:
            ep = qualified_episode(family, row["variant_index"], GEMMA_ID)
            if (
                ep["id"] != row["id"]
                or fingerprint(ep) != row["episode_sha256"]
                or meaningful_digest(ep) != row["meaningful_sha256"]
                or ep["checkpoint_revision"] != GEMMA_REVISION
            ):
                raise ValueError("qualified catalogue regeneration mismatch")
            tasks.append(
                {
                    key: row[key]
                    for key in (
                        "id",
                        "family",
                        "variant_index",
                        "episode_sha256",
                        "parameter_sha256",
                        "meaningful_sha256",
                    )
                }
            )
            public[row["id"]] = dict(
                objective=ep["objective"],
                parameters_json=json.dumps(ep["parameters"]),
                public_practice_json=json.dumps(
                    [{"prompt": i["prompt"], "answers": i["answers"]} for i in ep["public_practice"]]
                ),
            )
    blocks = sorted(((t, r) for t in tasks for r in (1, 2)), key=lambda b: rank(f"block|{b[0]['id']}|{b[1]}"))
    first = {t["id"]: i % 2 for i, t in enumerate(sorted(tasks, key=lambda v: rank(v["id"])))}
    attempts = []
    for task, replicate in blocks:
        first_arm = (first[task["id"]] + replicate - 1) % 2
        for index in (first_arm, 1 - first_arm):
            arm = ("explicit", "short")[index]
            slug = f"{task['family']}-{task['variant_index']:03d}-r{replicate}-{arm}"
            attempts.append(
                dict(
                    order=len(attempts) + 1,
                    slug=slug,
                    arm=arm,
                    replicate=replicate,
                    task_id=task["id"],
                    family=task["family"],
                    variant_index=task["variant_index"],
                    seed=int(rank(f"decode|{task['id']}|{replicate}")[:8], 16),
                )
            )
    if len(tasks) != 8 or len(attempts) != 32:
        raise ValueError("incomplete factorial")
    if (out / "design.json").exists():
        raise FileExistsError(out / "design.json")
    out.mkdir(parents=True, exist_ok=True)
    save(out / "public-rows.json", public)
    spec = dict(
        protocol=PROTOCOL,
        dataset_revision=REVISION,
        split="train",
        tasks=tasks,
        attempts=attempts,
        public_rows_sha256=hashlib.sha256((out / "public-rows.json").read_bytes()).hexdigest(),
        train_partition_sha256=hashlib.sha256(
            (catalog / "data/task_partitions.json").read_bytes()
        ).hexdigest(),
        selection="First two SHA256(protocol|id) values per family among upstream baseline-eligible training tasks with zero historical equivalent archive IDs and zero meaningful task digests in the test split. No outcome-dependent replacement.",
        editor="Qwen/Qwen2.5-7B-Instruct",
        editor_revision="a09a35458c702b33eeacc393d103063234e8bc28",
        target=GEMMA_ID,
        target_revision=GEMMA_REVISION,
        guidance=GUIDANCE,
        opening_manifest="grouped",
        editor_decoding=dict(max_turns=6, max_new_tokens=384, temperature=0.4, top_p=0.9, top_k=0),
        public_control="Copy the following word exactly. Reply with the word only: pencil",
        target_limits=dict(max_forwards=512, max_forward_tokens=131072),
        treatment="Both arms see identical public packet and guidance. Only short sessions bind the packet's public cases as the behavior-check and weight-preview defaults. Direct writes and explicit cases/token pairs are available in both. No internal measurements.",
        outcomes="Primary is short minus explicit mean native reward across all assigned matched attempts on independently eligible tasks; no submission=0. Report fresh ineligibility without replacement, task-paired differences averaged over replicates, and zero-score count. Secondary: valid checks, defaults/preview use, explicit/nonempty submissions, tokens, preservation and compute.",
    )
    save(out / "design.json", spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.catalog, args.out)
