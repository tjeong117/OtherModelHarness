"""Audit matched hosted-editor pairs and their independent qualified calibrations."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from analyze_preview_benchmark import audit_trial, digest, load
from run_preview_benchmark import save


def summarize(root, calibration):
    spec = load(root / "design.json")
    rows, exclusions, eligible = [], {}, {}
    for task in spec["tasks"]:
        record = load(calibration / (task["episode_sha256"] + ".json"))
        if record["id"] != task["id"] or record["episode_sha256"] != task["episode_sha256"]:
            raise ValueError("independent calibration identity mismatch")
        if record["status"] not in ("eligible", "ineligible"):
            raise ValueError("unfinished independent calibration")
        eligible[task["id"]] = record["status"] == "eligible"

    slugs = set()
    for trial in spec["attempts"]:
        if trial["slug"] in slugs:
            raise ValueError("duplicate assignment")
        slugs.add(trial["slug"])
        folder = root / "trials" / trial["slug"]
        if not (folder / "editor-result.json").exists():
            exclusions[trial["slug"]] = "not run"
            continue
        result = load(folder / "editor-result.json")
        if result["status"] not in ("submitted", "no_submission"):
            exclusions[trial["slug"]] = result["status"]
            continue
        row = audit_trial(folder, trial, hosted=True)
        if row["submitted"] and (row["grade"] == "scored") != eligible[row["task_id"]]:
            raise ValueError("independent calibration and grade disagree")
        row["usd_upper_bound"] = result["usd_upper_bound"]
        rows.append(row)

    paired = {}
    for task in spec["tasks"]:
        task_id = task["id"]
        matched = {r["arm"]: r for r in rows if r["task_id"] == task_id}
        if set(matched) != {"C", "D"}:
            continue
        c, d = matched["C"], matched["D"]
        if c["public_input_sha256"] != d["public_input_sha256"]:
            raise ValueError("matched task packets differ")
        if eligible[task_id]:
            paired[task_id] = dict(C=c["reward"], D=d["reward"], delta=d["reward"] - c["reward"])
            for metric in ("target_gain", "retention"):
                if c[metric] is not None and d[metric] is not None:
                    paired[task_id][metric + "_delta"] = d[metric] - c[metric]

    arms = {}
    counts = (
        "successful_behavior_calls",
        "internal_requests",
        "successful_internal_calls",
        "internal_changed_public_answers",
        "internal_changed_exact_matches",
        "matching_recipe_attempted_successfully",
        "matching_recipe_applied",
        "preview_requests",
        "successful_previews",
        "successful_commits",
        "tool_errors",
    )
    for arm in "CD":
        group = [r for r in rows if r["arm"] == arm]
        if not group:
            continue
        if len({r["tool_menu_sha256"] for r in group}) != 1:
            raise ValueError("tool schema changed during hosted cohort")
        scored = [r for r in group if r["grade"] == "scored"]
        total_retained = sum(r["retained"] for r in scored)
        controls = sum(r["retention_denominator"] for r in scored)
        arms[arm] = dict(
            assigned=sum(t["arm"] == arm for t in spec["attempts"]),
            audited=len(group),
            submitted=sum(r["submitted"] for r in group),
            nonempty=sum(bool(r["changed_parameters"]) for r in group),
            scored=len(scored),
            mean_paired_reward=mean(p[arm] for p in paired.values()) if paired else None,
            mean_gain_on_scored=mean(r["target_gain"] for r in scored) if scored else None,
            controls_retained_on_scored=total_retained,
            controls_on_scored=controls,
            pooled_retention_on_scored=total_retained / controls if controls else None,
            mean_editor_tokens=mean(r["total_tokens"] for r in group),
            mean_editor_input_tokens=mean(r["prompt_tokens"] for r in group),
            mean_editor_output_tokens=mean(r["completion_tokens"] for r in group),
            mean_target_forwards=mean(r["target_forwards"] for r in group),
            mean_target_forward_tokens=mean(r["target_forward_tokens"] for r in group),
            mean_elapsed_seconds=mean(r["elapsed_seconds"] for r in group),
            usd_upper_bound=sum(r["usd_upper_bound"] for r in group),
            episodes={k: sum(bool(r[k]) for r in group) for k in counts},
            calls={
                k: sum(r[k] for r in group)
                for k in counts
                if k.endswith(("calls", "requests", "previews", "commits", "errors"))
            },
            qualified_transitions_on_scored={
                label: dict(
                    sum((Counter(r["qualified_transitions"].get(label, {})) for r in scored), Counter())
                )
                for label in sorted({k for r in scored for k in r["qualified_transitions"]})
            },
        )
    deltas = [p["delta"] for p in paired.values()]
    contrast = {}
    if deltas:
        bootstrap = np.random.default_rng(29631).choice(deltas, (20000, len(deltas))).mean(axis=1)
        contrast = dict(
            pairs=len(deltas),
            mean=mean(deltas),
            descriptive_bootstrap_95=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            wins=sum(d > 0 for d in deltas),
            ties=deltas.count(0),
            losses=sum(d < 0 for d in deltas),
        )
        for metric in ("target_gain", "retention"):
            values = [p[metric + "_delta"] for p in paired.values() if metric + "_delta" in p]
            contrast[metric + "_scored_pairs"] = len(values)
            contrast["mean_" + metric + "_delta_on_scored_pairs"] = mean(values) if values else None
    ledger = load(root / "budget.json")
    cap, spent = ledger["cap_microdollars"] / 1e6, ledger["used_or_reserved_microdollars"] / 1e6
    if cap != spec["budget_usd"] or spent > cap:
        raise ValueError("hosted spending exceeds cap")
    if not exclusions and abs(sum(r["usd_upper_bound"] for r in rows) - spent) > 1e-6:
        raise ValueError("hosted cohort and durable ledger disagree")
    return dict(
        design_sha256=digest(root / "design.json"),
        editor=spec["editor"],
        assigned=len(slugs),
        eligible_tasks=sum(eligible.values()),
        calibrated=eligible,
        excluded_attempts=exclusions,
        cap_usd=cap,
        conservative_usd=spent,
        arms=arms,
        D_minus_C=contrast,
        pairs=paired,
        trials=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    save(args.out, summarize(args.root, args.calibration))
