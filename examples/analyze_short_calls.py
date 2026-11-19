"""Audit the paired local Qwen default-cases experiment from saved public traces."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
import tarfile

import numpy as np

from run_preview_benchmark import save


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def complete_calls_in_failed_generation(generation):
    if not generation.get("parse_error"):
        return 0
    valid = 0
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", generation["raw"], flags=re.DOTALL):
        try:
            call = json.loads(block)
        except ValueError:
            continue
        if isinstance(call, dict) and set(call) == {"name", "arguments"}:
            valid += isinstance(call["name"], str) and isinstance(call["arguments"], dict)
    return valid


def count_argument_tokens(root):
    """Count one equivalent full public check per task with the pinned editor tokenizer."""
    from transformers import AutoTokenizer

    design = read(root / "design.json")
    rows = read(root / "public-rows.json")
    tokenizer = AutoTokenizer.from_pretrained(
        design["editor"], revision=design["editor_revision"], local_files_only=True
    )
    counts = []
    for task in design["tasks"]:
        row = rows[task["id"]]
        cases = [
            dict(slice="improve", prompt=p["prompt"], answers=p["answers"])
            for p in json.loads(row["public_practice_json"])
        ]
        cases.append(dict(slice="preserve", prompt=design["public_control"], answers=["pencil"]))
        values = {}
        for arm, args in (("explicit", {"cases": cases}), ("short", {})):
            serialized = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
            values[arm] = len(tokenizer.encode(serialized, add_special_tokens=False))
        counts.append(
            dict(task_id=task["id"], family=task["family"], variant_index=task["variant_index"], **values)
        )
    return dict(
        editor=design["editor"],
        revision=design["editor_revision"],
        scope="One synthesized test_behavior call per task using exactly the declared public cases. "
        "Counts include compact JSON arguments only. Actual trajectory token counts are in summary.json.",
        tasks=counts,
    )


def audit(root):
    from diagnose_preview_editor import audit_trial as audit_decode

    design = read(root / "design.json")
    env = read(root / "environment.json")
    if (
        env["design_sha256"] != digest(root / "design.json")
        or design["public_rows_sha256"] != digest(root / "public-rows.json")
        or env["launch_archive_sha256"] != digest(root / "launch-source.tar.gz")
    ):
        raise ValueError("frozen inputs or launch archive differ")
    with tarfile.open(root / "launch-source.tar.gz") as archive:
        for name, expected in env["sources"].items():
            with archive.extractfile(name) as stream:
                if sha256(stream.read()).hexdigest() != expected:
                    raise ValueError(f"launch source hash differs for {name}")
    tasks = {t["id"]: t for t in design["tasks"]}
    calibrations = {}
    for task_id, task in tasks.items():
        calibration = read(root / "calibration" / (task["episode_sha256"] + ".json"))
        if any(calibration.get(k) != v for k, v in task.items()):
            raise ValueError("calibration identity differs from frozen selection")
        calibrations[task_id] = calibration
    menus, agreed_packets = {}, {}
    seen, seeds = set(), {}
    records = []
    for assigned in design["attempts"]:
        task_id, arm, replicate = (assigned[k] for k in ("task_id", "arm", "replicate"))
        key = (task_id, replicate, arm)
        if key in seen or arm not in ("explicit", "short") or replicate not in (1, 2):
            raise ValueError("duplicate or out-of-range assignment")
        seen.add(key)
        if seeds.setdefault((task_id, replicate), assigned["seed"]) != assigned["seed"]:
            raise ValueError("replicate seeds differ")
        trial = root / "trials" / assigned["slug"]
        result = read(trial / "editor-result.json")
        grade = read(trial / "grade.json")
        if any(result[k] != v or grade[k] != v for k, v in assigned.items()):
            raise ValueError("assignment mismatch")
        eligible = calibrations[task_id]["status"] == "eligible"
        submitted = result["status"] == "submitted"
        if eligible:
            required_status = "scored" if submitted else "no_submission"
        else:
            required_status = "ineligible"
        if grade["status"] != required_status or (not eligible and grade["reward"] is not None):
            raise ValueError("grade and independent calibration disagree")
        if submitted:
            receipt = read(trial / "receipt.json")
            if receipt != result["receipt"] or receipt["sha256"] != digest(trial / "patch.safetensors"):
                raise ValueError("submitted artifact or receipt mismatch")
            if eligible and (
                grade["patch_sha256"] != receipt["sha256"] or grade["metrics"]["reward"] != grade["reward"]
            ):
                raise ValueError("native grade mismatch")
        elif (
            result["receipt"] is not None
            or (trial / "patch.safetensors").exists()
            or (eligible and grade["reward"] != 0)
        ):
            raise ValueError("no-submission accounting mismatch")
        packet = digest(trial / "public-input.json")
        if agreed_packets.setdefault(task_id, packet) != packet:
            raise ValueError("public task packet changed across arms")
        public_cases = read(trial / "public-input.json")["public_cases"]
        tools = {t["function"]["name"]: t["function"] for t in read(trial / "tools.json")}
        if menus.setdefault(arm, tools) != tools:
            raise ValueError("within-arm tool menus differ")
        for name in ("test_behavior", "preview_weights"):
            if ("cases" in tools[name]["parameters"].get("required", [])) != (arm == "explicit"):
                raise ValueError("wrong default cases interface")
        diag = audit_decode(trial, design, set(env["editor_eos_token_ids"]))
        generations = [
            json.loads(line) for line in (trial / "native-generations.jsonl").read_text().splitlines()
        ]
        capped_incomplete = [
            g
            for g in generations
            if g.get("parse_error") == "incomplete native tool-call delimiter"
            and len(g["generated_ids"]) == design["editor_decoding"]["max_new_tokens"]
            and g["generated_ids"][-1] not in env["editor_eos_token_ids"]
        ]
        mentions_ids = sum(re.search(r'"token_ids"\s*:', g["raw"]) is not None for g in capped_incomplete)
        dropped_calls = [complete_calls_in_failed_generation(g) for g in generations]
        trace = [json.loads(line) for line in (trial / "trajectory.jsonl").read_text().splitlines()]
        first_editor = next(i for i, e in enumerate(trace) if e["type"] == "updater")
        events = [e for e in trace[first_editor:] if e["type"] == "tool"]
        successes = [e for e in events if "result" in e and "error" not in e]
        tests = [e for e in events if e["tool"] == "test_behavior"]
        previews = [e for e in events if e["tool"] == "preview_weights"]
        valid_tests = [e for e in tests if "result" in e and "error" not in e]
        valid_previews = [e for e in previews if "result" in e and "error" not in e]
        if submitted != any(e["tool"] == "submit_patch" for e in successes):
            raise ValueError("trace submit status differs")
        metrics = grade["metrics"] or {}
        control_n = calibrations[task_id].get("metrics", {}).get("control_retention_denominator")
        base_targets = calibrations[task_id].get("metrics", {}).get("groups", {}).get("targets", {})
        retained = metrics.get("controls_retained", control_n)
        if eligible and metrics and control_n != metrics["control_retention_denominator"]:
            raise ValueError("edited-vs-pristine control denominator mismatch")
        records.append(
            dict(
                **assigned,
                status=result["status"],
                eligible=eligible,
                native_reward=grade["reward"],
                nonempty=bool(submitted and result["receipt"]["changed_parameters"]),
                submitted=submitted,
                checks=len(tests),
                valid_checks=len(valid_tests),
                default_checks=sum("cases" not in e["arguments"] for e in valid_tests),
                full_packet_checks=sum(
                    e["arguments"].get("cases", public_cases) == public_cases for e in valid_tests
                ),
                rejections_for_token_ids=sum("token_ids" in e.get("error", "") for e in tests),
                previews=len(previews),
                valid_previews=len(valid_previews),
                default_previews=sum("cases" not in e["arguments"] for e in valid_previews),
                commits=sum(e["tool"] == "commit_preview" for e in successes),
                target_gain=metrics.get("gain"),
                correct_with_pristine_fallback=metrics.get("groups", {})
                .get("targets", {})
                .get("edited_successes", base_targets.get("base_successes")),
                hidden_target_count=base_targets.get("count"),
                controls=control_n,
                retained_with_pristine_fallback=retained,
                total_tool_errors=sum("error" in e for e in events),
                terminal=diag["terminal"],
                opening_prompt_tokens=generations[0]["usage"]["prompt_tokens"],
                turns=diag["turns"],
                capped=diag["generations_at_ceiling"],
                malformed=diag["malformed_generations"],
                incomplete_at_ceiling=diag["incomplete_delimiters_at_ceiling"],
                incomplete_at_ceiling_mentioning_token_ids=mentions_ids,
                malformed_with_complete_siblings=sum(n > 0 for n in dropped_calls),
                complete_blocks_in_rejected_responses=sum(dropped_calls),
                **result["editor_usage"],
                target_usage=result["target_usage"],
            )
        )
    if len(seen) != 4 * len(tasks):
        raise ValueError("missing balanced assignment")
    if menus["short"].keys() != menus["explicit"].keys() or any(
        menus["short"][k] != menus["explicit"][k]
        for k in menus["short"]
        if k not in {"test_behavior", "preview_weights"}
    ):
        raise ValueError("nonexperimental tools differ across arms")
    arms = {}
    for arm in ("explicit", "short"):
        rows = [r for r in records if r["arm"] == arm]
        eligible = [r for r in rows if r["eligible"]]
        totals = {
            k: sum(r[k] for r in rows)
            for k in (
                "submitted",
                "nonempty",
                "turns",
                "checks",
                "valid_checks",
                "default_checks",
                "full_packet_checks",
                "previews",
                "valid_previews",
                "default_previews",
                "commits",
                "rejections_for_token_ids",
                "capped",
                "malformed",
                "incomplete_at_ceiling",
                "incomplete_at_ceiling_mentioning_token_ids",
                "malformed_with_complete_siblings",
                "complete_blocks_in_rejected_responses",
                "total_tool_errors",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
        }
        arms[arm] = dict(
            attempts=len(rows),
            eligible=len(eligible),
            **totals,
            sessions_with_valid_check=sum(r["valid_checks"] > 0 for r in rows),
            sessions_with_valid_preview=sum(r["valid_previews"] > 0 for r in rows),
            sessions_with_default_check=sum(r["default_checks"] > 0 for r in rows),
            sessions_with_default_preview=sum(r["default_previews"] > 0 for r in rows),
            mean_reward=mean(r["native_reward"] for r in eligible),
            mean_opening_prompt_tokens=mean(r["opening_prompt_tokens"] for r in rows),
            zero_rewards=sum(r["native_reward"] == 0 for r in eligible),
            total_target_forwards=sum(r["target_usage"]["forwards"] for r in rows),
            terminal_counts=Counter(r["terminal"] for r in rows),
            eligible_submissions=sum(r["submitted"] for r in eligible),
            controls_with_fallback=sum(r["controls"] for r in eligible),
            retained_with_fallback=sum(r["retained_with_pristine_fallback"] for r in eligible),
            hidden_correct_with_fallback=sum(r["correct_with_pristine_fallback"] for r in eligible),
            hidden_target_count=sum(r["hidden_target_count"] for r in eligible),
        )
    paired = {}
    for task_id, t in tasks.items():
        values = {
            arm: [r["native_reward"] for r in records if r["task_id"] == task_id and r["arm"] == arm]
            for arm in arms
        }
        good = calibrations[task_id]["status"] == "eligible"
        paired[task_id] = dict(
            family=t["family"],
            variant_index=t["variant_index"],
            eligible=good,
            **values,
            delta=mean(values["short"]) - mean(values["explicit"]) if good else None,
        )
    ds = [p["delta"] for p in paired.values() if p["eligible"]]
    resampling = np.random.default_rng(90173).choice(ds, (20000, len(ds)), replace=True).mean(axis=1)
    return dict(
        protocol=design["protocol"],
        design_sha256=env["design_sha256"],
        arms=arms,
        paired=paired,
        primary=dict(
            eligible_tasks=len(ds),
            mean_difference=mean(ds),
            wins=sum(d > 0 for d in ds),
            losses=sum(d < 0 for d in ds),
            ties=ds.count(0),
            descriptive_task_bootstrap_95=np.percentile(resampling, [2.5, 97.5]).tolist(),
        ),
        trials=records,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count-arguments", action="store_true", help="Run the tokenizer-only payload comparison")
    args = p.parse_args()
    report = count_argument_tokens(args.root) if args.count_arguments else audit(args.root)
    save(args.out, report)
    print(
        json.dumps(
            report if args.count_arguments else {"arms": report["arms"], "primary": report["primary"]},
            indent=2,
        )
    )
