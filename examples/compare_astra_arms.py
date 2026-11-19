"""Pair the six Astra development episodes by family and seed."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from summarize_astra import provider_usage


CASES = {"alias": ("alias_acquisition", (1, 3, 11)),
         "suggestion": ("resistance_to_false_suggestions", (2, 3, 5))}
MATCH = ("requested_model", "reasoning", "max_turns", "max_output_tokens", "budget_cap_usd", "iwibench_revision")
USAGE = ("input_tokens", "cached_input_tokens", "uncached_input_tokens", "output_tokens",
         "total_tokens", "reasoning_tokens", "first_response_input_tokens", "responses_with_usage",
         "cost_upper_bound_usd")
COMMON_TOOLS = {"read_weights", "write_weights", "run_target"}
TOOLS = {"compact": COMMON_TOOLS | {"submit_patch"}, "native": COMMON_TOOLS | {"inspect_target"},
         "full": COMMON_TOOLS | {"submit_patch", "inspect_target", "contrast_target", "rank_attributions",
                                 "feature_examples", "patch_activations"}}


def read_optional(directory, name):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else {}


def counts(metrics):
    targets = metrics["groups"]["targets"]
    total = targets["count"]
    base, edited, retained = (targets[key] for key in ("base_successes", "edited_successes", "retained_successes"))
    opportunities = total - base
    result = {
        "target_items": total, "base_correct_targets": base, "base_wrong_targets": opportunities,
        "edited_correct_targets": edited, "base_correct_targets_retained": retained,
        "newly_correct_targets": edited - retained, "previously_correct_targets_lost": base - retained,
        "base_correct_controls": metrics["control_retention_denominator"],
        "controls_retained": metrics["controls_retained"],
    }
    if (not 0 <= retained <= base <= total or not 0 <= edited - retained <= opportunities
            or not 0 <= result["controls_retained"] <= result["base_correct_controls"]):
        raise ValueError("invalid grader counts")
    return result


def episode(directory, family, seed):
    protocol = read_optional(directory, "protocol.json")
    editor = read_optional(directory, "editor-result.json")
    grade = read_optional(directory, "grade.json")
    settings = {**protocol, **editor}
    native_prompt_size = None
    packet = read_optional(directory, "public-input.json")
    if settings.get("protocol") == "native-iwibench-v1" and packet.get("role") == "user":
        message = packet["content"]
        _, _, public_json = message.partition("\n\n")
        manifest = json.loads(public_json)["manifest"]
        encoded = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        if encoded not in public_json:
            raise ValueError("native manifest serialization differs from the recorded user message")
        native_prompt_size = {"user_message_utf8_bytes": len(message.encode()),
                              "manifest_utf8_bytes": len(encoded.encode()), "manifest_tensors": len(manifest["tensors"])}
    recovery = editor.get("recovery") or {}
    recovered = (settings.get("protocol") == "native-iwibench-v1"
                 and recovery.get("response_model_echo_available") is False
                 and recovery.get("usage_source") == "original_trajectory"
                 and recovery.get("sha256_verified") is True)
    integrity, warnings = [], []
    traced_usage = []
    if recovered:
        log = directory / "trajectory.jsonl"
        if log.exists():
            for line in log.read_text().splitlines():
                event = json.loads(line)
                if event.get("type") == "updater":
                    traced_usage.append(event.get("usage"))
        if not traced_usage or any(not isinstance(u, dict) for u in traced_usage):
            integrity.append("recovered original provider usage missing from trajectory")
            traced_usage = []
        elif editor.get("provider") and [r.get("usage") for r in editor["provider"]] != traced_usage:
            integrity.append("recovered provider usage differs from original trajectory")
    records = editor.get("provider") or []
    metering = {**editor, "provider": None} if recovered and not records else editor
    usage = provider_usage(metering, traced_usage)
    turn_usage = [r.get("usage") for r in records] or traced_usage
    usage["first_response_input_tokens"] = ((turn_usage[0] or {}).get("prompt_tokens") if turn_usage else None)
    if recovered:
        usage["source"] = "original_trajectory"
        bounds = [u.get("cost_upper_bound_usd") for u in traced_usage]
        if bounds and all(isinstance(bound, (int, float)) for bound in bounds):
            bound = round(sum(bounds), 8)
            if usage["cost_upper_bound_usd"] is None:
                usage["cost_upper_bound_usd"] = bound
            elif abs(usage["cost_upper_bound_usd"] - bound) > 1e-7:
                integrity.append("recovered cost differs from original trajectory")
    if not editor:
        usage["responses_with_usage"] = None
    i, o, cached = (usage[key] for key in ("input_tokens", "output_tokens", "cached_input_tokens"))
    usage["total_tokens"] = i + o if i is not None and o is not None else None
    usage["uncached_input_tokens"] = i - cached if i is not None and cached is not None else None
    if settings and (settings.get("family"), settings.get("seed")) != (family, seed):
        integrity.append("editor family/seed differs from expected directory")
    if editor and any(model != settings.get("requested_model") for model in usage["response_models"]):
        integrity.append("provider model differs from requested model")
    if editor and grade.get("status") == "scored" and not usage["response_models"]:
        if recovered:
            warnings.append("Recovered from original trajectory and submitted weights. The requested API model "
                            "is recorded, and the provider model echo is unavailable.")
        else:
            integrity.append("responding model missing")
    state = grade.get("status", "missing_grade" if editor else "missing_editor_result")
    stats, pristine, target = None, None, None
    if state == "scored":
        metrics = grade["metrics"]
        stats = counts(metrics)
        target = {key: grade[key] for key in ("episode_id", "model_id", "revision")}
        if (grade["family"] != family or metrics["episode_id"] != grade["episode_id"]
                or not grade["episode_id"].endswith(f"/development/{seed}")):
            integrity.append("grader episode differs from expected development episode")
        if grade["reward"] != metrics["reward"] or not metrics["calibration"]["eligible"]:
            integrity.append("grader reward or calibration is inconsistent")
        patch_hash = grade["patch_sha256"]
        receipt = editor.get("receipt") or read_optional(directory, "receipt.json")
        if receipt.get("sha256") and receipt["sha256"] != patch_hash:
            integrity.append("submitted and graded patch hashes differ")
        patch = directory / "patch.safetensors"
        if recovered and (not patch.exists() or receipt.get("sha256") != patch_hash):
            integrity.append("recovered local patch or matching receipt unavailable")
        if patch.exists():
            with patch.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != patch_hash:
                    integrity.append("local and graded patch hashes differ")
        # The same control denominator can hide a mismatch between control subgroups.
        pristine = {"groups": {name: [group["count"], group["base_successes"]]
                               for name, group in metrics["groups"].items()},
                    "calibration": metrics["calibration"]}
    reward = grade.get("reward")
    if state == "no_submission" and reward != 0:
        integrity.append("no-submission grade must record zero reward")
    return {
        "editor_result_present": bool(editor), "editor_status": editor.get("status"),
        "grade_status": state, "reward": reward,
        "reason": editor.get("reason") or grade.get("reasons"), "patch_sha256": grade.get("patch_sha256"),
        "target": target, "counts": stats, "pristine": pristine,
        "protocol": settings.get("protocol"), "settings": {k: settings.get(k) for k in MATCH},
        "public_input_sha256": hashlib.sha256((directory / "public-input.json").read_bytes()).hexdigest()
                               if (directory / "public-input.json").is_file() else None,
        "native_opening_packet": native_prompt_size,
        "target_limits": (settings.get("session") or {}).get("limits", settings.get("limits")),
        "tools_offered": settings.get("tools_offered"), "target_usage_including_host": editor.get("target_usage_including_host"),
        "provider": usage, "integrity_issues": integrity, "warnings": warnings,
    }


def pair(full_dir, control_dir, family, seed, control_label):
    full = episode(full_dir, family, seed)
    control = episode(control_dir, family, seed)
    arms = (full, control)
    missing = [name for name, arm in zip(("full", "control"), arms)
               if arm["grade_status"].startswith("missing") or not arm["editor_result_present"]]
    mismatches = [f"{name}: {issue}" for name, arm in zip(("full", "control"), arms)
                  for issue in arm["integrity_issues"]]
    for label, arm in zip(("full", control_label), arms):
        if arm["editor_result_present"] and set(arm["tools_offered"] or []) != TOOLS[label]:
            mismatches.append(f"offered tools differ from {label} protocol")
    if control_label == "compact" and not missing:
        for key in ("public_input_sha256", "target_limits"):
            if full[key] is None or control[key] is None or full[key] != control[key]:
                mismatches.append(f"compact and full must supply the same {key}")
    for name in MATCH:
        f, c = (arm["settings"][name] for arm in arms)
        if f is not None and c is not None and f != c:
            mismatches.append(f"{name} differs")
        elif not missing and (f is None or c is None):
            mismatches.append(f"{name} missing from settings")
    if all(arm["target"] is not None for arm in arms):
        if full["target"] != control["target"]:
            mismatches.append("graded episode ID or target checkpoint differs")
        if full["pristine"] != control["pristine"]:
            mismatches.append("pristine group or calibration counts differ")
    valid = not missing and not mismatches
    scored_pair = valid and all(arm["grade_status"] == "scored" for arm in arms)
    reward_pair = valid and all(arm["grade_status"] in ("no_submission", "scored")
                                and arm["reward"] is not None for arm in arms)
    return {"family": family, "seed": seed, "missing_arms": missing, "mismatches": mismatches,
            "both_freshly_scored": scored_pair, "paired_reward_available": reward_pair,
            "reward_difference_full_minus_control": full["reward"] - control["reward"] if reward_pair else None,
            "full": full, "control": control}


def totals(pairs):
    scored = [p for p in pairs if p["both_freshly_scored"]]
    rewards = [p for p in pairs if p["paired_reward_available"]]
    result = {"scheduled_pairs": len(pairs), "paired_rewards": len(rewards),
              "paired_fresh_grades": len(scored),
              "pairs_with_missing_results": sum(bool(p["missing_arms"]) for p in pairs),
              "pairs_with_evidence_warnings": sum(bool(p["full"]["warnings"] or p["control"]["warnings"]) for p in pairs),
              "pairs_with_mismatches": sum(bool(p["mismatches"]) for p in pairs)}
    for arm in ("full", "control"):
        pooled = None
        if scored:
            pooled = {k: sum(p[arm]["counts"][k] for p in scored) for k in scored[0][arm]["counts"]}
        usage = {}
        for key in USAGE:
            values = [p[arm]["provider"][key] for p in pairs if p[arm]["provider"][key] is not None]
            usage[key] = {"sum_reported": round(sum(values), 8) if values else None, "episodes_reporting": len(values)}
        result[arm] = {
            "grade_status_counts": dict(Counter(p[arm]["grade_status"] for p in pairs)),
            "mean_reward_on_paired_rewards": mean(p[arm]["reward"] for p in rewards) if rewards else None,
            "counts_on_paired_fresh_grades": pooled,
            "provider_usage_all_scheduled": usage,
        }
    differences = [p["reward_difference_full_minus_control"] for p in rewards]
    result["mean_reward_difference_full_minus_control"] = mean(differences) if differences else None
    result["full_wins_ties_losses"] = {"wins": sum(x > 0 for x in differences),
                                        "ties": sum(x == 0 for x in differences),
                                        "losses": sum(x < 0 for x in differences)}
    tokens = [result[a]["provider_usage_all_scheduled"]["total_tokens"] for a in ("full", "control")]
    measured = all(t["episodes_reporting"] == len(pairs) for t in tokens)
    result["total_token_ratio_full_to_control"] = (
        tokens[0]["sum_reported"] / tokens[1]["sum_reported"] if measured and tokens[1]["sum_reported"] else None)
    return result


def compare(full, control, control_label):
    pairs = []
    for case, (family, seeds) in CASES.items():
        for seed in seeds:
            name = f"{case}-{seed}"
            pairs.append(pair(full / name, control / name, family, seed, control_label))
    return {
        "control_label": control_label,
        "design": "One trajectory per arm per seed on six previously selected development episodes. "
                  "Full precedes control. Pairing is by task and seed. The prompted diagnostic is excluded.",
        "reward_summary": "Equal episode weights. A recorded no-submission earns zero. Missing, ineligible, "
                          "incompatible and infrastructure-failed trajectories have separate counts. "
                          "Target and control counts require two fresh native grades.",
        "token_summary": "Input includes cached input. Output includes reasoning. Total is input plus output. "
                         "First-response input includes the initial editor instructions, task packet and tool schemas. "
                         "Cost is a conservative provider upper bound in USD, excluding GPU charges. "
                         "Coverage for each usage sum includes all scheduled cases with that measurement.",
        "pairs": pairs,
        "per_family": {family: totals([p for p in pairs if p["family"] == family])
                       for family, _ in CASES.values()},
        "overall": totals(pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--control-label", choices=("native", "compact"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.full, args.control, args.control_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
