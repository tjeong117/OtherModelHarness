"""Audit and summarize the four prespecified runtime SAE access trials from disk."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

from compare_astra_arms import episode, read_optional
from summarize_astra import trace


CASES = (("alias", "alias_acquisition", 1), ("suggestion", "resistance_to_false_suggestions", 2))
ARMS = ("fit", "no-fit")
FIT_CAPS = ("max_fitted_dictionaries", "max_sae_fit_steps")
EXPECTED_COMMON_TOOLS = {
    "read_weights", "write_weights", "run_target", "inspect_target", "contrast_target",
    "patch_activations", "rank_attributions", "feature_examples", "submit_patch",
    "discover_target", "select_sae", "tokenize_target",
}
USAGE_FIELDS = ("input_tokens", "cached_input_tokens", "uncached_input_tokens", "output_tokens",
                "reasoning_tokens", "total_tokens", "cost_upper_bound_usd")


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def shared_settings(protocol):
    # The two SAE-specific limits and the resulting advertised schema size are part of treatment.
    session = dict(protocol.get("session", {}))
    limits = session.pop("limits", {})
    session.pop("tool_schema", None)
    session["limits"] = {k: v for k, v in limits.items() if k not in FIT_CAPS}
    return {k: v for k, v in protocol.items() if k not in ("arm", "tools_offered", "session")} | {"session": session}


def audit_arm(path, case, family, seed, arm, *, expected_protocol="runtime-sae-access-v1"):
    data = episode(path, family, seed)
    issues = data["integrity_issues"]
    protocol = read_optional(path, "protocol.json")
    editor = read_optional(path, "editor-result.json")
    schemas = read_optional(path, "tools.json") or []
    session = protocol.get("session", {})
    caps = session.get("limits", {})
    cap_values = [caps.get(k) for k in FIT_CAPS]
    if (protocol.get("protocol"), protocol.get("case"), protocol.get("arm")) != (expected_protocol, case, arm):
        issues.append("missing or unexpected protocol, case or arm")
    if any(editor.get(k) != value for k, value in protocol.items()) and editor:
        issues.append("editor-result and protocol settings differ")
    if cap_values != ([3, 256] if arm == "fit" else [0, 0]):
        issues.append("unexpected SAE fitting limits")
    if "sae" not in session or session["sae"] is not None:
        issues.append("expected an empty starting dictionary")
    offered = [s["function"]["name"] for s in schemas]
    expected = EXPECTED_COMMON_TOOLS | ({"fit_sae"} if arm == "fit" else set())
    if offered != protocol.get("tools_offered") or len(offered) != len(expected) or set(offered) != expected:
        issues.append("unexpected or missing advertised tool schemas")
    schema_json = json.dumps(schemas, separators=(",", ":"), ensure_ascii=False)
    if {"utf8_bytes": len(schema_json.encode())} != session.get("tool_schema"):
        issues.append("saved schema size differs from session metadata")
    path_trace = path / "trajectory.jsonl"
    calls, _, _, traced_usage, lineage, target_usage = trace(path_trace, False, issues)
    calls = calls or {}
    records = editor.get("provider", [])
    if editor:
        # An incomplete provider response can be charged before the updater sees it.
        handed_to_updater = [r.get("usage") for r in records if r.get("status") == "completed"]
        if traced_usage != handed_to_updater:
            issues.append("completed provider usage differs from updater trace")
        if any(r.get("model") != protocol.get("requested_model") for r in records):
            issues.append("missing or unexpected provider model echo")
        if records and editor.get("status") == "submitted" and len(records) != len(handed_to_updater):
            issues.append("submitted arm contains an incomplete provider response")
        if any(u.get("total_tokens") != u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
               for u in traced_usage):
            issues.append("provider totals inconsistent")
        if target_usage is not None and target_usage != editor.get("target_usage_including_host"):
            issues.append("target meter differs from trajectory")
        if dict(calls.get("successful", {})) != editor.get("successful_model_tool_counts", {}):
            issues.append("successful tool counts differ from trace")
        charge = editor.get("cost_upper_bound_usd")
        spent = sum(r["usage"]["cost_upper_bound_usd"] for r in records)
        if charge is None or charge < spent - 1e-7 or charge > protocol["budget_cap_usd"] + 1e-7:
            issues.append("cost upper bound outside recorded provider charges and budget")
        elif editor.get("status") in ("submitted", "no_submission") and abs(charge - spent) > 1e-7:
            issues.append("completed edit charges differ from summed provider usage")
    first_task = None
    fit_calls = []
    feature_uses, writes = [], []
    if path_trace.exists():
        for line in path_trace.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                break  # trace() already recorded the invalid line.
            if event.get("type") == "updater" and first_task is None:
                first_task = event.get("task")
            if event.get("type") == "tool" and event.get("tool") == "fit_sae" and "error" not in event:
                fit = event["result"]
                fit_calls.append({k: fit.get(k) for k in ("dictionary_id", "layer", "n_features", "top_k",
                                                          "steps", "source_revision", "training", "validation", "cost")})
            if event.get("type") == "tool" and "error" not in event:
                tool = event["tool"]
                args = event.get("arguments", {})
                result = event.get("result", {})
                probe = result.get("probe") or {}
                if probe.get("kind") == "sae" or tool == "feature_examples":
                    feature_uses.append({"tool": tool, "call": event.get("call"), "probe": probe,
                                         "feature": args.get("feature"), "scale": args.get("scale"),
                                         "intervention": args.get("intervention"),
                                         "ranking": result.get("ranking"), "margin_changes": result.get("margin_changes")})
                if tool == "write_weights":
                    writes.append({"call": event.get("call"), "operation": args.get("operation"),
                                   "tensor": args.get("tensor"), "selection": args.get("selection"),
                                   "factor": args.get("factor"), "changed_parameters": result.get("changed_parameters")})
    if traced_usage and (not first_task or first_task != json.dumps(read_optional(path, "public-input.json"))):
        issues.append("actual updater task differs from saved public input")
    if arm == "no-fit" and (fit_calls or (target_usage or {}).get("sae_fit_steps", 0)):
        issues.append("control obtained a fitted SAE")
    receipt = editor.get("receipt") or {}
    if data["grade_status"] in ("scored", "ineligible"):
        if data["patch_sha256"] != receipt.get("sha256") or not receipt:
            issues.append("no matching patch receipt for native grade")
        if data.get("target") and any(data["target"][k] != session.get(k) for k in ("model_id", "revision")):
            issues.append("graded target differs from editing target")
    data.update(
        changed_parameters=receipt.get("changed_parameters"),
        actual_user_prompt_sha256=sha256(first_task) if first_task else None,
        tool_schemas_sha256=sha256(schema_json) if schemas else None,
        provider_statuses=dict(Counter(r.get("status") for r in records)),
        successful_model_tool_counts=dict(calls.get("successful", {})),
        requested_model_tool_counts=dict(calls.get("requested", {})),
        successful_probe_modes=dict(calls.get("successful_probe_modes", {})),
        writes_by_source_kind=(lineage or {}).get("writes_by_source_kind", {}),
        fitted_dictionaries=fit_calls,
        successful_feature_uses=feature_uses, successful_writes=writes,
    )
    return data, schemas, shared_settings(protocol)


def summarize_pair(root, case, family, seed, design):
    arms, schemas, settings = {}, {}, {}
    issues = []
    for arm in ARMS:
        path = root / f"{case}-{seed}" / arm
        arms[arm], schemas[arm], settings[arm] = audit_arm(path, case, family, seed, arm)
        issues.extend(f"{arm}: {item}" for item in arms[arm]["integrity_issues"])
        if design and arms[arm]["editor_result_present"]:
            expected = {"requested_model": design["updater"], "reasoning": design["reasoning"],
                        "max_turns": design["max_turns_including_submission"],
                        "max_output_tokens": design["max_output_tokens_per_response"],
                        "budget_cap_usd": design["provider_cost_ceiling_usd_per_trajectory"],
                        "iwibench_revision": design["iwibench_revision"]}
            for k, value in expected.items():
                if settings[arm].get(k) != value:
                    issues.append(f"{arm}: {k} differs from the prespecified design")
            target = settings[arm]["session"]
            if (target.get("model_id"), target.get("revision")) != (design["target"], design["target_revision"]):
                issues.append(f"{arm}: target differs from the prespecified design")
    fit, control = arms.values()
    if [s for s in schemas["fit"] if s["function"]["name"] != "fit_sae"] != schemas["no-fit"]:
        issues.append("the shared tool schemas or their order differ")
    if settings["fit"] != settings["no-fit"]:
        issues.append("settings outside fitting access differ")
    for key in ("public_input_sha256", "actual_user_prompt_sha256"):
        if fit[key] and control[key] and fit[key] != control[key]:
            issues.append(f"{key} differs")
    if fit["pristine"] and control["pristine"] and fit["pristine"] != control["pristine"]:
        issues.append("pristine group scores or calibration differ")
    eligible = not issues and all(a["editor_result_present"] and a["grade_status"] in ("scored", "no_submission")
                                  and a["reward"] is not None for a in arms.values())
    return {"case": case, "family": family, "seed": seed, "arms": arms, "integrity_issues": issues,
            "paired_reward_available": eligible,
            "both_freshly_scored": eligible and all(a["grade_status"] == "scored" for a in arms.values()),
            "reward_difference_fit_minus_no_fit": fit["reward"] - control["reward"] if eligible else None}


def instructed_followup(root, pairs, plan):
    case, family, seed = CASES[1]
    name = f"{case}-{seed}"
    reference = next(p for p in pairs if (p["case"], p["seed"]) == (case, seed))
    control = reference["arms"]["fit"]
    baseline_path = root / name / "fit"
    instructed_path = root / "instructed-followup" / name
    instructed, schemas, _ = audit_arm(instructed_path, case, family, seed, "fit",
                                        expected_protocol="runtime-sae-instructed-v1")
    issues = list(reference["integrity_issues"]) + list(instructed["integrity_issues"])
    if (plan["case"], plan["seed"]) != (case, seed):
        issues.append("follow-up design case or seed differs")
    baseline_config = read_optional(baseline_path, "protocol.json")
    instructed_config = read_optional(instructed_path, "protocol.json")
    if (baseline_config | {"protocol": "runtime-sae-instructed-v1"}) != instructed_config:
        issues.append("instructed and autonomous protocols differ beyond protocol ID")
    if read_optional(baseline_path, "tools.json") != schemas:
        issues.append("instructed and autonomous schemas differ")
    if instructed["grade_status"] == "scored" and not (instructed_path / "patch.safetensors").is_file():
        issues.append("instructed patch missing for verification against the grader digest")
    base_packet = read_optional(baseline_path, "public-input.json")
    expected_packet = {**base_packet, "objective": base_packet.get("objective", "") + " " + plan["instruction"]}
    if expected_packet != read_optional(instructed_path, "public-input.json"):
        issues.append("instructed packet differs beyond the prespecified instruction")
    if (control["pristine"] and instructed["pristine"] and control["pristine"] != instructed["pristine"]):
        issues.append("instructed pristine grading groups or calibration differ")
    if any(p["arms"]["fit"]["fitted_dictionaries"] for p in pairs):
        issues.append("autonomous fitting invalidates the prespecified follow-up condition")
    comparable = (not issues and all(a["grade_status"] in ("scored", "no_submission") and a["reward"] is not None
                                     for a in (control, instructed)))
    return {"case": case, "seed": seed, "scope": plan["scope"], "instruction": plan["instruction"],
            "comparison_arm": f"{name}/fit", "integrity_issues": issues,
            "instructed": instructed, "autonomous_fit_reward": control["reward"],
            "reward_difference_instructed_minus_autonomous_fit": instructed["reward"] - control["reward"]
                                                                 if comparable else None}


def summarize(root, design=None, *, followup_design=None):
    pairs = [summarize_pair(root, *case, design) for case in CASES]
    rewards = [p for p in pairs if p["paired_reward_available"]]
    scored = [p for p in pairs if p["both_freshly_scored"]]
    differences = [p["reward_difference_fit_minus_no_fit"] for p in rewards]
    overall = {"scheduled_pairs": len(pairs), "paired_rewards": len(rewards), "paired_fresh_grades": len(scored),
               "pairs_with_integrity_issues": sum(bool(p["integrity_issues"]) for p in pairs),
               "mean_reward_difference_fit_minus_no_fit": mean(differences) if differences else None,
               "fit_wins_ties_losses": {"wins": sum(d > 0 for d in differences), "ties": differences.count(0),
                                        "losses": sum(d < 0 for d in differences)}}
    for arm in ARMS:
        rows = [p["arms"][arm] for p in pairs]
        all_usage = {}
        for key in USAGE_FIELDS:
            values = [row["provider"][key] for row in rows if row["provider"][key] is not None]
            all_usage[key] = {"sum_reported": round(sum(values), 8) if values else None,
                              "episodes_reporting": len(values)}
        pooled = {k: sum(p["arms"][arm]["counts"][k] for p in scored)
                  for k in scored[0]["arms"][arm]["counts"]} if scored else None
        overall[arm] = {
            "grade_status_counts": dict(Counter(row["grade_status"] for row in rows)),
            "mean_reward_on_paired_rewards": mean(p["arms"][arm]["reward"] for p in rewards) if rewards else None,
            "counts_on_paired_fresh_grades": pooled,
            "episodes_with_completed_editor_response": sum(bool(row["provider_statuses"].get("completed")) for row in rows),
            "episodes_requesting_fit": sum(bool(row["requested_model_tool_counts"].get("fit_sae")) for row in rows),
            "episodes_successfully_fitting": sum(bool(row["fitted_dictionaries"]) for row in rows),
            "provider_usage_all_scheduled": all_usage,
        }
    report = {"design": "Availability of runtime SAE fitting. One autonomous trajectory per arm and task. "
                      "No pretrained dictionary. Equal episode weights. Input tokens include cached tokens. "
                      "A missing submission earns zero. A provider stop, ineligible grade or missing artifact "
                      "does not constitute a scored result. Native item counts use freshly graded pairs only.",
            "pairs": pairs, "overall": overall}
    if (root / "instructed-followup/suggestion-2/grade.json").is_file() and followup_design is not None:
        report["instructed_followup"] = instructed_followup(root, pairs, followup_design)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--design", type=Path,
                        default=Path(__file__).resolve().parents[1] / "evidence/astra-sae-ab-v1/design.json")
    args = parser.parse_args()
    plan = args.design.read_text()
    followup_file = args.design.with_name("instructed-followup-design.json")
    followup = followup_file.read_text() if (args.root / "instructed-followup/suggestion-2/grade.json").is_file() else None
    report = summarize(args.root, json.loads(plan), followup_design=json.loads(followup) if followup else None)
    if followup:
        report["instructed_followup"]["design_artifact_sha256"] = sha256(followup)
    report["design_artifact_sha256"] = sha256(plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
