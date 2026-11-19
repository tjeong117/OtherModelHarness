"""Summarize subdirectories of an Astra pilot run from local artifacts."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def select(obj, names):
    return {k: obj[k] for k in names.split() if k in obj}


def load(path, issues, optional=False):
    if not path.exists():
        if not optional:
            issues.append(f"missing {path.name}")
        return {}
    try:
        with path.open() as stream:
            obj = json.load(stream)
            if isinstance(obj, dict):
                return obj
    except (ValueError, UnicodeDecodeError):
        pass
    issues.append(f"invalid {path.name}")
    return {}


def diagnostic(event, ordinals):
    name = event["tool"]
    result, args = event["result"], event.get("arguments", {})
    details = {"tool": name, "call": event.get("call"), "weight_revision": event.get("weight_revision")}
    # Local ordinals identify matching prompts across tools without publishing prompt text.
    def ordinal(prompt):
        return ordinals.setdefault(prompt, len(ordinals))

    if name in ("patch_activations", "rank_attributions"):
        details.update(source_prompt_ordinal=ordinal(args["source_prompt"]),
                       target_prompt_ordinal=ordinal(args["target_prompt"]))
        details.update(select(result, "token_ids evaluated_sites"))
        details["source"] = select(result["source"], "position token_id margin")
        details["baseline"] = select(result["baseline"], "position token_id margin")
        details["ranking"] = [select(x, "layer component estimated_margin_change margin_change patched_margin")
                              for x in result["ranking"]]
    elif name == "feature_examples":
        details.update(select(result, "feature margin_token_ids active_prompts evaluated_prompts active_tokens "
                                      "evaluated_tokens activation_frequency"))
        details["examples"] = []
        for x in result["examples"]:
            row = select(x, "position input_token_id activation")
            row["prompt_ordinal"] = ordinal(args["prompts"][x["prompt_index"]])
            if "suppression" in x:
                row["suppression"] = select(x["suppression"], "baseline_margin suppressed_margin margin_change")
            details["examples"].append(row)
    elif name == "inspect_target" and (args.get("intervention") or result.get("margin_changes")):
        details["intervention"] = select(args.get("intervention") or {}, "feature value scale")
        details.update(select(result, "margin_changes"))
        details.update(select(args, "token_ids"))
        details["prompt_ordinals"] = [ordinal(p) for p in args["prompts"]]
        for k in ("baseline_outputs", "outputs"):
            details[k] = [select(row, "margin token_id") for row in result.get(k, [])]
    else:
        return None
    return details


def register_captures(e, captures):
    """Index only handles the target actually returned; preserve origins across slice reads."""
    result, name = e.get("result") or {}, e["tool"]
    origin = {"tool": name, "call": e.get("call"), "weight_revision": result.get("revision", e.get("weight_revision"))}
    origin.update(select(result.get("probe") or {}, "kind layer"))
    if name == "read_weights" and isinstance(result.get("handle"), str):
        parent = (e.get("arguments") or {}).get("handle")
        if parent and parent in captures:
            if parent != result["handle"]:
                captures[result["handle"]] = {**captures[parent], "via_read_call": e.get("call")}
        else:
            captures[result["handle"]] = {**origin, "kind": "weight" if parent is None else "unknown"}
    elif name == "contrast_target" and isinstance(result.get("direction"), str):
        captures[result["direction"]] = {**origin, "capture": "contrast_direction"}
    elif name in ("inspect_target", "run_target"):
        for group in ("outputs", "baseline_outputs"):
            for index, row in enumerate(result.get(group, [])):
                for obs in row.get("observations", []):
                    if isinstance(obs.get("handle"), str):
                        captures[obs["handle"]] = {
                            **origin, "capture": group, "prompt_index": index, **select(obs, "position step"),
                        }


def write_handles(expression, path=()):
    if isinstance(expression, dict):
        if isinstance(expression.get("handle"), str):
            yield path, expression["handle"]
        for key, v in expression.items():
            yield from write_handles(v, (*path, key))


def link_write(e, captures, lineage):
    args = e["arguments"]
    lineage["successful_writes"] += 1
    revision = e.get("weight_revision")
    sources, unresolved = [], 0
    for path, handle in write_handles(args):
        source = captures.get(handle)
        if source and source["weight_revision"] == revision - 1:
            sources.append({"operand_path": ".".join(path), **source})
        else:
            unresolved += 1
    if not (sources or unresolved):
        return
    kinds = {entry.get("kind", "unknown") for entry in sources}
    lineage["writes_with_resolved_handles"] += int(bool(sources))
    lineage["writes_with_activation_handles"] += int(bool(kinds & {"residual", "sae"}))
    lineage["writes_by_source_kind"].update(kinds)
    lineage["writes"].append({"call": e.get("call"), "weight_revision_after": revision,
                              "operation": args.get("operation"), "sources": sources,
                              "unresolved_references": unresolved})


def trace(path, prompted, issues):
    counts = {k: Counter() for k in ("requested", "successful", "errors", "unconfirmed", "after_submission", "successful_probe_modes")}
    pending, usage, measurements, ordinals = [], [], [], {}
    submitted = host_check_next = False
    last_target_usage = None
    host, captures = Counter(), {}
    lineage = {"successful_writes": 0, "writes_with_resolved_handles": 0, "writes_with_activation_handles": 0,
               "writes_by_source_kind": Counter(), "writes": []}

    def flush():
        for tool, _, invalid in pending:
            category = "after_submission" if submitted else "errors" if invalid else "unconfirmed"
            counts[category][tool] += 1
        pending.clear()

    if not path.exists():
        issues.append(f"missing {path.name}")
        return None, None, None, [], None, None
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                e = json.loads(line)
            except ValueError:
                issues.append(f"incomplete or invalid {path.name} line {line_no}")
                break
            if e["type"] == "updater":
                flush()
                host_check_next = False
                if e.get("usage") is not None:
                    usage.append(e["usage"])
                for call in (e.get("response") or {}).get("tool_calls") or []:
                    fn = call["function"]
                    try:
                        args = json.loads(fn["arguments"])
                        invalid = len(fn["arguments"]) > 262144
                    except (TypeError, ValueError, RecursionError):
                        args, invalid = None, True
                    pending.append((fn["name"], args, invalid))
                    counts["requested"][fn["name"]] += 1
                continue
            if e["type"] != "tool":
                continue
            name = e["tool"]
            last_target_usage = e.get("target_usage", last_target_usage)
            register_captures(e, captures)
            # FeedbackSession immediately performs its own run_target after every successful write.
            if host_check_next and name == "run_target":
                host[name] += 1
                host_check_next = False
                continue
            host_check_next = False
            index = next((i for i, (tool, args, invalid) in enumerate(pending)
                          if not invalid and name == tool and ("arguments" not in e or args == e["arguments"])), None)
            if index is None:
                host[name] += 1
                continue
            for tool, _, invalid in pending[:index]:
                counts["errors" if invalid else "unconfirmed"][tool] += 1
            del pending[:index + 1]
            if "error" in e:
                counts["errors"][name] += 1
            else:
                counts["successful"][name] += 1
                args = e.get("arguments") or {}
                probe = args.get("probe") or {}
                if probe.get("kind"):
                    mode = ".".join((name, probe["kind"], "intervention" if args.get("intervention") else "observation"))
                    counts["successful_probe_modes"][mode] += 1
                submitted = name == "submit_patch"
                host_check_next = name == "write_weights"
                if name == "write_weights":
                    link_write(e, captures, lineage)
                if prompted:
                    data = diagnostic(e, ordinals)
                    if data:
                        measurements.append(data)
    flush()
    return counts, host, measurements, usage, lineage, last_target_usage


def provider_usage(editor, traced):
    records = editor.get("provider")
    rows = [r["usage"] for r in records if r.get("usage")] if records is not None else traced
    out = {"source": "provider_records" if records is not None else "completed_trace_turns",
           "response_models": sorted({r["model"] for r in records or [] if r.get("model")}),
           "responses_with_usage": len(rows),
           "cost_upper_bound_usd": editor.get("cost_upper_bound_usd")}
    # Missing detail means the provider supplied no measurement for that field.
    for name, path in (("input_tokens", ("prompt_tokens",)), ("output_tokens", ("completion_tokens",)),
                       ("cached_input_tokens", ("prompt_tokens_details", "cached_tokens")),
                       ("reasoning_tokens", ("completion_tokens_details", "reasoning_tokens"))):
        values = [r.get(path[0]) if len(path) == 1 else (r.get(path[0]) or {}).get(path[1]) for r in rows]
        out[name] = sum(values) if rows and all(type(x) is int for x in values) else None
    return out


def episode(directory):
    issues = []
    editor = load(directory / "editor-result.json", issues)
    meta = editor or load(directory / "protocol.json", issues, optional=True)
    grade = load(directory / "grade.json", issues)
    case = meta.get("case")
    prompted = case == "diagnostic"
    session = meta.get("session") or load(directory / "session.json", issues, optional=True)
    receipt = load(directory / "receipt.json", issues, optional=editor.get("status") != "submitted")
    if receipt and editor.get("receipt") and receipt != editor["receipt"]:
        issues.append("editor-result and receipt disagree")
    if grade.get("patch_sha256") and receipt and grade["patch_sha256"] != receipt.get("sha256"):
        issues.append("native grader and receipt patch digests disagree")
    counts, host, measurements, usage, lineage, last_target_usage = trace(directory / "trajectory.jsonl", prompted, issues)
    metrics = grade.get("metrics") or {}
    numerical = {k: v for k, v in metrics.items() if type(v) in (bool, int, float) or v is None}
    if metrics.get("calibration"):
        numerical["calibration"] = select(metrics["calibration"], "eligible completed_count calibration_count correct_count")
    if metrics.get("groups"):
        numerical["groups"] = {name: select(group, "count base_successes edited_successes retained_successes "
                                                  "retention_denominator retention spillover")
                               for name, group in metrics["groups"].items()}
    return {
        "directory": directory.name, "case": case, "family": meta.get("family"), "seed": meta.get("seed"),
        "behavior_context": "prompted_diagnostic" if prompted else "task_directed" if case else None,
        "artifacts_complete": not issues, "artifact_issues": issues, "editor_status": editor.get("status"),
        "stop_reason": editor.get("reason") if editor.get("status") == "provider_stopped" else None,
        "requested_model": meta.get("requested_model"), "tools_offered": meta.get("tools_offered"),
        "model_tool_counts": counts, "host_or_unmatched_tool_events": host,
        "target": select(session, "model_id revision chat_template_sha256 block_count sae") or None,
        "iwibench_revision": meta.get("iwibench_revision"),
        "target_usage_including_host": editor.get("target_usage_including_host", last_target_usage),
        "target_usage_source": ("editor_result" if "target_usage_including_host" in editor
                                else "last_tool_event" if last_target_usage is not None else None),
        "patch": select(receipt, "sha256 bytes format changed_parameters revision") or None,
        "write_lineage": lineage,
        "provider": provider_usage(editor, usage), "diagnostic_measurements": measurements,
        "grade": {"status": grade.get("status"), "reward": grade.get("reward"), "metrics": numerical or None,
                  **select(grade, "model_id revision patch_sha256 reasons")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=Path, help="directory containing one subdirectory per episode")
    parser.add_argument("--out", required=True, type=Path, help="output summary JSON")
    args = parser.parse_args()
    directories = sorted(d for d in args.runroot.iterdir() if d.is_dir())
    if not directories:
        parser.error("runroot has no episode directories")
    cases = [episode(d) for d in directories]
    bounds = [case["provider"]["cost_upper_bound_usd"] for case in cases]
    report = {
        "episodes": cases, "patch_receipts": sum(case["patch"] is not None for case in cases),
        "episodes_with_missing_or_invalid_artifacts": sum(not case["artifacts_complete"] for case in cases),
        "episodes_with_cost_bound": sum(x is not None for x in bounds),
        "sum_reported_cost_upper_bounds_usd": round(sum(x for x in bounds if x is not None), 6)
                                               if any(x is not None for x in bounds) else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=args.out.parent, delete=False, encoding="utf-8") as temp:
        json.dump(report, temp, indent=2, allow_nan=False)
        temp.write("\n")
    Path(temp.name).replace(args.out)


if __name__ == "__main__":
    main()
