"""Audit the matched qualified IWIBench study using local editor and grader artifacts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from safetensors import safe_open

from astra_qualified_native import SCHEMA_SHA
from compare_astra_arms import counts, read_optional
from iwi_harness.session import encode_json
from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION
from iwi_harness.updater import run_updater, sae_guided_objective


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("paged", "eager", "native")
COUNTERS = ("calls", "input_tokens", "cached_input_tokens", "uncached_input_tokens", "output_tokens", "total_tokens")
LAYOUT_SCHEMA_SHA = "674bffd26d5fe4446745a036f38bc2c805e316f12e7c5e0221c1c8b61eaee5aa"
NATIVE_MANIFEST_SHA = "469d006bfa0e956850b41ffdc0f31d158971291f860729da4521f74726d5fb36"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def usage(records):
    values = [r["usage"] for r in records]
    i = sum(r["prompt_tokens"] for r in values)
    o = sum(r["completion_tokens"] for r in values)
    c = sum(r["prompt_tokens_details"]["cached_tokens"] for r in values)
    for r in values:
        check(0 <= r["prompt_tokens_details"]["cached_tokens"] <= r["prompt_tokens"]
              and r["total_tokens"] == r["prompt_tokens"] + r["completion_tokens"], "inconsistent usage")
    return dict(calls=len(values), input_tokens=i, cached_input_tokens=c, uncached_input_tokens=i-c,
                output_tokens=o, total_tokens=i+o,
                reported_cost_upper_bound_usd=round(sum(r["cost_upper_bound_usd"] for r in values), 8))


def audit_layout(path, protocol, events, schema, arm):
    packet = read_optional(path, "public-input.json")
    opening = read_optional(path, "opening-manifest.json")
    first = next((t for t in events if t["type"] == "updater"), None)
    objective_hash = sha(encode_json(packet).encode())
    check(protocol["sae_guided"] and protocol["objective_sha256"] == opening["objective_sha256"] == objective_hash,
          "public objective or guided policy mismatch")
    if first:
        check(first["task"] == encode_json(packet), "actual first recorded user input differs")
    check(opening["mode"] == arm and opening["tool_schema_sha256"] == sha(encode_json(schema).encode()) == LAYOUT_SCHEMA_SHA,
          "opening mode or schema mismatch")
    entries, queries, pages = [], [], []
    for e in events:
        if e["type"] == "updater":
            break
        if e.get("tool") == "read_weights":
            queries.append(e["arguments"])
            pages.append(e["result"])
            entries.extend([t["name"], t["shape"]] for t in e["result"]["tensors"])
    check(len(queries) == opening["metadata_tool_calls"] and len(entries) == opening["available_entries"]
          and sha(encode_json(entries).encode()) == opening["ordered_entries_sha256"], "metadata replay mismatch")
    replay = iter(zip(queries,pages))
    def lookup(name, arguments):
        recorded, result = next(replay)
        check(name == "read_weights" and arguments == recorded, "opening query differs")
        return result
    class OpeningVerified(Exception):
        pass
    def capture(messages, tools):
        check(sha(messages[0]["content"].encode()) == opening["opening_system_sha256"]
              and messages[1]["content"] == encode_json(packet) and tools == schema, "opening system or task differs")
        raise OpeningVerified
    with TemporaryDirectory() as temp:
        session = SimpleNamespace(info=protocol["session"], tools=schema, artifact_dir=Path(temp), call=lookup)
        try:
            run_updater(session,encode_json(packet), capture, max_turns=protocol["max_turns"], opening_manifest=arm)
        except OpeningVerified:
            pass
        check(read_optional(Path(temp),"opening-manifest.json") == opening, "opening metadata replay differs")
    check((protocol["session"]["model_id"], protocol["session"]["revision"]) == (GEMMA_ID, GEMMA_REVISION),
          "layout checkpoint mismatch")
    return {"packet_sha256": objective_hash, "tools_sha256": opening["tool_schema_sha256"],
            "session": protocol["session"], "practice_tokens_per_answer": protocol["practice_tokens_per_answer"],
            "metadata_queries": queries, "ordered_entries_sha256": opening["ordered_entries_sha256"]}


def audit_arm(root, task, arm, design):
    path = root / task["family"] / arm
    proto = read_optional(path, "protocol.json")
    editor = read_optional(path, "editor-result.json")
    grade = read_optional(path, "grade.json")
    marker = root / "controller" / f"{task['family']}-{task['variant_index']}-{arm}.exit"
    status = editor.get("status", "missing_editor_result")
    result = dict(terminal=False, editor_status=status, grade_status=grade.get("status"), reason=editor.get("reason"))
    if not editor:
        result["controller_exit"] = marker.read_text().strip() if marker.exists() else None
        return result, None, []
    expected = {"task_id": task["id"], "family": task["family"], "variant_index": task["variant_index"],
                "episode_sha256": task["episode_sha256"], "parameter_sha256": task["parameter_sha256"],
                "requested_model": design["requested_model"], "reasoning": design["reasoning"],
                "max_turns": design["max_turns"], "max_output_tokens": design["max_output_tokens"],
                "budget_cap_usd": design["budget"]["max_usd_per_edit"], "arm": arm,
                "catalog_version": "baseline-qualified-v1"}
    check(all(proto.get(k) == v for k,v in expected.items()) and all(editor.get(k) == v for k,v in proto.items()),
          "protocol or editor metadata differs from design")
    events = [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]
    turns = [e for e in events if e["type"] == "updater"]
    records = editor["provider"]
    check(len(records) <= design["max_turns"] and [x["usage"] for x in turns] ==
          [x["usage"] for x in records if x["status"] == "completed"], "recorded editor usages differ from provider")
    check(all(r["model"] == design["requested_model"] and r["usage"]["completion_tokens"] <= design["max_output_tokens"]
              for r in records), "responding model or output limit mismatch")
    metered = usage(records)
    bound = editor["cost_upper_bound_usd"]
    check(metered["reported_cost_upper_bound_usd"] <= bound + 1e-8 and bound <= proto["budget_cap_usd"] + 1e-8,
          "provider cost or reservation exceeds recorded limit")
    result.update(usage=metered, first_request=usage(records[:1]), cost_upper_bound_usd=bound,
                  unknown_usage_reservation_usd=round(bound-metered["reported_cost_upper_bound_usd"],8),
                  target_usage=editor["target_usage_including_host"], public_practice_checks=editor["host_public_practice_checks"])
    schemas = read_optional(path, "tools.json")
    if arm == "native":
        sig = None
        check(proto["protocol"] == "qualified-stock-native-v1" and proto["dataset_revision"] == design["hub_revision"]
              and (proto["target"],proto["target_revision"]) == (GEMMA_ID,GEMMA_REVISION), "native provenance mismatch")
        check(sha((path / "tools.json").read_bytes()) == proto["schemas_file_sha256"] == SCHEMA_SHA
              and set(proto["tools_offered"]) == {"run_target","inspect_target","read_weights","write_weights"},
              "native schema mismatch")
        from astra_native_control import native_prompt
        packet = read_optional(path, "public-input.json")
        inner = json.loads(packet["content"].split("\n\n",1)[1])
        check(packet == {"role":"user","content":native_prompt(inner)}
              and sha(encode_json(inner["manifest"]).encode()) == NATIVE_MANIFEST_SHA,
              "native public user packet differs from the stock template")
        result["public_objective"] = sha(inner["objective"].encode())
        result["public_practice"] = sha(json.dumps(inner["practice"], sort_keys=True).encode())
    else:
        check(proto["protocol"] == "qualified-layout-v1", "unexpected layout protocol")
        sig = audit_layout(path, proto, events, schemas, arm)
        result["opening"] = read_optional(path, "opening-manifest.json")
        packet = read_optional(path, "public-input.json")
        result["public_practice"] = sha(json.dumps(packet["practice"], sort_keys=True).encode())
        suffix = sae_guided_objective("",design["max_turns"])
        check(packet["objective"].endswith(suffix), "unrecognized guided prompt")
        result["public_objective"] = sha(packet["objective"][:-len(suffix)].encode())
    result["successful_editor_tools"] = (editor["successful_model_tool_counts"] if arm == "native" else
                                         dict(Counter(call["tool"] for call in editor["model_calls"] if "error" not in call)))
    if status != "submitted":
        check(editor.get("receipt") is None and not (path / "receipt.json").exists(), "unsubmitted run has receipt")
        result["terminal"] = True
        if status == "no_submission":
            result.update(effective_reward=0.0, effective_reward_source="host_zero_for_no_submission")
        return result, sig, records
    receipt = read_optional(path, "receipt.json")
    patch = path / "patch.safetensors"
    check(receipt == editor["receipt"] and sha(patch.read_bytes()) == receipt["sha256"]
          and receipt["bytes"] == patch.stat().st_size, "submission or patch digest mismatch")
    with safe_open(patch, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        payload_tensors = len(f.keys())
    check((metadata["model_id"], metadata["revision"], metadata["format"]) ==
          (GEMMA_ID,GEMMA_REVISION,receipt["format"]), "submitted patch metadata mismatch")
    check((receipt["changed_parameters"] > 0) == (payload_tensors > 0), "weight count and patch payload disagree")
    result.update(patch_sha256=receipt["sha256"], patch_bytes=receipt["bytes"], patch_payload_tensors=payload_tensors,
                  changed_parameters=receipt["changed_parameters"])
    if not grade:
        return result, sig, records
    for k,v in {"episode_id":task["id"],"family":task["family"],"episode_sha256":task["episode_sha256"],
                "dataset_revision":design["hub_revision"],"ranked_split":"test","model_id":GEMMA_ID,"revision":GEMMA_REVISION,
                "patch_sha256":receipt["sha256"]}.items():
        check(grade.get(k) == v, f"qualified grade mismatch: {k}")
    check(grade["status"] in ("scored", "ineligible"), "unrecognized qualified grader outcome")
    if grade["status"] == "scored":
        m=grade["metrics"]
        check(m["episode_id"] == task["id"] and m["calibration"]["eligible"] and
              m["reward"] == grade["reward"] and 0 <= grade["reward"] <= 1, "qualified grade/calibration inconsistency")
        result.update(reward=grade["reward"], effective_reward=grade["reward"], effective_reward_source="qualified_native_grade",
                      counts=counts(m),
                      pristine_identity=sha(encode_json({k:[v["count"],v["base_successes"]]
                                                         for k,v in m["groups"].items()}).encode()))
    result["terminal"] = True
    return result, sig, records


def aggregated(rows, arm):
    observed = [row[arm] for row in rows if "usage" in row[arm]]
    scored = [r for r in observed if "counts" in r]
    totals = {k:sum(r["usage"][k] for r in observed) for k in COUNTERS}
    count_totals = {k:sum(r["counts"][k] for r in scored) for k in (scored[0]["counts"] if scored else ())}
    effective = [r for r in observed if "effective_reward" in r]
    return {"scheduled":len(rows), "terminal":sum(row[arm]["terminal"] for row in rows),
            "editor_results_with_usage_accounting":len(observed), "scored":len(scored),
            "nonempty_submissions":sum(r.get("changed_parameters", 0) > 0 for r in observed),
            "unchanged_submissions":sum(r.get("changed_parameters") == 0 for r in observed),
            "editor_status":dict(Counter(row[arm]["editor_status"] for row in rows)), "usage":totals,
            "cost_upper_bound_usd":round(sum(r["cost_upper_bound_usd"] for r in observed),8),
            "mean_reward_scored":sum(r["reward"] for r in scored)/len(scored) if scored else None,
            "effective_reward_outcomes":dict(Counter(r["effective_reward_source"] for r in effective)),
            "mean_effective_reward_all_scheduled":sum(r["effective_reward"] for r in effective)/len(rows)
                                                if len(effective)==len(rows) else None,
            "counts_scored":count_totals}


def paired(rows, comparator):
    usable = [r for r in rows if "usage" in r["paged"] and "usage" in r[comparator]]
    usable_quality = [r for r in rows if "reward" in r["paged"] and "reward" in r[comparator]]
    pair = {"scheduled_tasks":len(rows),"usage_pairs":len(usable),"quality_pairs":len(usable_quality)}
    token_diffs = [r["paged"]["usage"]["total_tokens"]-r[comparator]["usage"]["total_tokens"] for r in usable]
    pair["paged_token_counts_lower_equal_higher"] = [sum(d<0 for d in token_diffs),token_diffs.count(0),sum(d>0 for d in token_diffs)]
    effective = [r for r in rows if "effective_reward" in r["paged"] and "effective_reward" in r[comparator]]
    ds = [r["paged"]["effective_reward"]-r[comparator]["effective_reward"] for r in effective]
    pair["effective_reward_pairs"] = len(ds)
    pair["paged_effective_reward_wins_ties_losses"] = [sum(d>0 for d in ds),ds.count(0),sum(d<0 for d in ds)]
    pair["mean_effective_reward_difference_all_scheduled"] = sum(ds)/len(rows) if len(ds)==len(rows) else None
    for label, field in (("all_turns","usage"),("opening","first_request"),
                         ("equal_turn_prefix","prefix_"+comparator)):
        present = [r for r in usable if field in r["paged"] and field in r[comparator]]
        p = {k:sum(r["paged"][field][k] for r in present) for k in COUNTERS}
        q = {k:sum(r[comparator][field][k] for r in present) for k in COUNTERS}
        pair[label] = {"pairs":len(present),"paged":p,comparator:q,
                       "saving_fraction":{k:1-p[k]/q[k] if q[k] else None for k in COUNTERS if k!="calls"}}
    if usable:
        p_cost=sum(r["paged"]["cost_upper_bound_usd"] for r in usable)
        q_cost=sum(r[comparator]["cost_upper_bound_usd"] for r in usable)
        pair["recorded_cost_upper_bounds_usd"]={"paged":round(p_cost,8),comparator:round(q_cost,8)}
    if usable_quality:
        for arm in ("paged",comparator):
            pair[arm+"_mean_reward_paired"]=sum(r[arm]["reward"] for r in usable_quality)/len(usable_quality)
        deltas = [r["paged"]["reward"] - r[comparator]["reward"] for r in usable_quality]
        pair["paged_wins_ties_losses"] = [sum(d>0 for d in deltas), deltas.count(0), sum(d<0 for d in deltas)]
    return pair


def summarize(root, design, grouped_design=None):
    tasks = design["tasks"]
    check(len({t["id"] for t in tasks}) == len(tasks), "duplicate predeclared task")
    check(all(set(t["arms"]) == set(ARMS) for t in tasks), "unexpected predeclared arm")
    rows=[]
    arms_to_audit = (*ARMS, "grouped") if grouped_design else ARMS
    for task in tasks:
        row={"task_id":task["id"],"family":task["family"],"variant_index":task["variant_index"]}
        sigs, records={},{}
        for arm in arms_to_audit:
            try:
                row[arm], sigs[arm], records[arm] = audit_arm(root,task,arm,design)
            except (ValueError, OSError, KeyError) as error:
                raise ValueError(f"{task['family']}/{arm}: {error}") from error
        for arm in ("eager","grouped"):
            if sigs.get(arm) and sigs["paged"]:
                check(sigs["paged"] == sigs[arm], f"{arm}: layout control shared metadata or public packet mismatch")
        common = [row[a] for a in arms_to_audit if "usage" in row[a]]
        for k in ("public_objective","public_practice","pristine_identity"):
            check(len({r[k] for r in common if k in r}) <= 1,f"{task['family']}: mismatching {k}")
        for comparator in arms_to_audit[1:]:
            if not records["paged"] or not records[comparator]:
                continue
            n=min(len(records["paged"]),len(records[comparator]))
            for a in ("paged",comparator):
                row[a]["prefix_"+comparator]=usage(records[a][:n])
        row["paged_differences"] = {}
        for comparator in arms_to_audit[1:]:
            left, right = row["paged"],row[comparator]
            if "usage" in left and "usage" in right:
                metrics = {key:left["usage"][key]-right["usage"][key] for key in COUNTERS}
                metrics["cost_upper_bound_usd"] = round(left["cost_upper_bound_usd"]-right["cost_upper_bound_usd"],8)
                if "effective_reward" in left and "effective_reward" in right:
                    metrics["effective_reward"] = left["effective_reward"]-right["effective_reward"]
                if "reward" in left and "reward" in right:
                    metrics["reward"] = left["reward"]-right["reward"]
                    metrics["counts"] = {key:left["counts"][key]-right["counts"][key] for key in left["counts"]}
                row["paged_differences"][comparator] = metrics
        rows.append(row)
    arms={a:aggregated(rows,a) for a in ARMS}
    cost=sum(a["cost_upper_bound_usd"] for a in arms.values())
    check(cost <= design["budget"]["authorized_additional_usd"]+1e-8,"cohort exceeds budget")
    report = {"tasks":rows,"arms":arms,"comparisons":{a:paired(rows,a) for a in ("eager","native")},
            "primary_complete":all(a["terminal"]==len(rows) for a in arms.values()),
            "total_recorded_cost_upper_bound_usd":round(cost,8),
            "quality_scope":"Reward and item counts include fresh scored submissions, with failures listed in all scheduled tasks. "
                            "No predeclared equivalence margin. Token-usage totals include reported attempts. "
                            "Unreported/unknown requests carry their reserved cost upper bound where known."}

    if grouped_design:
        grouped_arm = aggregated(rows,"grouped")
        grouped_cost = grouped_arm["cost_upper_bound_usd"]
        combined = round(cost+grouped_cost,8)
        cap = design["budget"]["authorized_additional_usd"]
        reservation = len(rows) * design["budget"]["max_usd_per_edit"]
        primary_edits_available = all(a["editor_results_with_usage_accounting"] == len(rows) for a in arms.values())
        gate_met = primary_edits_available and cost + reservation <= cap + 1e-8
        check(combined <= cap + 1e-8, "combined budget exceeded")
        if grouped_arm["editor_results_with_usage_accounting"]:
            check(gate_met, "grouped comparison lacks available funds or primary editor records")
        report["grouped_followup"] = {"separate_design":grouped_design["study"], "arm":grouped_arm,
                                      "complete":grouped_arm["terminal"] == len(rows),
                                      "gate_met":gate_met, "reserved_ceiling_usd":reservation,
                                      "paged_vs_grouped":paired(rows,"grouped"),
                                      "combined_cost_upper_bound_usd":combined}
    report["complete"] = not pending_arms(report)
    return report


def pending_arms(report):
    arms = dict(report["arms"])
    if "grouped_followup" in report:
        arms["grouped"] = report["grouped_followup"]["arm"]
    return [f"{name} {arm['terminal']}/{arm['scheduled']} terminal"
            for name, arm in arms.items() if arm["terminal"] < arm["scheduled"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"runs/qualified-live-manifest-v1")
    parser.add_argument("--design",type=Path,default=ROOT/"evidence/heldout-token-audit/live-design.json")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--require-complete", action="store_true",
                        help="Fail when any configured arm, including the grouped follow-up, has pending editors or grades.")
    args=parser.parse_args()
    design=json.loads(args.design.read_text())
    extra=args.design.with_name("grouped-live-design.json")
    grouped=json.loads(extra.read_text()) if extra.exists() else None
    result=summarize(args.root,design,grouped)
    if grouped:
        result["grouped_followup"]["design_sha256"]=sha(extra.read_bytes())
    result["design_sha256"]=sha(args.design.read_bytes())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("complete","primary_complete","total_recorded_cost_upper_bound_usd")}))
    if args.require_complete and not result["complete"]:
        raise SystemExit("Waiting for scheduled editors or grades (" + ", ".join(pending_arms(result)) + ")")
