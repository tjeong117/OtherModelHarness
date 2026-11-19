"""Reconstruct editor behavior in the five-task held-out pilot, without running a target or grading."""

import argparse
from collections import Counter, deque
import json
from pathlib import Path

from summarize_astra import trace


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence/heldout-token-audit"


def native_handles(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "handle" and isinstance(value, str):
                yield value, path
            else:
                yield from native_handles(value, (*path, key))
    elif isinstance(obj, list):
        for item in obj:
            yield from native_handles(item, path)


def native_behavior(events):
    pending = deque()
    requested, successful, errors, probe_kinds = (Counter() for _ in range(4))
    captures, writes = {}, []
    call = 0
    for event in events:
        if event["type"] == "updater":
            assert not pending
            for choice in event["response"].get("tool_calls") or []:
                fn = choice["function"]
                pending.append((fn["name"], json.loads(fn["arguments"])["request"]))
                requested[fn["name"]] += 1
            continue
        if event["type"] != "tool":
            continue
        call += 1
        name, args = pending.popleft()
        assert name == event["tool"]
        if not event["ok"]:
            errors[name] += 1
            continue
        successful[name] += 1
        if name == "inspect_target":
            probe_kinds.update(args["kinds"])
            try:
                result = json.loads(event["public_response"])
            except json.JSONDecodeError:
                # Native clips long display responses. A subsequent write using an
                # unresolvable operand will fail the name lookup below.
                continue
            for handle, path in native_handles(result):
                captures[handle] = (call, path[-1])
        if name == "write_weights":
            sources = sorted({captures[h] for h, _ in native_handles(args["value"])})
            writes.append({
                "call": call, "tensor": args["tensor"], "operation": args["operation"],
                "capture_sources": [{"call": n, "kind": kind} for n, kind in sources],
            })
    assert not pending
    return {
        "requested": requested, "successful": successful, "errors": errors,
        "inspection_kinds": probe_kinds, "writes": writes,
    }


def harness_behavior(path, events, editor):
    issues = []
    counts, host, _, _, lineage, _ = trace(path, False, issues)
    assert not issues and not counts["unconfirmed"], (path, issues, counts)
    assert counts["successful"] == Counter(c["tool"] for c in editor["model_calls"] if "error" not in c)
    sources = {w["call"]: w["sources"] for w in lineage["writes"]}
    assert all(w["unresolved_references"] == 0 for w in lineage["writes"])
    fits, interventions, writes = [], [], []
    turn = 0
    for e in events:
        if e["type"] == "updater":
            turn += 1
        if e["type"] != "tool" or "error" in e:
            continue
        args, result = e["arguments"], e["result"]
        if e["tool"] == "fit_sae":
            fits.append({key: result[key] for key in ("layer", "n_features", "top_k", "positions", "steps")})
        if e["tool"] == "inspect_target" and args.get("intervention"):
            assert args["intervention"]["scale"] == 0 and result["probe"]["kind"] == "sae"
            baseline = result["baseline_outputs"]
            treated = result["outputs"]
            active = [
                any(obs["pre_suppression_activation"] != 0 for obs in out["observations"])
                for out in baseline
            ]
            assert bool(result.get("suppression_feedback")) == (not any(active))
            interventions.append({
                "call": e["call"], "updater_turn": turn, "layer": result["probe"]["layer"],
                "feature": args["intervention"]["feature"],
                "active_by_prompt": active, "ordered_margin_changes": result["margin_changes"],
                "argmax_changed_by_prompt": [
                    b["predicted_next_token_id"] != s["predicted_next_token_id"]
                    for b, s in zip(baseline, treated, strict=True)
                ],
                "received_inactive_feature_feedback": bool(result.get("suppression_feedback")),
            })
        if e["tool"] == "write_weights":
            captures = {(s["call"], s["kind"], s.get("layer")) for s in sources.get(e["call"], [])}
            writes.append({
                "call": e["call"], "updater_turn": turn, "tensor": args["tensor"], "operation": args["operation"],
                "capture_sources": [
                    {"call": n, "kind": kind, "layer": layer} for n, kind, layer in sorted(captures)
                ],
            })
    manifest_pages = sum(
        c["tool"] == "read_weights" and not (c["arguments"].get("tensor") or c["arguments"].get("handle"))
        for c in editor["model_calls"] if "error" not in c
    )
    return {
        "requested": counts["requested"], "successful": counts["successful"], "errors": counts["errors"],
        "host_tools": host, "probe_modes": counts["successful_probe_modes"], "fits": fits,
        "suppressions": interventions, "editor_manifest_pages": manifest_pages, "writes": writes,
    }


def arm_totals(rows):
    totals = {
        "runs": len(rows), "submitted": sum(r["status"] == "submitted" for r in rows),
        "successful_writes": sum(len(r["writes"]) for r in rows),
    }
    for kind in ("residual", "sae"):
        totals[f"writes_reusing_{kind}_capture"] = sum(
            any(s["kind"] == kind for s in w["capture_sources"]) for r in rows for w in r["writes"]
        )
    if "fits" in rows[0]:
        totals.update(
            fits=sum(len(r["fits"]) for r in rows),
            runs_with_sae_contrast=sum(r["probe_modes"].get("contrast_target.sae.observation", 0) > 0 for r in rows),
            suppressions=sum(len(r["suppressions"]) for r in rows),
            runs_with_active_feature=sum(any(any(s["active_by_prompt"]) for s in r["suppressions"]) for r in rows),
            runs_with_margin_change=sum(any(any(s["ordered_margin_changes"]) for s in r["suppressions"]) for r in rows),
            runs_with_margin_change_returned_before_first_write=sum(
                any(any(s["ordered_margin_changes"]) and s["updater_turn"] < r["writes"][0]["updater_turn"]
                    for s in r["suppressions"]) for r in rows if r["writes"]
            ),
            runs_with_argmax_change=sum(any(any(s["argmax_changed_by_prompt"]) for s in r["suppressions"]) for r in rows),
            inactive_feature_feedback=sum(s["received_inactive_feature_feedback"] for r in rows for s in r["suppressions"]),
            editor_manifest_pages=sum(r["editor_manifest_pages"] for r in rows),
        )
    return totals


def audit(trials):
    tasks = json.loads((EVIDENCE / "live-design.json").read_text())["tasks"]
    arms = ("paged", "eager", "grouped", "native")
    rows = []
    for task in tasks:
        for arm in arms:
            folder = trials / task["family"] / arm
            editor = json.loads((folder / "editor-result.json").read_text())
            assert (editor["task_id"], editor["family"], editor["arm"]) == (task["id"], task["family"], arm)
            assert editor["status"] in ("submitted", "no_submission")
            path = folder / "trajectory.jsonl"
            events = [json.loads(line) for line in path.read_text().splitlines()]
            row = {
                "family": task["family"], "task_id": task["id"], "arm": arm, "status": editor["status"],
                "reason": editor["reason"], "receipt_changed_parameters": (editor["receipt"] or {}).get("changed_parameters"),
                "turns": sum(e["type"] == "updater" for e in events),
                "turns_with_public_assistant_text": sum(
                    bool(e["response"].get("content")) for e in events if e["type"] == "updater"
                ),
                "host_public_practice_checks": editor["host_public_practice_checks"],
            }
            behavior = native_behavior(events) if arm == "native" else harness_behavior(path, events, editor)
            if arm == "native":
                assert behavior["successful"] == editor["successful_model_tool_counts"]
            else:
                assert editor["sae_guided"]
            rows.append({**row, **behavior})
    return {"by_arm": {arm: arm_totals([r for r in rows if r["arm"] == arm]) for arm in arms}, "episodes": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=EVIDENCE / "live-trials")
    parser.add_argument("--out", type=Path, default=EVIDENCE / "tool-behavior.json")
    args = parser.parse_args()
    args.out.write_text(json.dumps(audit(args.trials), indent=2) + "\n")
