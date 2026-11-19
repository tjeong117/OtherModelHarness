"""Compare an Astra updater using the pinned IWIBench editor with a full-tool episode.

Install IWIBench 0.2.6 at ff1bbcc. On a controller with its pinned [prime]
extra, run with --export-schemas PATH to cache the four registered wire schemas.
On the GPU host, pass --schemas PATH to reuse that cache without installing
Verifiers in the target environment. Native calibration files are host-private.
"""
import argparse
import asyncio
from collections import Counter
from decimal import Decimal
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


FAMILIES = {"alias": "alias_acquisition", "suggestion": "resistance_to_false_suggestions"}
NAMES = {"read_weights", "write_weights", "run_target", "inspect_target"}
VERIFIERS_REVISION = "5dad16d40e2d025672fdb7aa75b6540c0053e4e0"


def save(path, value):
    with path.open("x", encoding="utf-8") as fp:
        json.dump(value, fp, ensure_ascii=False, allow_nan=False, indent=2)
        fp.write("\n")


async def registered_schemas():
    from iwibench.taskset import IWIBenchTools, ToolConfig
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("iwibench-tools")
    IWIBenchTools(ToolConfig()).register(mcp)
    return [{"type": "function", "function": {"name": tool.name, "description": tool.description or "",
                                               "parameters": tool.input_schema}}
            for tool in await mcp.list_tools()]


def native_prompt(created):
    # IWIBenchTask.setup, with precisely its public fields and serialization.
    public = {k: created[k] for k in ("objective", "practice", "manifest", "token_ids", "limits")}
    return ("Edit the target model's weights to achieve the objective below. "
            "Use the four provided tools. Your final natural-language reply submits "
            "the current weight patch. It cannot change the target's prompt or inference code.\n\n"
            + json.dumps(public, ensure_ascii=False, separators=(",", ":")))


def bounded_content(text):
    # Pinned null harness uses middle-out truncation on tool outputs only.
    raw = text.encode()
    if len(raw) <= 20000:
        return text
    head = raw[:10000].decode(errors="ignore")
    tail = raw[-10000:].decode(errors="ignore")
    return (f"Warning: truncated output (original token count: {(len(text) + 3) // 4})\n"
            f"Total output lines: {text.count(chr(10)) + 1}\n\n"
            f"{head}\n[... {len(raw) - 20000} bytes truncated ...]\n{tail}")


def tool_reply(service, sid, validators, name, raw):
    from pydantic_core import to_json

    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        text = f"error: invalid JSON in tool arguments ({exc}); resend the call with valid JSON"
        return text, False
    if not isinstance(args, dict):
        return f"error: tool arguments must be a JSON object, got {type(args).__name__}; resend as an object", False
    if name not in validators:
        return f"error: unknown tool {name!r}", False
    problem = next(validators[name].iter_errors(args), None)
    if problem:
        return bounded_content(f"Input validation error: {problem.message}"), False
    # Native ToolRequest.model_dump(exclude_none=True) and service method defaults.
    payload = {k: v for k, v in args["request"].items() if v is not None}
    try:
        result = service.call(sid, name, payload)
    except (ValueError, TypeError, KeyError) as exc:
        return bounded_content(f"Error executing tool {name}: Invalid tool request: {str(exc)[:512]}"), False
    # MCP 2.2.0 renders a dictionary as one pretty-printed text item.
    return bounded_content(to_json(result, indent=2).decode()), True


def run_native_loop(service, created, tools, complete, max_turns, trace):
    from jsonschema import Draft202012Validator

    validators = {t["function"]["name"]: Draft202012Validator(t["function"]["parameters"]) for t in tools}
    if set(validators) != NAMES:
        raise ValueError("expected all four native editor tools")
    messages = [{"role": "user", "content": native_prompt(created)}]
    successful = {}
    calls = 0
    for turn in range(1, max_turns + 1):
        reply = complete(messages, tools)
        msg = reply["choices"][0]["message"]
        event = {"type": "updater", "turn": turn, "usage": reply["usage"], "response": msg}
        if getattr(complete, "records", None):
            event["provider"] = complete.records[-1]
        trace.write(json.dumps(event) + "\n")
        trace.flush()
        messages.append(msg)
        batch = msg.get("tool_calls") or []
        if not batch:
            return "natural_language", successful
        for call in batch:
            action = call["function"]
            content, ok = tool_reply(service, created["session_id"], validators, action["name"], action["arguments"])
            if ok:
                successful[action["name"]] = successful.get(action["name"], 0) + 1
            calls += 1
            trace.write(json.dumps({"type": "tool", "tool": action["name"], "ok": ok,
                                    "public_response": content}) + "\n")
            trace.flush()
            messages.append({"role": "tool", "name": action["name"], "tool_call_id": call["id"], "content": content})
            if calls >= 128:
                return "tool_limit", successful
    return "turn_limit", successful


def edit(args, tools, key):
    from iwibench.runtime import NativeTarget
    from iwibench.service import Service
    from iwibench.config import VERIFIERS_REVISION as pinned_verifiers
    from iwi_harness.iwibench_adapter import IWIBENCH_REVISION, _episode
    from iwi_harness.openai_responses import AstraBudget, AstraResponses, AstraBudgetExceeded, AstraProviderError
    from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION

    if pinned_verifiers != VERIFIERS_REVISION:
        raise ValueError("unexpected pinned Verifiers revision")
    ep = _episode(FAMILIES[args.case], args.seed, "development")
    budget = AstraBudget(args.budget_usd)
    complete = AstraResponses(args.model, api_key=key, budget=budget, reasoning="medium", max_output_tokens=4096)
    meta = {"case": args.case, "family": ep["family"], "seed": args.seed, "protocol": "native-iwibench-v1",
            "requested_model": args.model, "reasoning": "medium", "max_output_tokens": 4096,
            "max_turns": args.max_turns, "budget_cap_usd": args.budget_usd, "iwibench_revision": IWIBENCH_REVISION,
            "verifiers_revision": VERIFIERS_REVISION, "target": GEMMA_ID, "target_revision": GEMMA_REVISION,
            "target_usage_scope": "Native service counters for editor tools. Calibration and fresh grading are excluded.",
            "tools_offered": [t["function"]["name"] for t in tools],
            "schema_sha256": hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()}
    runtime = NativeTarget(GEMMA_ID, GEMMA_REVISION)
    with TemporaryDirectory(prefix="iwi-native-private-") as tmp:
        service = Service(runtime, tmp)
        created = service.create(ep["family"], GEMMA_ID, args.seed, "development", trace_id=args.out.name)
        sid = created["session_id"]
        meta["limits"] = created["limits"]
        save(args.out / "protocol.json", meta)
        save(args.out / "public-input.json", {"role": "user", "content": native_prompt(created)})
        receipt, reason, status, successes = None, None, "submitted", {}
        with (args.out / "trajectory.jsonl").open("x") as trace:
            try:
                reason, successes = run_native_loop(service, created, tools, complete, args.max_turns, trace)
            except (AstraBudgetExceeded, AstraProviderError) as exc:
                status, reason = "provider_stopped", str(exc)
                successes = dict(Counter(e["tool"] for e in service.status(sid)["trajectory"] if e.get("ok")))
        if status == "submitted":
            patch = service.submit(sid)
            local = args.out / "patch.safetensors"
            shutil.copyfile(patch["path"], local)
            receipt = {"sha256": patch["sha256"], "bytes": local.stat().st_size, "format": patch["format"],
                       "revision": patch["weight_revision"], "changed_parameters": patch["changed_parameters"]}
            save(args.out / "receipt.json", receipt)
        report = {**meta, "status": status, "reason": reason, "receipt": receipt, "provider": complete.records,
                  "successful_model_tool_counts": successes, "cost_upper_bound_usd": budget.spent_upper_bound_usd,
                  "target_usage_including_host": service.status(sid)["counters"],
                  "host_public_practice_checks": 0}
        save(args.out / "editor-result.json", report)
        return report


def recover_report(out):
    """Recover a locally submitted patch from its original disk transcript without calling an editor."""
    from safetensors import safe_open

    meta = json.loads((out / "protocol.json").read_text())
    if meta["protocol"] != "native-iwibench-v1":
        raise ValueError("expected a native-control episode")
    patch = out / "patch.safetensors"
    with patch.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with safe_open(patch, framework="pt", device="cpu") as sf:
        metadata = sf.metadata()
        if (metadata["model_id"], metadata["revision"]) != (meta["target"], meta["target_revision"]):
            raise ValueError("patch does not belong to the run's pinned checkpoint")
        sparse = json.loads(metadata.get("sparse", "[]"))
        tensors = {row[key] for row in sparse for key in ("indices", "values")}
        changed = (sum(sf.get_slice(row["indices"]).get_shape()[0] for row in sparse)
                   if tensors == set(sf.keys()) else None)
    events = [json.loads(line) for line in (out / "trajectory.jsonl").read_text().splitlines()]
    turns = [e for e in events if e["type"] == "updater"]
    calls = [e for e in events if e["type"] == "tool"]
    counts = Counter(e["tool"] for e in calls if e["ok"])
    if not turns or turns[-1]["turn"] != len(turns) or len(turns) > meta["max_turns"]:
        raise ValueError("incomplete native editor transcript")
    if not turns[-1]["response"].get("tool_calls"):
        reason = "natural_language"
    elif len(calls) >= 128:
        reason = "tool_limit"
    elif len(turns) == meta["max_turns"]:
        reason = "turn_limit"
    else:
        raise ValueError("transcript does not show normal native termination")
    provider = [e.get("provider") or {"model": None, "usage": e["usage"]} for e in turns]
    last_counters = None
    for event in reversed(calls):
        if event["ok"]:
            # Native output truncation preserves the tail, which contains the counters.
            tail = event["public_response"].rpartition('\n  "counters": ')[2]
            if tail:
                last_counters, _ = json.JSONDecoder().raw_decode(tail)
                break
    receipt = {"sha256": digest, "format": metadata["format"], "bytes": patch.stat().st_size,
               "changed_parameters": changed, "revision": counts["write_weights"]}
    report = {**meta, "status": "submitted", "reason": reason, "receipt": receipt, "provider": provider,
              "successful_model_tool_counts": dict(counts),
              "cost_upper_bound_usd": float(sum(Decimal(str(r["usage"]["cost_upper_bound_usd"])) for r in provider)),
              "target_usage_including_host": last_counters, "host_public_practice_checks": 0,
              "recovery": {"response_model_echo_available": all(r.get("model") for r in provider),
                           "usage_source": "original_trajectory", "sha256_verified": True,
                           "sha256_source": "original copied patch bytes and common fresh verifier",
                           "target_counter_source": "last successful tool reply"}}
    for file, value in (("receipt.json", receipt), ("editor-result.json", report)):
        path = out / file
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f"existing {file} differs from recovery")
        else:
            save(path, value)
    return report


def grade_result(out, report):
    from iwi_harness.iwibench_adapter import grade_patch

    if report["receipt"]:
        grade = grade_patch(out / "patch.safetensors", family=report["family"], seed=report["seed"],
                            expected_sha256=report["receipt"]["sha256"])
    else:
        grade = {"status": report["status"], "reward": None}
    save(out / "grade.json", grade)
    print(json.dumps({"case": report["case"], "seed": report["seed"], "status": report["status"],
                      "reason": report["reason"], "reward": grade["reward"],
                      "cost_upper_bound_usd": report["cost_upper_bound_usd"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-schemas", type=Path)
    parser.add_argument("--recover", type=Path, help="finish grading a previously submitted native patch offline")
    parser.add_argument("--schemas", type=Path, help="read cached tool schemas generated by --export-schemas")
    parser.add_argument("--case", choices=FAMILIES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default="gpt-6.2-astra")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--budget-usd", type=float, default=15)
    args = parser.parse_args()
    if args.recover:
        out = args.recover
        if (out / "grade.json").exists():
            parser.error("recovery destination already has a grade")
        result = out / "editor-result.json"
        report = json.loads(result.read_text()) if result.exists() else recover_report(out)
        grade_result(out, report)
        return
    if args.export_schemas:
        save(args.export_schemas, asyncio.run(registered_schemas()))
        return
    if args.case is None or args.seed is None or args.out is None:
        parser.error("--out, --case and --seed are required to run an episode")
    if not 1 <= args.max_turns <= 64:
        parser.error("IWIBench permits 1 to 64 turns")
    key = os.environ.pop("OPENAI_API_KEY", None)
    other_key = os.environ.pop("OPENAI_KEY", None)
    key = key or other_key
    del other_key
    if not key:
        parser.error("set OPENAI_API_KEY")
    tools = json.loads(args.schemas.read_text()) if args.schemas else asyncio.run(registered_schemas())
    args.out.mkdir(parents=True, exist_ok=False)
    report = edit(args, tools, key)
    del key
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    grade_result(args.out, report)


if __name__ == "__main__":
    main()
