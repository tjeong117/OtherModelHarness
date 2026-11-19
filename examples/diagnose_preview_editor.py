"""Audit Qwen's native decode boundaries and failures from the public traces."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit_trial(path, design, eos_ids):
    result = json.loads((path / "editor-result.json").read_text())
    native = read_lines(path / "native-generations.jsonl")
    trace = read_lines(path / "trajectory.jsonl")
    responses = [e for e in trace if e["type"] == "updater"]
    if len(native) != len(responses) or len(native) != result["turns"]:
        raise ValueError(f"{path.name}: generation count mismatch")
    usage = Counter()
    ceilings = malformed = incomplete = plain = 0
    limit = design["editor_decoding"]["max_new_tokens"]
    for gen, event in zip(native, responses, strict=True):
        if gen["usage"] != event["usage"] or len(gen["generated_ids"]) != gen["usage"]["completion_tokens"]:
            raise ValueError(f"{path.name}: generation token accounting mismatch")
        usage.update(gen["usage"])
        capped = len(gen["generated_ids"]) == limit and gen["generated_ids"][-1] not in eos_ids
        ceilings += capped
        malformed += bool(gen.get("parse_error"))
        incomplete += bool(capped and gen.get("parse_error") == "incomplete native tool-call delimiter")
        plain += event["outcome"] == "text" and not gen.get("parse_error")
    if dict(usage) != result["editor_usage"]:
        raise ValueError(f"{path.name}: total editing usage mismatch")

    last = responses[-1]["response"]
    last_tools = [c["function"]["name"] for c in last.get("tool_calls", [])]
    if result["status"] == "submitted":
        terminal = "explicit_submission"
    elif native[-1].get("parse_error"):
        terminal = "malformed_native_output"
    elif not last_tools:
        terminal = "text_without_tool_call"
    elif "submit_patch" in last_tools:
        terminal = "submission_tool_failed_at_turn_limit"
    elif len(responses) == design["editor_decoding"]["max_turns"]:
        terminal = "last_turn_used_for_other_tools"
    else:
        raise ValueError(f"{path.name}: unknown terminal condition")

    first_editor = next(i for i, e in enumerate(trace) if e["type"] == "updater")
    errors = [e for e in trace[first_editor:] if e["type"] == "tool" and "error" in e]
    tool_errors = Counter(e["tool"] for e in errors)
    token_id_rejections = sum(e["tool"] == "test_behavior" and e["error"].startswith("token_ids")
                              for e in errors)
    return dict(slug=result["slug"], arm=result["arm"], turns=len(native),
                generations_at_ceiling=ceilings, malformed_generations=malformed,
                incomplete_delimiters_at_ceiling=incomplete, plain_text_generations=plain,
                terminal=terminal, last_tools=last_tools, tool_errors=tool_errors,
                test_behavior_token_id_rejections=token_id_rejections)


def audit(path, allow_partial):
    design_path = path / "design.json"
    design = json.loads(design_path.read_text())
    env = json.loads((path / "environment.json").read_text())
    eos_ids = set(env["editor_eos_token_ids"])
    rows = []
    for trial in design["attempts"]:
        directory = path / "trials" / trial["slug"]
        if not (directory / "editor-result.json").exists():
            if allow_partial:
                continue
            raise ValueError(f"Missing completed editor trace {trial['slug']}")
        rows.append(audit_trial(directory, design, eos_ids))
    by_arm = {}
    for arm in design["arms"]:
        sample = [r for r in rows if r["arm"] == arm]
        by_arm[arm] = dict(
            attempts=len(sample),
            **{k: sum(r[k] for r in sample) for k in
               ("turns","generations_at_ceiling","malformed_generations","incomplete_delimiters_at_ceiling",
                "plain_text_generations","test_behavior_token_id_rejections")},
            terminal_reason=Counter(r["terminal"] for r in sample),
            tool_errors_by_function=sum((r["tool_errors"] for r in sample),Counter()),
        )
    return dict(design_sha256=sha256(design_path.read_bytes()).hexdigest(),
                editor_model=design["editor"], editor_revision=design["editor_revision"],
                counts="Actual Qwen tokenizer IDs; cap hits exclude generations that ended on EOS. "
                       "Malformed native blocks remain text and terminate the standard editing loop. "
                       "Tool errors exclude host opening-manifest reads.", by_arm=by_arm, trials=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = audit(args.path, args.allow_partial)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["by_arm"], indent=2))
