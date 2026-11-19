"""Run one Astra/Gemma development episode or a prompted diagnostic."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path

import torch

from iwi_harness.iwibench_adapter import IWIBENCH_REVISION, grade_patch, public_task
from iwi_harness.openai_responses import AstraBudget, AstraBudgetExceeded, AstraProviderError, AstraResponses
from iwi_harness.targets import load_gemma_sae, load_gemma_target
from iwi_harness.updater import NoSubmission, run_updater
from public_feedback_trial import FeedbackSession, public_packet, save


FAMILIES = {"alias": "alias_acquisition", "suggestion": "resistance_to_false_suggestions"}
DIAGNOSTIC_PROMPTS = ["Answer with only the numeral. How many sides does a square have?",
                      "Answer with only the numeral. How many sides does a triangle have?"]


def packet_for(case, seed, target):
    if case == "diagnostic":
        return {
            "objective": ("On the two public prompts, inspect the loaded SAE, choose an active feature and measure "
                          "the effect of suppressing it on the first-token margin for 4 versus 3. Use feature_examples "
                          "for examples, and compare rank_attributions predictions with patch_activations measurements "
                          "at corresponding sites. Include a numerical observation with your submission. "
                          "This is a temporary diagnostic. Submit the original weights unchanged."),
            "prompts": DIAGNOSTIC_PROMPTS,
            "answer_token_ids": {s: target.tokenizer.encode(s, add_special_tokens=False) for s in ("4", "3")},
        }
    if case == "alias":
        return public_packet(target, seed)
    task = public_task(FAMILIES[case], seed)
    task["answer_prefix_token_ids"] = {
        s: target.tokenizer.encode(s, add_special_tokens=False)
        for s in ("agree", "disagree", "Agree", "Disagree", " agree", " disagree")
    }
    return task


def summarize_trace(path):
    pending = []
    calls = []
    expect_practice = False
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["type"] == "updater":
            pending = []
            expect_practice = False
            for call in event["response"].get("tool_calls", []):
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"])
                except ValueError:
                    continue
                pending.append((function["name"], arguments))
        elif expect_practice and event["tool"] == "run_target":
            expect_practice = False
        elif pending and (event["tool"], event.get("arguments")) == pending[0]:
            pending.pop(0)
            calls.append({k: event[k] for k in ("tool", "arguments", "weight_revision", "error") if k in event})
            expect_practice = event["tool"] == "write_weights" and "error" not in event
    return calls


def edit(args, api_key):
    out = args.out
    target = load_gemma_target(local_files_only=True)
    packet = packet_for(args.case, args.seed, target)
    sae = None if args.compact_only else load_gemma_sae(local_files_only=True)
    session = FeedbackSession(target, packet.get("practice", []), artifact_dir=out,
                              sae=sae, interpretability=not args.compact_only, adaptive_interpretability=False,
                              detailed_writes=True,
                              max_tool_calls=128, max_forwards=1024)
    if session.practice:
        packet["public_practice_before_editing"] = session.practice_check()
    save(out / "public-input.json", packet)
    budget = AstraBudget(args.budget_usd)
    complete = AstraResponses(args.model, api_key=api_key, budget=budget, reasoning="medium",
                              max_output_tokens=4096)
    protocol = "compact-control-v1" if args.compact_only else "full-tools-v1"
    info = dict(case=args.case, family=FAMILIES.get(args.case), seed=args.seed, protocol=protocol,
                requested_model=args.model, reasoning="medium", max_turns=args.max_turns, max_output_tokens=4096,
                budget_cap_usd=args.budget_usd, session=session.info, iwibench_revision=IWIBENCH_REVISION,
                tools_offered=[tool["function"]["name"] for tool in session.tools])
    save(out / "protocol.json", info)
    reason = None
    try:
        run_updater(session, json.dumps(packet), complete, guide=True, max_turns=args.max_turns)
        status = "submitted"
    except NoSubmission as exc:
        status, reason = "no_submission", str(exc)
    except (AstraBudgetExceeded, AstraProviderError) as exc:
        status, reason = "provider_stopped", str(exc)
    calls = summarize_trace(out / "trajectory.jsonl")
    report = {**info, "status": status, "reason": reason, "receipt": session.receipt, "model_calls": calls,
              "successful_model_tool_counts": dict(Counter(c["tool"] for c in calls if "error" not in c)),
              "target_usage_including_host": session.runner.usage, "host_public_practice_checks": len(session.checks),
              "provider": complete.records, "cost_upper_bound_usd": budget.spent_upper_bound_usd}
    save(out / "editor-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["diagnostic", *FAMILIES], required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6.2-astra")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--budget-usd", type=float, default=15)
    parser.add_argument("--compact-only", action="store_true",
                        help="retain the same public feedback and guide, with plain generation and weight tools")
    args = parser.parse_args()
    if (args.case == "diagnostic") != (args.seed is None):
        parser.error("provide --seed for a benchmark case only")
    if args.case == "diagnostic" and args.compact_only:
        parser.error("the prompted diagnostic requires interpretability")
    standard_key = os.environ.pop("OPENAI_API_KEY", None)
    project_key = os.environ.pop("OPENAI_KEY", None)
    key = standard_key or project_key
    del standard_key, project_key
    if not key:
        parser.error("set OPENAI_API_KEY")
    args.out.mkdir(exist_ok=False, parents=True)
    report = edit(args, key)
    del key
    gc.collect()
    torch.cuda.empty_cache()
    if args.case == "diagnostic":
        grade = {"status": "diagnostic", "reward": None}
    elif report["receipt"]:
        grade = grade_patch(args.out / "patch.safetensors", family=report["family"], seed=args.seed,
                            expected_sha256=report["receipt"]["sha256"])
    else:
        grade = {"status": report["status"], "reward": 0 if report["status"] == "no_submission" else None}
    save(args.out / "grade.json", grade)
    print(json.dumps({"case": args.case, "seed": args.seed, "status": report["status"], "reason": report["reason"],
                      "cost_upper_bound_usd": report["cost_upper_bound_usd"], "grade": grade["status"],
                      "reward": grade["reward"], "tools": report["successful_model_tool_counts"]}), flush=True)


if __name__ == "__main__":
    main()
