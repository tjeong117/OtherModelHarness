"""Run one matched IWIBench editing trial with optional runtime SAE fitting.

Run each case and seed twice, once per arm, with identical model, turn, output
and USD limits. Give each arm a new output directory. --edit-only allows the
host to collect all edits before running --grade-only in clean native workers.
"""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path

import torch

from astra_pilot import FAMILIES, packet_for, summarize_trace
from public_feedback_trial import FeedbackSession, save
from iwi_harness.iwibench_adapter import IWIBENCH_REVISION, grade_patch
from iwi_harness.openai_responses import AstraBudget, AstraBudgetExceeded, AstraProviderError, AstraResponses
from iwi_harness.targets import load_gemma_target
from iwi_harness.updater import NoSubmission, run_updater, sae_guided_objective


class MatchedSession(FeedbackSession):
    def __init__(self, target, practice, *, arm, **kwargs):
        if arm not in ("fit", "no-fit"):
            raise ValueError("arm must be fit or no-fit")
        self.fit_enabled = arm == "fit"
        super().__init__(target, practice, sae=None, detailed_writes=True, max_forwards=1024,
                         max_fitted_dictionaries=3 if self.fit_enabled else 0,
                         max_sae_fit_steps=256 if self.fit_enabled else 0, **kwargs)

    def _tool_schemas(self, *, detailed_writes):
        schemas = super()._tool_schemas(detailed_writes=detailed_writes)
        return [tool for tool in schemas if self.fit_enabled or tool["function"]["name"] != "fit_sae"]


def edit(args, key):
    request_check = getattr(args, "request_sae_check", False)
    sae_guided = getattr(args, "sae_guided", False)
    if sae_guided and (args.arm != "fit" or request_check):
        raise ValueError("--sae-guided requires --arm fit and its own prompt condition")
    if request_check and (args.case != "suggestion" or args.arm != "fit"):
        raise ValueError("--request-sae-check requires --case suggestion --arm fit")
    budget = AstraBudget(args.budget_usd)
    complete = AstraResponses(args.model, api_key=key, budget=budget, reasoning="medium",
                              max_output_tokens=args.max_output_tokens)
    target = load_gemma_target(local_files_only=True)
    packet = packet_for(args.case, args.seed, target)
    protocol = "runtime-sae-access-v1"
    if request_check:
        protocol = "runtime-sae-instructed-v1"
        packet["objective"] += (" Fit a small SAE using public prompts of your choice. Use a learned feature to "
                                "check a hypothesis relevant to this edit, then make your best weight edit and "
                                "submit within the same turn budget.")
    elif sae_guided:
        protocol = "runtime-sae-guided-v2"
        packet["objective"] = sae_guided_objective(packet["objective"], args.max_turns)
    session = MatchedSession(target, packet.get("practice", []), arm=args.arm, artifact_dir=args.out)
    if session.practice:
        packet["public_practice_before_editing"] = session.practice_check()
    save(args.out / "public-input.json", packet)
    settings = dict(protocol=protocol, case=args.case, family=FAMILIES[args.case],
                    seed=args.seed, arm=args.arm, requested_model=args.model, reasoning="medium",
                    max_turns=args.max_turns, max_output_tokens=args.max_output_tokens,
                    budget_cap_usd=args.budget_usd, iwibench_revision=IWIBENCH_REVISION,
                    session=session.info, tools_offered=[t["function"]["name"] for t in session.tools])
    save(args.out / "protocol.json", settings)
    save(args.out / "tools.json", session.tools)
    reason = None
    try:
        run_updater(session, json.dumps(packet), complete, guide=True, max_turns=args.max_turns)
        status = "submitted"
    except NoSubmission as exc:
        status, reason = "no_submission", str(exc)
    except (AstraBudgetExceeded, AstraProviderError) as exc:
        status, reason = "provider_stopped", str(exc)
    path = args.out / "trajectory.jsonl"
    calls = summarize_trace(path) if path.is_file() else []
    report = dict(**settings, status=status, reason=reason, receipt=session.receipt, model_calls=calls,
                  successful_model_tool_counts=dict(Counter(c["tool"] for c in calls if "error" not in c)),
                  target_usage_including_host=session.runner.usage, host_public_practice_checks=len(session.checks),
                  provider=complete.records, cost_upper_bound_usd=budget.spent_upper_bound_usd)
    save(args.out / "editor-result.json", report)
    print(json.dumps({k: report[k] for k in ("arm", "case", "seed", "status", "reason",
                                             "cost_upper_bound_usd", "successful_model_tool_counts")}), flush=True)


def grade(out):
    report = json.loads((out / "editor-result.json").read_text())
    if report["receipt"]:
        result = grade_patch(out / "patch.safetensors", family=report["family"], seed=report["seed"],
                             expected_sha256=report["receipt"]["sha256"])
    else:
        result = {"status": report["status"], "reward": 0.0 if report["status"] == "no_submission" else None}
    save(out / "grade.json", result)
    print(json.dumps({"arm": report["arm"], "case": report["case"], "seed": report["seed"],
                      "grade": result["status"], "reward": result["reward"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=FAMILIES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=("fit", "no-fit"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6.2-astra")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--budget-usd", type=float, help="hard limit for this arm; required for editing")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--request-sae-check", action="store_true",
                        help="ask the updater to fit and use an SAE in a separate suggestion fit-arm trial")
    prompt.add_argument("--sae-guided", action="store_true",
                        help="use the SAE-guided editing policy in a separate fit-arm trial")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--edit-only", action="store_true")
    mode.add_argument("--grade-only", action="store_true")
    args = parser.parse_args()
    if args.sae_guided and (args.grade_only or args.arm != "fit"):
        parser.error("--sae-guided requires editing with --arm fit")
    if args.request_sae_check and (args.grade_only or args.case != "suggestion" or args.arm != "fit"):
        parser.error("--request-sae-check requires editing with --case suggestion --arm fit")
    if not args.grade_only:
        if None in (args.case, args.seed, args.arm, args.budget_usd):
            parser.error("editing requires --case, --seed, --arm, and --budget-usd")
        if args.max_turns < 1:
            parser.error("--max-turns must be positive")
        key = os.environ.pop("OPENAI_API_KEY", None)
        project_key = os.environ.pop("OPENAI_KEY", None)
        key = key or project_key
        del project_key
        if not key:
            parser.error("set OPENAI_API_KEY")
        args.out.mkdir(parents=True, exist_ok=False)
        edit(args, key)
        del key
        gc.collect()
        torch.cuda.empty_cache()
    if not args.edit_only:
        grade(args.out)


if __name__ == "__main__":
    main()
