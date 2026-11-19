"""Collect a live editor trajectory on a public qualified catalogue row, varying only manifest visibility."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from astra_pilot import summarize_trace
from public_feedback_trial import FeedbackSession, PRACTICE_TOKENS, save
from iwi_harness.openai_responses import AstraBudget, AstraBudgetExceeded, AstraProviderError, AstraResponses
from iwi_harness.session import encode_json
from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION, load_gemma_target
from iwi_harness.updater import NoSubmission, run_updater, sae_guided_objective


def public_row(path, task_id):
    if path.is_dir():
        path = path / "data/test.jsonl"
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path, filters=[("id", "=", task_id)]).to_pylist()
    elif path.suffix == ".json":
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = [rows]
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row["id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one catalogue row for {task_id}, found {len(matches)}")
    row = matches[0]
    if (row["catalog_version"], row["calibration_status"], row["model_id"], row["checkpoint_revision"]) != (
            "baseline-qualified-v1", "eligible", GEMMA_ID, GEMMA_REVISION):
        raise ValueError("expected a qualified row for the pinned Gemma checkpoint")
    return row


def edit(args, key):
    row = public_row(args.catalog, args.task_id)
    packet = {"objective": row["objective"], "parameters": json.loads(row["parameters_json"]),
              "practice": json.loads(row["public_practice_json"])}
    if args.sae_guided:
        packet["objective"] = sae_guided_objective(packet["objective"], args.max_turns)
    budget = AstraBudget(args.budget_usd)
    model = "gpt-6.2-astra"
    complete = AstraResponses(model, api_key=key, budget=budget, reasoning="medium",
                              max_output_tokens=args.max_output_tokens)
    target = load_gemma_target(local_files_only=True)
    session = FeedbackSession(target, packet["practice"], artifact_dir=args.out, detailed_writes=True,
                              max_forwards=1024)
    if session.practice:
        packet["public_practice_before_editing"] = session.practice_check()
    objective = encode_json(packet)
    save(args.out / "public-input.json", packet)
    save(args.out / "tools.json", session.tools)
    protocol = dict(protocol="qualified-layout-v1", arm=args.opening_manifest, task_id=row["id"],
                    family=row["family"], seed=row["seed"], variant_index=row["variant_index"],
                    catalog_version=row["catalog_version"], episode_sha256=row["episode_sha256"],
                    parameter_sha256=row["parameter_sha256"], source_commit=row["source_commit"],
                    verifiers_revision=row["verifiers_revision"], requested_model=model, reasoning="medium",
                    max_turns=args.max_turns, max_output_tokens=args.max_output_tokens, budget_cap_usd=args.budget_usd,
                    sae_guided=args.sae_guided, practice_tokens_per_answer=PRACTICE_TOKENS,
                    objective_sha256=hashlib.sha256(objective.encode()).hexdigest(), session=session.info)
    save(args.out / "protocol.json", protocol)
    status, reason = "infrastructure_error", None
    try:
        run_updater(session, objective, complete, opening_manifest=args.opening_manifest, max_turns=args.max_turns)
        status = "submitted"
    except NoSubmission as exc:
        status, reason = "no_submission", str(exc)
    except (AstraBudgetExceeded, AstraProviderError) as exc:
        status, reason = "provider_stopped", str(exc)
    finally:
        trace = args.out / "trajectory.jsonl"
        report = dict(**protocol, status=status, reason=reason, receipt=session.receipt,
                      model_calls=summarize_trace(trace) if trace.exists() else [],
                      target_usage_including_host=session.runner.usage,
                      host_public_practice_checks=len(session.checks), provider=complete.records,
                      cost_upper_bound_usd=budget.spent_upper_bound_usd)
        save(args.out / "editor-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True, help="official public row(s) as JSONL, JSON or Parquet")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--opening-manifest", choices=("eager", "paged", "grouped"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--sae-guided", action="store_true")
    args = parser.parse_args()
    key = os.environ.pop("OPENAI_API_KEY", None) or os.environ.pop("OPENAI_KEY", None)
    if not key or args.max_turns < 1:
        parser.error("set OPENAI_API_KEY and provide at least one turn")
    args.out.mkdir(exist_ok=False, parents=True)
    report = edit(args, key)
    print(json.dumps({k: report[k] for k in ("arm", "task_id", "status", "reason", "cost_upper_bound_usd")}))


if __name__ == "__main__":
    main()
