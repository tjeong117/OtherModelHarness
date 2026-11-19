"""Run the four-tool native editor on an exact pinned IWIBench test catalogue task.

Native service admission and submission run locally. Qualified semantic predicates
that require a remote judge are refused here. Grade the saved patch separately.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from astra_native_control import VERIFIERS_REVISION, native_prompt, run_native_loop, save
from iwi_harness.openai_responses import AstraBudget, AstraBudgetExceeded, AstraProviderError, AstraResponses


ROOT = Path(__file__).resolve().parents[1]
HUB_REVISION = "300a9d7271d9e3a9dd5bfc14edcbceccb66d5966"
TEST_SHA = "d0a12f4d5ec9128ee5303739eb9c0a7331b1ec1e2225f8f7e256e95af0fa598b"
SCHEMA_SHA = "b9213c41c66cda048f6f69617fa42b6805951d16c072631067dc845f1f7a96b0"


def select_task(path, task_id):
    import iwibench
    from qualified_eval import assert_deterministic, regenerate_row, verify_install

    verify_install(Path(iwibench.__file__).resolve().parents[2])
    raw = (path / "data/test.jsonl" if path.is_dir() else path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEST_SHA:
        raise ValueError("Test catalogue digest differs from the pinned public release")
    matches = [row for line in raw.splitlines() if (row := json.loads(line))["id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one pinned test task for {task_id!r}")
    row = matches[0]
    episode = regenerate_row(row)
    if episode["seed"] != row["seed"] or row["catalog_version"] != "baseline-qualified-v1":
        raise ValueError("Qualified episode and public row identities disagree")
    assert_deterministic(episode)
    return row


def forbid_remote_judge(_):
    raise RuntimeError("This qualified task requires a semantic-judge lookup during native admission")


def edit(args, row, tools, key):
    from iwibench.config import VERIFIERS_REVISION as runtime_verifiers
    from iwibench.runtime import NativeTarget
    from iwibench.service import Service
    from iwibench.terra_semantics import JudgedChecker

    if row["verifiers_revision"] != VERIFIERS_REVISION or runtime_verifiers != VERIFIERS_REVISION:
        raise ValueError("Unexpected Verifiers revision")
    budget = AstraBudget(args.budget_usd)
    complete = AstraResponses(args.model, api_key=key, budget=budget, reasoning="medium",
                              max_output_tokens=args.max_output_tokens)
    meta = {"protocol": "qualified-stock-native-v1", "arm": "native", "task_id": row["id"],
            "family": row["family"], "seed": row["seed"], "variant_index": row["variant_index"],
            "dataset_revision": HUB_REVISION, "dataset_partition": "test", "catalog_version": row["catalog_version"],
            "episode_sha256": row["episode_sha256"], "parameter_sha256": row["parameter_sha256"],
            "verifiers_revision": VERIFIERS_REVISION, "target": row["model_id"], "target_revision": row["checkpoint_revision"],
            "requested_model": args.model, "reasoning": "medium", "max_turns": args.max_turns,
            "max_output_tokens": args.max_output_tokens, "budget_cap_usd": args.budget_usd,
            "tools_offered": [tool["function"]["name"] for tool in tools], "schemas_file_sha256": SCHEMA_SHA}
    runtime = NativeTarget(row["model_id"], row["checkpoint_revision"])
    with TemporaryDirectory(prefix="iwi-qualified-native-private-") as tmp:
        service = Service(runtime, tmp)
        service.semantic_checker = JudgedChecker(forbid_remote_judge)
        created = service.create(row["family"], row["model_id"], row["seed"], "development",
                                 trace_id=args.out.name, variant_index=row["variant_index"],
                                 catalog_version=row["catalog_version"])
        if created["objective"] != row["objective"] or created["practice"] != json.loads(row["public_practice_json"]):
            raise ValueError("Native service returned a different public task")
        meta["limits"] = created["limits"]
        save(args.out / "protocol.json", meta)
        save(args.out / "public-input.json", {"role": "user", "content": native_prompt(created)})
        save(args.out / "tools.json", tools)
        sid = created["session_id"]
        status, reason, receipt, successes = "controller_error", None, None, {}
        try:
            with (args.out / "trajectory.jsonl").open("x") as trace:
                reason, successes = run_native_loop(service, created, tools, complete, args.max_turns, trace)
            patch = service.submit(sid)
            dest = args.out / "patch.safetensors"
            shutil.copyfile(patch["path"], dest)
            with dest.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != patch["sha256"]:
                    raise ValueError("Exported patch hash differs from native service submission")
            receipt = {"sha256": patch["sha256"], "bytes": dest.stat().st_size, "format": patch["format"],
                       "revision": patch["weight_revision"], "changed_parameters": patch["changed_parameters"]}
            save(args.out / "receipt.json", receipt)
            status = "submitted"
        except (AstraBudgetExceeded, AstraProviderError) as exc:
            status, reason = "provider_stopped", str(exc)
            successes = dict(Counter(e["tool"] for e in service.status(sid)["trajectory"] if e.get("ok")))
        finally:
            report = {**meta, "status": status, "reason": reason, "receipt": receipt, "provider": complete.records,
                      "successful_model_tool_counts": successes, "cost_upper_bound_usd": budget.spent_upper_bound_usd,
                      "target_usage_including_host": service.status(sid)["counters"], "host_public_practice_checks": 0}
            save(args.out / "editor-result.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--model", default="gpt-6.2-astra")
    parser.add_argument("--schemas", type=Path, default=ROOT / "runs/astra-native-tool-schemas.json")
    args = parser.parse_args()
    if not 1 <= args.max_turns <= 64 or not 1 <= args.max_output_tokens <= 4096:
        parser.error("Invalid native turn count or output-token cap")
    key = os.environ.pop("OPENAI_API_KEY", None)
    project_key = os.environ.pop("OPENAI_KEY", None)
    key = key or project_key
    del project_key
    if not key:
        parser.error("Set OPENAI_API_KEY")
    row = select_task(args.catalog, args.task_id)
    raw = args.schemas.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCHEMA_SHA:
        parser.error("Schema cache differs from the validated native four-tool export")
    args.out.mkdir(parents=True, exist_ok=False)
    report = edit(args, row, json.loads(raw), key)
    print(json.dumps({k: report[k] for k in ("task_id", "status", "reason", "cost_upper_bound_usd")}))


if __name__ == "__main__":
    main()
