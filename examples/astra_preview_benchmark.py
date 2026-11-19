"""Execute and score the frozen matched hosted cohort, with a durable shared spend cap."""

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
from time import monotonic

from qualified_eval import grade_qualified_patch, verify_install
from run_preview_benchmark import public_packet, save
from iwi_harness.openai_responses import AstraBudget, AstraBudgetExceeded, AstraProviderError, AstraResponses
from iwi_harness.session import EditingSession, encode_json
from iwi_harness.targets import load_gemma_target
from iwi_harness.updater import NoSubmission, run_updater


@contextmanager
def lock(path, *, blocking=True):
    with path.open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


class PersistentBudget(AstraBudget):
    """Record reservations before network I/O so interrupted requests remain charged."""

    def __init__(self, cap, path):
        self.path = path
        super().__init__(cap)
        if path.exists():
            self._used = json.loads(path.read_text())["used_or_reserved_microdollars"]
            if type(self._used) is not int or not 0 <= self._used <= self._limit:
                raise ValueError("invalid persisted budget")

    def _persist(self):
        save(self.path, dict(cap_microdollars=self._limit, used_or_reserved_microdollars=self._used))

    def _reserve(self, amount):
        super()._reserve(amount)
        self._persist()

    def _settle(self, reserved, actual_bound):
        super()._settle(reserved, actual_bound)
        self._persist()


def edit(spec, trial, row, dest, key, budget):
    start, initial_cost = monotonic(), budget.spent_upper_bound_usd
    target = load_gemma_target(local_files_only=True)
    internal = trial["arm"] == "D"
    session = EditingSession(
        target,
        artifact_dir=dest,
        experiment_tools_only=True,
        detailed_writes=True,
        behavior_checks=True,
        interpretability=internal,
        internal_measurements=internal,
        edit_preview=True,
        **spec["target_limits"],
    )
    packet = public_packet(row, target.tokenizer, spec)
    save(dest / "public-input.json", packet)
    save(dest / "tools.json", session.tools)
    decoding = spec["editor_decoding"]
    callback = AstraResponses(
        spec["editor"],
        api_key=key,
        budget=budget,
        reasoning=decoding["reasoning"],
        max_output_tokens=decoding["max_output_tokens"],
    )
    reason = None
    try:
        run_updater(
            session,
            encode_json(packet),
            callback,
            max_turns=decoding["max_turns"],
            opening_manifest=spec["opening_manifest"],
        )
        status = "submitted"
    except NoSubmission as exc:
        status, reason = "no_submission", str(exc)
    except AstraBudgetExceeded as exc:
        status, reason = "budget_stopped", str(exc)
    except AstraProviderError as exc:
        reason = str(exc)
        status = "no_submission" if "(max_output_tokens)" in reason else "provider_stopped"
    turns = [record["usage"] for record in callback.records]
    usage = {k: sum(t[k] for t in turns) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    result = dict(
        **trial,
        status=status,
        reason=reason,
        receipt=session.receipt,
        editor_usage=usage,
        provider=callback.records,
        turns=len(turns),
        target_usage=session.runner.usage,
        elapsed_seconds=round(monotonic() - start, 3),
        usd_upper_bound=round(budget.spent_upper_bound_usd - initial_cost, 6),
        cumulative_usd_upper_bound=budget.spent_upper_bound_usd,
        tools_offered=[t["function"]["name"] for t in session.tools],
    )
    save(dest / "editor-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("edit", "grade"))
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.design.read_text())
    catalog = verify_install(args.catalog)
    raw = (catalog / "data/test.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["test_file_sha256"]:
        raise ValueError("qualified catalog mismatch")
    from iwi_harness import weights

    for name, sha in spec["qwen_harness_sha256"].items():
        if hashlib.sha256(Path(weights.__file__).with_name(name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"matched harness source changed: {name}")
    rows = {row["id"]: row for line in raw.splitlines() if (row := json.loads(line))}
    key = os.environ.pop("OPENAI_API_KEY", None)
    if args.action == "edit" and not key:
        raise ValueError("hosted editor requires an API key")
    trials = args.out / "trials"
    trials.mkdir(parents=True, exist_ok=True)
    with lock(args.out / (args.action + ".lock"), blocking=False):
        budget = PersistentBudget(spec["budget_usd"], args.out / "budget.json")
        for trial in spec["attempts"]:
            dest = trials / trial["slug"]
            done = dest / "editor-result.json"
            print(f"{args.action} {trial['order']}/16 {dest.name}", flush=True)
            if args.action == "edit":
                if done.exists():
                    continue
                if dest.exists():
                    # A lost process may have already made paid requests. Never reroll it.
                    save(
                        done,
                        dict(
                            **trial,
                            status="interrupted",
                            reason="prior started attempt",
                            receipt=None,
                            usd_upper_bound=None,
                            provider=[],
                            editor_usage={},
                            turns=0,
                        ),
                    )
                    continue
                import torch

                with lock(args.gpu_lock):
                    result = edit(spec, trial, rows[trial["task_id"]], dest, key, budget)
                    gc.collect()
                    torch.cuda.empty_cache()
                print(
                    json.dumps(
                        {k: result[k] for k in ("slug", "status", "turns", "cumulative_usd_upper_bound")}
                    ),
                    flush=True,
                )
                if result["status"] in ("provider_stopped", "budget_stopped"):
                    break
            elif done.exists() and not (dest / "grade.json").exists():
                result = json.loads(done.read_text())
                if result["status"] == "submitted":
                    with lock(args.gpu_lock):
                        grade = grade_qualified_patch(
                            catalog,
                            trial["family"],
                            trial["variant_index"],
                            dest / "patch.safetensors",
                            result["receipt"]["sha256"],
                            work=args.cache,
                        )
                elif result["status"] == "no_submission":
                    grade = dict(status="no_submission", reward=0.0, metrics=None)
                else:
                    grade = dict(status=result["status"], reward=None, metrics=None)
                save(dest / "grade.json", {**trial, **grade})
                print(json.dumps({k: grade[k] for k in ("status", "reward")}), flush=True)


if __name__ == "__main__":
    main()
