"""Two public-feedback development trials with a frozen protocol on the cached A100 stack.

Run python examples/public_feedback_trial.py --out runs/public-feedback, with the pinned
IWIBench checkout on PYTHONPATH and both Gemma and Qwen checkpoints already cached.
"""
import argparse
import gc
import json
from pathlib import Path
import re
from time import perf_counter

import torch
from iwibench.families import episode, public_episode

from iwi_harness import EditingSession
from iwi_harness.iwibench_adapter import grade_patch
from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION, load_gemma_sae, load_gemma_target
from iwi_harness.updater import NoSubmission, chat_completions, run_updater
from qwen_http_trial import MAX_NEW_TOKENS, MAX_TURNS, MODEL, REVISION, local_qwen_endpoint


FAMILY = "alias_acquisition"
SEEDS = (3, 1)
PRACTICE_TOKENS = 12
PROTOCOL = {
    "family": FAMILY, "split": "development", "seeds": list(SEEDS),
    "updater": MODEL, "updater_revision": REVISION, "target": GEMMA_ID, "target_revision": GEMMA_REVISION,
    "max_updater_turns": MAX_TURNS, "max_updater_tokens_per_turn": MAX_NEW_TOKENS, "decoding": "greedy",
    "guide": True, "detailed_writes": True, "max_public_answer_tokens": PRACTICE_TOKENS,
    "feedback": "Generate answers on the published practice prompts before editing and after each successful write. "
                "Show public answers together with generations; all feedback consumes the standard session budget.",
    "alignment": "Report target-tokenizer positions and IDs for the alias in public practice and the announced "
                 "referent in the same sentence. Omit standalone word spellings and private occurrence records.",
    "evaluation": "Run both updater trajectories before fresh native grading. A missing submission earns zero. "
                  "One trajectory per seed. This is a joint exploratory interface change after reviewing seed 3.",
}


def save(path, value):
    with path.open("x", encoding="utf-8") as fp:
        json.dump(value, fp, indent=2, allow_nan=False)
        fp.write("\n")


def display_span(target, prompt, word, match):
    tokenizer = target.tokenizer
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
    offset = rendered.index(prompt)
    alignment = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = alignment["input_ids"]
    if ids != target.encode(prompt):
        raise ValueError("Token span alignment differs from the target's native prompt encoding")
    positions = [i for i, (a, b) in enumerate(alignment["offset_mapping"])
                 if a < offset + match.end() and b > offset + match.start()]
    if not positions:
        raise ValueError("The public word has no aligned tokens in the rendered prompt")
    return {"prompt": prompt, "word": word, "character_range": [match.start(), match.end()],
            "token_positions_in_rendered_prompt": positions, "token_ids": [ids[i] for i in positions]}


def public_packet(target, seed):
    # Consume only the published editor projection of the native episode.
    packet = public_episode(episode(FAMILY, seed, "development", model_id=GEMMA_ID))
    spans = []
    for binding in packet["parameters"]["bindings"]:
        alias, referent = binding["alias"], binding["referent"]
        for item in packet["public_practice"]:
            prompt = item["prompt"]
            for match in re.finditer(r"\b" + re.escape(alias) + r"\b", prompt):
                paired = prompt[:match.start()] + referent + prompt[match.end():]
                other = re.compile(re.escape(referent)).match(paired, match.start())
                spans.append({"practice": display_span(target, prompt, alias, match),
                              "matched_public_word": display_span(target, paired, referent, other)})
    if not spans:
        raise ValueError("Expected public alias examples in this development family")
    return {"objective": packet["objective"], "practice": packet["public_practice"],
            "target_tokenization_of_public_text": spans}


class FeedbackSession(EditingSession):
    """Augment successful write replies with the same public behavioral check."""

    def __init__(self, target, practice, **kwargs):
        super().__init__(target, **kwargs)
        self.practice = practice
        self.checks = []

    def practice_check(self):
        start = perf_counter()
        usage_before = self.runner.usage.copy()
        result = super().call("run_target", {"prompts": [row["prompt"] for row in self.practice],
                                             "max_new_tokens": PRACTICE_TOKENS})
        cases = [{"prompt": row["prompt"], "accepted_answers": row["answers"], "generated": output["text"],
                  "exact_after_strip": output["text"].strip() in row["answers"]}
                 for row, output in zip(self.practice, result["outputs"], strict=True)]
        check = {"weight_revision": self.store.revision, "cases": cases,
                 "cost": {key: value - usage_before[key] for key, value in self.runner.usage.items()},
                 "wall_seconds": perf_counter() - start}
        self.checks.append(check)
        return cases

    def call(self, name, arguments):
        result = super().call(name, arguments)
        if name == "write_weights":
            try:
                result = {**result, "public_practice_after_write": self.practice_check()}
            except ValueError as exc:
                result = {**result, "public_practice_error": str(exc)}
        return result


def edit_seed(out, seed):
    out.mkdir()
    target = load_gemma_target(local_files_only=True)
    packet = public_packet(target, seed)
    session = FeedbackSession(target, packet["practice"], artifact_dir=out,
                              sae=load_gemma_sae(local_files_only=True), detailed_writes=True)
    packet["public_practice_before_editing"] = session.practice_check()
    save(out / "public-input.json", packet)
    with local_qwen_endpoint(out) as (url, _, turns):
        try:
            receipt = run_updater(session, json.dumps(packet), chat_completions(url, MODEL),
                                  max_turns=MAX_TURNS, guide=True)
            reason = None
        except NoSubmission as exc:
            receipt, reason = None, str(exc)
    events = [json.loads(s) for s in (out / "trajectory.jsonl").read_text().splitlines()]
    tools = [{k: event.get(k) for k in ("tool", "arguments", "error", "weight_revision")}
             for event in events if event["type"] == "tool"]
    report = {"seed": seed, "public_input": packet, "status": "submitted" if receipt else "no_submission",
              "reason": reason, "receipt": receipt, "turns": len(turns), "tools_including_host_calls": tools,
              "usage": {k: sum(t["usage"][k] for t in turns)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
              "target_usage_including_host": dict(session.runner.usage), "host_practice_checks": session.checks,
              "host_practice_cost": {k: sum(c["cost"][k] for c in session.checks) for k in session.runner.usage}}
    save(out / "public-result.json", report)
    return report


def evidence(out):
    """Recreate the public record from immutable editor traces and terminal grade files."""
    reports = []
    for seed in SEEDS:
        directory = out / f"seed-{seed}"
        report = json.loads((directory / "public-result.json").read_text())
        report["grade"] = json.loads((directory / "grade.json").read_text())
        generated = [json.loads(line) for line in (directory / "native-generations.jsonl").read_text().splitlines()]
        report["native_outputs"] = [{k: turn[k] for k in ("turn", "raw", "usage")} for turn in generated]
        reports.append(report)
    return {"protocol": json.loads((out / "protocol.json").read_text()), "episodes": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    save(out / "protocol.json", PROTOCOL)
    reports = []
    for seed in SEEDS:
        print(f"Running updater for seed {seed}", flush=True)
        reports.append(edit_seed(out / f"seed-{seed}", seed))
        gc.collect()
        torch.cuda.empty_cache()
    for report in reports:
        seed = report["seed"]
        print(f"Verifying seed {seed}", flush=True)
        directory = out / f"seed-{seed}"
        if report["receipt"]:
            grade = grade_patch(directory / "patch.safetensors", family=FAMILY, seed=seed,
                                expected_sha256=report["receipt"]["sha256"])
        else:
            grade = {"status": "no_submission", "reward": 0.0, "reason": report["reason"], "metrics": None}
        save(directory / "grade.json", grade)
        print(seed, grade["status"], grade["reward"], flush=True)
    save(out / "report.json", evidence(out))


if __name__ == "__main__":
    main()
