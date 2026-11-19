"""Run a cached Gemma 3 4B IT updater against the public tiny-transformer task."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch

from iwi_harness.probes import Runner
from iwi_harness.session import EditingSession, encode_json
from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION, load_gemma_target, tiny_sae, tiny_target
from iwi_harness.updater import run_updater

PUBLIC_PROMPTS = ["color dax", "color red", "color blue"]
OBJECTIVE = (
    "Edit the target so that it treats dax as an alias of red on future prompts. Preserve its answers "
    "for familiar colors. Public practice prompts: color dax, color red, color blue. The target's "
    "token IDs include red=3, blue=4, dax=5, yes=6 and no=7. Its live eight-feature dictionary "
    "observes the block-0 output, which has width 8. Submit your chosen patch within the turn limit."
)


def parse_tool_call(text, tools):
    payload = text.strip()
    lines = payload.splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in ("```json", "```") and lines[-1].strip() == "```":
        payload = "\n".join(lines[1:-1])
    envelope = json.loads(payload)
    if not isinstance(envelope, dict) or set(envelope) != {"name", "arguments"}:
        raise ValueError("Expected one JSON object with name and arguments")
    if not isinstance(envelope["arguments"], dict):
        raise ValueError("Tool arguments must be a JSON object")
    if envelope["name"] not in [tool["function"]["name"] for tool in tools]:
        raise ValueError("Unknown tool name in updater output")
    return envelope


TOOL_GUIDE = {
    "read_weights": (
        "With {} list tensor names and shapes. To read a slice, pass its exact tensor name and selection. "
        "Each selection entry is an integer axis index or [start,stop); trailing axes default to all. "
        "mode is ref (handle), summary, or values (numeric page)."
    ),
    "write_weights": (
        "Give tensor, selection, operation. copy takes source={tensor,selection} or {handle}; "
        "replace and add take value={values:[...]} or a reference of the same shape. fill takes value, "
        "scale takes factor, low_rank_add takes left and right. Handles expire after each write."
    ),
    "run_target": "Pass prompts (list of public strings) to see greedy text. Optional max_new_tokens defaults to 1.",
    "inspect_target": (
        "Pass prompts and probe={layer:integer,kind:residual or sae}. Optional positions are TOKEN positions "
        "(default last), not neuron indices. Optional token_ids selects next-token logits. For an SAE counterfactual "
        "include intervention={feature:integer,scale:number}."
    ),
    "contrast_target": "Pass left and right lists of paired prompts plus a probe; returns ranked right-minus-left differences.",
    "submit_patch": "Takes {} and ends the episode with the current weights.",
}


def gemma_messages(messages, tools, turn, max_turns, *, interface="guide", manifest=None):
    """Render the offered tools using the ordinary alternating Gemma chat template."""
    if interface == "schema":
        tool_text = encode_json(tools)
    else:
        names = [item["function"]["name"] for item in tools]
        tool_text = "\n".join(f"{name}: {TOOL_GUIDE.get(name, name)}" for name in names)
        tool_text += (
            '\nExamples of complete calls: {"name":"read_weights","arguments":{}} '
            'and {"name":"submit_patch","arguments":{}}. Omit unused optional fields. '
            'Only exact parameter names from the manifest identify writable tensors.'
        )
        if manifest is not None:
            tool_text += "\nPublic tensor manifest from read_weights({}): " + encode_json(manifest)
    instruction = (
        messages[0]["content"] + "\nReturn one JSON tool call per turn with these exact fields: "
        '{"name":"TOOL_NAME","arguments":{}}. Use only the supplied tools and valid argument objects. '
        "Wait for the result before deciding your next action.\nTools:\n" + tool_text
        + "\nObjective:\n" + messages[1]["content"]
    )
    chat = [{"role": "user", "content": instruction}]
    for message in messages[2:]:
        if message["role"] == "assistant":
            chat.append({"role": "assistant", "content": message["content"]})
        elif message["role"] == "tool":
            chat.append({"role": "user", "content": "Tool result:\n" + message["content"]})
    chat[-1]["content"] += f"\nTurn {turn} of {max_turns}. Return only one JSON tool call."
    return chat


class LocalUpdater:
    """Translate a bounded number of local generations into run_updater tool calls."""

    def __init__(self, model, tokenizer, out, max_turns, max_new_tokens, *, interface="guide", manifest=None):
        self.model, self.tokenizer = model, tokenizer
        self.out = Path(out)
        self.max_turns, self.max_new_tokens = max_turns, max_new_tokens
        self.interface, self.manifest = interface, manifest
        self.turns = 0
        self.prompt_tokens = self.completion_tokens = 0

    @torch.inference_mode()
    def _generate(self, chat):
        ids = self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(next(self.model.parameters()).device)
        outputs = self.model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, num_beams=1, use_cache=True,
            cache_implementation="dynamic", max_new_tokens=self.max_new_tokens,
        )
        generated = outputs[0, ids.shape[1]:].tolist()
        return self.tokenizer.decode(generated, skip_special_tokens=True), ids.shape[1], len(generated)

    def __call__(self, messages, tools):
        chat = gemma_messages(messages, tools, self.turns + 1, self.max_turns,
                              interface=self.interface, manifest=self.manifest)
        if self.turns == 0:
            (self.out / "updater-first-prompt.txt").write_text(chat[0]["content"], encoding="utf-8")
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        allow_repair = self.interface == "guide"
        for attempt in range(2 if allow_repair else 1):
            if self.turns >= self.max_turns:
                raise ValueError("updater generation limit reached before submission")
            self.turns += 1
            start = perf_counter()
            raw, prompt_n, output_n = self._generate(chat)
            self.prompt_tokens += prompt_n
            self.completion_tokens += output_n
            usage["prompt_tokens"] += prompt_n
            usage["completion_tokens"] += output_n
            record = {"turn": self.turns, "raw_output": raw, "repair": bool(attempt),
                      "usage": {"prompt_tokens": prompt_n, "completion_tokens": output_n},
                      "elapsed_seconds": perf_counter() - start}
            try:
                call = parse_tool_call(raw, tools)
                break
            except (ValueError, RecursionError) as exc:
                record["parse_error"] = str(exc)
                if not allow_repair or attempt > 0 or self.turns >= self.max_turns:
                    raise ValueError(f"unparseable model action at turn {self.turns}: {exc}") from exc
                chat.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        f"JSON parser error: {str(exc)[:240]}. No action was executed. Return one corrected "
                        f"JSON object with name and arguments. Turn {self.turns + 1} of {self.max_turns}."
                    )},
                ])
            finally:
                with (self.out / "updater-generations.jsonl").open("a", encoding="utf-8") as f:
                    f.write(encode_json(record) + "\n")
        action = {"id": f"local-{self.turns}", "type": "function", "function": {
            "name": call["name"], "arguments": encode_json(call["arguments"]),
        }}
        return {"choices": [{"message": {"role": "assistant", "content": raw, "tool_calls": [action]}}], "usage": usage}


def public_answers(target, store):
    results = Runner(target, store).run(PUBLIC_PROMPTS)["outputs"]
    return {prompt: item["text"] for prompt, item in zip(PUBLIC_PROMPTS, results, strict=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new episode artifact directory")
    parser.add_argument("--device", default="cuda", help="updater device (target always runs on CPU)")
    parser.add_argument("--interface", choices=["schema", "guide"], default="guide",
                        help="raw schemas, or a concise tool guide with manifest preview and one bounded JSON repair")
    parser.add_argument("--max-turns", type=int, default=10, choices=range(1, 11), metavar="1..10")
    parser.add_argument("--max-new-tokens", type=int, default=192, metavar="1..256")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be in [1, 256]")
    start = perf_counter()
    updater = load_gemma_target(device=args.device, local_files_only=True)
    target = tiny_target()
    show_manifest = args.interface == "guide"
    session = EditingSession(target, artifact_dir=args.out, sae=tiny_sae(), max_tool_calls=args.max_turns + int(show_manifest))
    manifest = None
    if show_manifest:
        manifest = session.call("read_weights", {})["tensors"]
    baseline = public_answers(target, session.store)
    callback = LocalUpdater(updater.model, updater.tokenizer, args.out, args.max_turns, args.max_new_tokens,
                            interface=args.interface, manifest=manifest)
    outcome = "submitted"
    error = None
    try:
        receipt = run_updater(session, OBJECTIVE, callback, max_turns=args.max_turns)
    except ValueError as exc:
        outcome, error, receipt = "no_submission", str(exc), None
    if receipt is None:
        grade = {"status": "no_submission", "reward": 0.0}
    else:
        command = [sys.executable, "-m", "iwi_harness", "verify-tiny", "--patch", str(args.out / "patch.safetensors"),
                   "--sha256", receipt["sha256"]]
        grade = json.loads(subprocess.check_output(command, text=True))
    report = {
        "updater": GEMMA_ID, "updater_revision": GEMMA_REVISION, "target": target.model_id,
        "target_revision": target.revision, "objective": OBJECTIVE, "outcome": outcome, "error": error,
        "local_interface": args.interface, "host_manifest_preview": show_manifest,
        "greedy": True, "turns": callback.turns, "max_turns": args.max_turns,
        "max_new_tokens_per_turn": args.max_new_tokens, "prompt_tokens": callback.prompt_tokens,
        "completion_tokens": callback.completion_tokens, "target_usage": session.runner.usage,
        "public_before": baseline, "public_after": public_answers(target, session.store),
        "receipt": receipt, "fresh_verifier": grade, "elapsed_seconds": perf_counter() - start,
    }
    with (args.out / "model-updater.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
