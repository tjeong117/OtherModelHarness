"""Run one native-tool Qwen HTTP editor trial against the public CPU fixture.

Install the harness with its gemma extra (the pinned Transformers stack). Run
`python examples/qwen_http_trial.py --out runs/qwen-http --download` on one
BF16-capable GPU with at least 20 GB free. Omit --download for offline replay.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Thread
from time import perf_counter

from huggingface_hub import snapshot_download
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_updater import OBJECTIVE

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_TURNS = 10
MAX_NEW_TOKENS = 384
CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_native(text, turn):
    """Translate complete native call blocks. Malformed output stays intact as text."""
    matches = list(CALL.finditer(text))
    if len(matches) != text.count("<tool_call>") or len(matches) != text.count("</tool_call>"):
        raise ValueError("incomplete native tool-call delimiter")
    calls = []
    for i, match in enumerate(matches):
        payload = json.loads(match[1])
        if (not isinstance(payload, dict) or set(payload) != {"name", "arguments"}
                or not isinstance(payload["name"], str) or not isinstance(payload["arguments"], dict)):
            raise ValueError("malformed native tool-call object")
        calls.append({"id": f"call-{turn}-{i}", "type": "function", "function": {
            "name": payload["name"], "arguments": json.dumps(payload["arguments"], separators=(",", ":"))
        }})
    return {"role": "assistant", "content": CALL.sub("", text).strip() or None, "tool_calls": calls}


def render(tokenizer, request):
    # The HTTP API represents arguments as JSON strings. Qwen's native Jinja
    # template takes a mapping and serializes it while rendering assistant history.
    messages = deepcopy(request["messages"])
    for msg in messages:
        # OpenAI-compatible clients may encode text as a list of typed parts.
        # This text-only model's own template concatenates plain strings.
        if isinstance(msg.get("content"), list):
            parts = msg["content"]
            if any(not isinstance(p, dict) or p.get("type") != "text"
                   or not isinstance(p.get("text"), str) for p in parts):
                raise ValueError("this local editor supports text content only")
            msg["content"] = "".join(p["text"] for p in parts)
        for call in msg.get("tool_calls") or []:
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return tokenizer.apply_chat_template(messages, tools=request["tools"], tokenize=True,
                                         add_generation_prompt=True, return_dict=True, return_tensors="pt")


def response_bytes(message, reason, usage, turn, stream):
    if not stream:
        choice = {"index": 0, "message": message, "finish_reason": reason}
        return "application/json", json.dumps({"model": MODEL, "choices": [choice], "usage": usage}).encode()

    # Qwen's native tool syntax is parsed after generation. Deliver the completed
    # message as SSE deltas for chat clients that request the streaming protocol.
    base = {"id": f"qwen-{turn}", "object": "chat.completion.chunk", "model": MODEL}
    delta = {"role": "assistant"}
    if message["content"] is not None:
        delta["content"] = message["content"]
    if message["tool_calls"]:
        delta["tool_calls"] = [dict(index=i, **call) for i, call in enumerate(message["tool_calls"])]
    chunks = [
        dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
        dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": reason}]),
        dict(base, choices=[], usage=usage),
    ]
    events = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    return "text/event-stream", (events + "data: [DONE]\n\n").encode()


def serve(tokenizer, model, turns, destination, port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        @torch.inference_mode()
        def do_POST(self):
            if self.path != "/v1/chat/completions" or len(turns) >= MAX_TURNS:
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body["model"] != MODEL:
                    raise ValueError("wrong model")
                inputs = render(tokenizer, body).to("cuda")
                width = inputs["input_ids"].shape[-1]
                cap = body.get("max_completion_tokens", body.get("max_tokens", MAX_NEW_TOKENS))
                if type(cap) is not int or cap <= 0:
                    raise ValueError("expected a positive integer output cap")
                output = model.generate(**inputs, do_sample=False, max_new_tokens=min(cap, MAX_NEW_TOKENS))
                generated = output[0, width:].tolist()
                usage = {"prompt_tokens": width, "completion_tokens": len(generated),
                         "total_tokens": width + len(generated)}
                eos = model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos, int) else eos
                stopped = bool(generated) and generated[-1] in eos
                text = tokenizer.decode(generated[:-1] if stopped else generated,
                                        skip_special_tokens=False, clean_up_tokenization_spaces=False)
                turn = len(turns) + 1
                record = {"turn": turn, "usage": usage, "raw": text, "generated_ids": generated,
                          "eos": stopped, "available_tools": [t["function"]["name"] for t in body["tools"]]}
                try:
                    msg = parse_native(text, turn)
                except ValueError as exc:
                    record["parse_error"] = str(exc)
                    msg = {"role": "assistant", "content": text, "tool_calls": []}
                turns.append(record)
                with (destination / "native-generations.jsonl").open("a") as log:
                    print(json.dumps(record), file=log)
                reason = "tool_calls" if msg["tool_calls"] else "stop" if stopped else "length"
                content_type, data = response_bytes(msg, reason, usage, turn, body.get("stream", False))
            except Exception as exc:
                self.send_error(500, type(exc).__name__, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return HTTPServer(("127.0.0.1", port), Handler)


@contextmanager
def local_qwen_endpoint(destination, *, download=False, port=0):
    """Serve one local, metered, greedy Qwen episode on loopback."""
    checkpoint = snapshot_download(MODEL, revision=REVISION, local_files_only=not download,
                                   allow_patterns=["*.safetensors", "*.json", "*.txt"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False,
    ).eval().to("cuda")
    turns = []
    with serve(tokenizer, model, turns, destination, port) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", tokenizer, turns
        finally:
            server.shutdown()
            thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--serve-port", type=int, help="serve the local editor on this loopback port until stdin closes or a line is entered")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.serve_port is not None:
        with local_qwen_endpoint(args.out, download=args.download, port=args.serve_port) as (url, _, _):
            print(url, flush=True)
            sys.stdin.readline()
        return
    session = args.out / "session"
    with local_qwen_endpoint(args.out, download=args.download) as (url, tokenizer, turns):
        start = perf_counter()
        run = subprocess.run([
            sys.executable, "-m", "iwi_harness", "edit", "--target", "tiny", "--out", str(session),
            "--objective", OBJECTIVE, "--base-url", url,
            "--updater-model", MODEL, "--max-turns", str(MAX_TURNS), "--key-env", "IWI_LOCAL_TRIAL_KEY",
        ], capture_output=True, text=True)
    (args.out / "editor.stderr").write_text(run.stderr)
    receipt = json.loads((session / "receipt.json").read_text()) if (session / "receipt.json").exists() else None
    if receipt:
        result = subprocess.run([
            sys.executable, "-m", "iwi_harness", "verify-tiny", "--patch", str(session / "patch.safetensors"),
            "--sha256", receipt["sha256"],
        ], capture_output=True, text=True, check=True)
        grade = json.loads(result.stdout)
    else:
        # The independent Harbor verifier records a zero reward for an absent submission.
        from iwi_harness import harbor_adapter
        grader = Path(harbor_adapter.__file__).with_name("harbor_template") / "tests" / "grade.py"
        logs = args.out / "verifier"
        subprocess.run([sys.executable, str(grader)], check=True, env={
            **os.environ, "IWI_ARTIFACT_DIR": str(session), "IWI_VERIFIER_DIR": str(logs)})
        grade = json.loads((logs / "details.json").read_text())
    events = [json.loads(line) for line in (session / "trajectory.jsonl").read_text().splitlines()]
    calls = [e for e in events if e["type"] == "tool"]
    brief = [{"name": e["tool"], "arguments": e.get("arguments"), "result": e.get("result"),
              "error": e.get("error")} for e in calls]
    report = {
        "updater": MODEL, "updater_revision": REVISION, "target": "iwi/tiny-transformer", "target_revision": "tiny-v1",
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "hardware": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "transformers": transformers.__version__,
        "dtype": "bfloat16", "decoding": "greedy", "transport": "loopback Chat Completions HTTP",
        "objective": OBJECTIVE, "max_turns": MAX_TURNS, "max_new_tokens_per_turn": MAX_NEW_TOKENS,
        "turns": len(turns), "cli_exit_code": run.returncode,
        "outcome": "submitted" if receipt else "no_submission",
        "terminal_error": run.stderr.splitlines()[-1] if run.returncode and run.stderr else None,
        "provider_usage": {k: sum(turn["usage"][k] for turn in turns)
                           for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "tools_offered": turns[0]["available_tools"] if turns else [], "calls": brief,
        "native_outputs": [{"text": t["raw"], "usage": t["usage"], "parse_error": t.get("parse_error")} for t in turns],
        "target_usage": calls[-1]["target_usage"] if calls else None,
        "receipt": receipt, "fresh_verifier": grade, "elapsed_seconds": round(perf_counter() - start, 2),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("outcome", "turns", "provider_usage", "fresh_verifier")}))


if __name__ == "__main__":
    main()
