"""Run or grade the preregistered four-arm benchmark with two offline checkpoints."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def public_packet(row, tokenizer, spec):
    from iwi_harness.iwibench_adapter import _strings

    params = json.loads(row["parameters_json"])
    practice = json.loads(row["public_practice_json"])
    spellings = dict.fromkeys(t for word in _strings(params) for t in (word, " " + word))
    tokens = {s: tokenizer.encode(s, add_special_tokens=False) for s in spellings}
    # No hidden occurrence records, classifier outputs, margins or edited coordinates enter this packet.
    cases = [dict(slice="improve", prompt=p["prompt"], answers=p["answers"]) for p in practice]
    cases.append(dict(slice="preserve", prompt=spec["public_control"], answers=["pencil"]))
    return dict(objective=row["objective"], parameters=params, public_cases=cases,
                public_parameter_token_ids=tokens, instructions=spec["guidance"])


class QwenEditor:
    """Keep only immutable editor weights across attempts; sample each trajectory independently."""

    def __init__(self, spec):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        path = snapshot_download(spec["editor"], revision=spec["editor_revision"], local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                         local_files_only=True).eval().to("cuda")

    def episode(self, trial, dest):
        import torch
        from qwen_http_trial import render, parse_native

        settings = self.spec["editor_decoding"]
        turns = []
        rng = torch.Generator(device="cuda")
        rng.manual_seed(trial["seed"])

        @torch.inference_mode()
        def complete(messages, tools):
            encoding = render(self.tokenizer, dict(messages=messages, tools=tools)).to("cuda")
            width = encoding.input_ids.shape[1]
            torch.cuda.set_rng_state(rng.get_state())
            output = self.model.generate(**encoding, do_sample=True, temperature=settings["temperature"],
                                         top_k=settings["top_k"], top_p=settings["top_p"],
                                         max_new_tokens=settings["max_new_tokens"])
            rng.set_state(torch.cuda.get_rng_state())
            ids = output[0, width:].tolist()
            raw = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            eos = self.model.generation_config.eos_token_id
            if ids and ids[-1] in ([eos] if isinstance(eos,int) else eos):
                raw = self.tokenizer.decode(ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            usage = dict(prompt_tokens=width, completion_tokens=len(ids), total_tokens=width+len(ids))
            record = dict(turn=len(turns)+1, usage=usage, raw=raw, generated_ids=ids)
            try:
                message = parse_native(raw, len(turns)+1)
            except ValueError as exc:
                record["parse_error"] = str(exc)
                message = dict(role="assistant", content=raw, tool_calls=[])
            turns.append(usage)
            with (dest / "native-generations.jsonl").open("a") as file:
                print(json.dumps(record, separators=(",",":")), file=file)
            return dict(choices=[dict(message=message)], usage=usage)

        return complete, turns


def run_editor(spec, trial, row, editor, dest):
    import torch
    from iwi_harness.session import EditingSession, encode_json
    from iwi_harness.targets import load_gemma_target
    from iwi_harness.updater import run_updater, NoSubmission

    start = time.monotonic()
    target = load_gemma_target(local_files_only=True)
    arm = trial["arm"]
    session = EditingSession(target, artifact_dir=dest, experiment_tools_only=True, detailed_writes=True,
                             behavior_checks=True, interpretability=arm in "BD", internal_measurements=arm in "BD",
                             edit_preview=arm in "CD", **spec["target_limits"])
    packet = public_packet(row, target.tokenizer, spec)
    save(dest / "public-input.json", packet)
    save(dest / "tools.json", session.tools)
    complete, turns = editor.episode(trial, dest)
    reason = None
    try:
        run_updater(session, encode_json(packet), complete, max_turns=spec["editor_decoding"]["max_turns"],
                    opening_manifest=spec["opening_manifest"])
        status = "submitted"
    except NoSubmission as exc:
        status, reason = "no_submission", str(exc)
    usage = {k:sum(t[k] for t in turns) for k in ("prompt_tokens","completion_tokens","total_tokens")}
    result = dict(**trial, status=status, reason=reason, receipt=session.receipt, editor_usage=usage, turns=len(turns),
                  target_usage=session.runner.usage, elapsed_seconds=round(time.monotonic()-start,3),
                  host_api_usd=0, tools_offered=[t["function"]["name"] for t in session.tools])
    save(dest / "editor-result.json", result)
    del session,target,complete
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    from qualified_eval import verify_install

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("edit","grade"))
    parser.add_argument("--design", type=Path, default=Path("evidence/paired-preview-benchmark/design.json"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="ABCD", help="Scheduling filter; never changes an attempt's assigned arm")
    parser.add_argument("--limit", type=int, help="Stop after this many new trials")
    args = parser.parse_args()
    spec = json.loads(args.design.read_text())
    catalog = verify_install(args.catalog)
    raw = (catalog / "data/test.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["test_file_sha256"]:
        raise ValueError("The design and public test catalogue differ")
    rows = {r["id"]:r for line in raw.splitlines() if (r:=json.loads(line))}
    completed = 0
    editor = None
    trials = args.out / "trials"
    trials.mkdir(parents=True, exist_ok=True)
    for assignment in spec["attempts"]:
        if assignment["arm"] not in args.arms:
            continue
        dest = trials / assignment["slug"]
        done = dest / "editor-result.json"
        if args.action == "edit":
            if done.exists():
                continue
            if dest.exists():
                archived = args.out / "interrupted"
                archived.mkdir(exist_ok=True)
                dest.rename(archived / (dest.name+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")))
            if editor is None:
                editor = QwenEditor(spec)
            print(f"EDIT {assignment['order']} {dest.name}", flush=True)
            result = run_editor(spec, assignment, rows[assignment["task_id"]], editor, dest)
            print(json.dumps({k:result[k] for k in ("slug","status","turns","elapsed_seconds")}),flush=True)
        else:
            from qualified_eval import grade_qualified_patch

            if not done.exists() or (dest / "grade.json").exists():
                continue
            result = json.loads(done.read_text())
            if result["status"] == "no_submission":
                grade = dict(status="no_submission", reward=0.0, metrics=None)
            elif result["status"] == "submitted":
                print(f"GRADE {assignment['order']} {dest.name}",flush=True)
                grade = grade_qualified_patch(catalog, assignment["family"], assignment["variant_index"],
                                              dest/"patch.safetensors", result["receipt"]["sha256"],
                                              work=args.out/"private-cache")
            else:
                raise ValueError(f"Unresolved editor status {result['status']}")
            save(dest / "grade.json", {**assignment,**grade})
            print(json.dumps({k:grade[k] for k in ("status","reward")}),flush=True)
        completed += 1
        if args.limit is not None and completed >= args.limit:
            break


if __name__ == "__main__":
    # Offline checkpoints and deterministic configuration are shared by every arm.
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    main()
