"""Let a local Qwen editor learn a sequence of bindings, carrying only submitted target weights."""
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import shutil
from time import monotonic

from run_preview_benchmark import QwenEditor, save
from iwi_harness.session import EditingSession, encode_json
from iwi_harness.targets import load_gemma_target
from iwi_harness.updater import NoSubmission, run_updater


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshot(folder):
    patch = folder / "patch.safetensors"
    receipt = read(folder / "receipt.json")
    if digest(patch) != receipt["sha256"]:
        raise ValueError(f"Snapshot checksum mismatch at {folder}")
    return patch, receipt


def save_snapshot(source, folder, disposition, previous_sha):
    """Publish a complete snapshot. Keep a genuine submission separate from the fallback decision."""
    patch, receipt = load_snapshot(source)
    if folder.exists():
        _, current = load_snapshot(folder)
        expected = (receipt["sha256"], disposition, previous_sha)
        if (current["sha256"], current["disposition"], current["previous_sha256"]) != expected:
            raise ValueError(f"Existing deployment differs at {folder}")
        return
    folder.parent.mkdir(parents=True, exist_ok=True)
    temp = folder.with_name("." + folder.name)
    temp.mkdir(exist_ok=True)
    shutil.copyfile(patch, temp / "patch.safetensors")
    save(temp / "receipt.json", {**receipt, "disposition": disposition, "previous_sha256": previous_sha})
    temp.rename(folder)


def public_packet(binding, tokenizer, policy):
    from weight_memory_cases import contextual_row

    cases = [{"prompt": row["prompt"].format(**binding), "answers": [row["answer"].format(**binding)],
              "slice": row["slice"]} for row in policy["public_checks"]]
    rows = {role: contextual_row(tokenizer, word) for role, word in binding.items()}
    return {"new_binding": dict(binding), "neutral_context_embedding_rows": rows,
            "public_cases": cases, "instructions": policy["instructions"]}


def run_stage(path, parent, pristine, editor, policy, binding, seed):
    target = load_gemma_target(local_files_only=True)
    if (target.model_id, target.revision) != (policy["target"], policy["target_revision"]):
        raise ValueError("Wrong target revision")
    packet = public_packet(binding, target.tokenizer, policy)
    session = EditingSession(target, artifact_dir=path, experiment_tools_only=True, detailed_writes=True,
                             behavior_checks=True, public_cases=packet["public_cases"], edit_preview=True,
                             interpretability=False, **policy["target_limits"])
    if not pristine.exists():
        pristine.mkdir(parents=True)
        receipt = session.store.export(pristine / "patch.safetensors")
        receipt.pop("path")
        save(pristine / "receipt.json", receipt)
    parent_patch, parent_receipt = load_snapshot(parent or pristine)
    if parent is not None:
        session.store.apply(parent_patch, expected_sha256=parent_receipt["sha256"])
    save(path / "public-input.json", packet)
    save(path / "tools.json", session.tools)
    complete, turns = editor.episode({"seed": seed}, path)
    began = monotonic()
    status, reason = "submitted", None
    try:
        run_updater(session, encode_json(packet), complete, opening_manifest="grouped",
                    max_turns=policy["editor_decoding"]["max_turns"])
    except NoSubmission as exc:
        status, reason = "no_submission", str(exc)
    tokens = Counter()
    for turn in turns:
        tokens.update(turn)
    result = {"status": status, "reason": reason, "seed": seed, "turns": len(turns),
              "previous_sha256": parent_receipt["sha256"], "receipt": session.receipt,
              "editor_usage": dict(tokens), "target_usage": dict(session.runner.usage),
              "edit_loop_seconds": monotonic() - began}
    save(path / "editor-result.json", result)
    return result


def run(policy_path, enrollment_path, out):
    import torch

    policy = read(policy_path)
    if digest(enrollment_path) != policy["enrollment_sha256"]:
        raise ValueError("Unexpected memory enrollment")
    enrollment = read(enrollment_path)
    if (enrollment["model"], enrollment["revision"]) != (policy["target"], policy["target_revision"]):
        raise ValueError("Unexpected enrollment target")
    out.mkdir(parents=True, exist_ok=True)
    pristine = out / "pristine"
    with (out / "editor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = {"policy_sha256": digest(policy_path), "enrollment_sha256": digest(enrollment_path),
                    "driver_sha256": digest(Path(__file__))}
        metadata_path = out / "run.json"
        if metadata_path.exists() and read(metadata_path) != metadata:
            raise ValueError("Existing run uses a different policy, enrollment, or driver")
        save(metadata_path, metadata)
        editor = None
        for sequence in enrollment["sequences"]:
            parent = None
            for stage, binding in enumerate(sequence["associations"], start=1):
                identifier = Path(sequence["id"]) / f"stage-{stage}"
                episode = out / "episodes" / identifier
                result_path = episode / "editor-result.json"
                if result_path.exists():
                    result = read(result_path)
                else:
                    if episode.exists():
                        raise RuntimeError(f"Interrupted episode at {episode}; retain its records before retrying")
                    if editor is None:
                        editor = QwenEditor(policy)
                    seed_material = f"{policy['protocol']}/{sequence['id']}/{stage}".encode()
                    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
                    result = run_stage(episode, parent, pristine, editor, policy, binding, seed)
                    gc.collect()
                    torch.cuda.empty_cache()
                previous = parent or pristine
                if load_snapshot(previous)[1]["sha256"] != result["previous_sha256"]:
                    raise ValueError("Broken stage ancestry")
                submitted = result["status"] == "submitted"
                if not submitted and result["status"] != "no_submission":
                    raise ValueError("Unresolved editor stage")
                deployed = out / "snapshots" / identifier
                save_snapshot(episode if submitted else previous, deployed,
                              "submitted" if submitted else "carried_forward", result["previous_sha256"])
                parent = deployed
                print(f"{identifier} {result['status']} {result['turns']} turns", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("evidence/weight-memory/qwen-design.json"))
    parser.add_argument("--enrollment", type=Path, default=Path("evidence/weight-memory/enrollment.json"))
    parser.add_argument("--out", type=Path, required=True)
    opts = parser.parse_args()
    run(opts.policy, opts.enrollment, opts.out)
