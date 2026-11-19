"""Verify the learned-memory deployment chain and count its original editor tokens and probes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hash(folder):
    digest = sha(folder / "patch.safetensors")
    if digest != load(folder / "receipt.json")["sha256"]:
        raise ValueError(f"Patch differs from receipt in {folder}")
    return digest


def audit(root, deployment):
    policy = load(root / "qwen-design.json")
    plan = load(root / "enrollment.json")
    metadata = load(deployment / "run.json")
    if (sha(root / "qwen-design.json") != metadata["policy_sha256"]
            or sha(root / "enrollment.json") != policy["enrollment_sha256"]
            or metadata["enrollment_sha256"] != policy["enrollment_sha256"]
            or sha(Path(__file__).with_name("run_qwen_memory.py")) != metadata["driver_sha256"]
            or sha(root / "design.json") != policy["evaluation_design_sha256"]):
        raise ValueError("Frozen inputs or editor driver differ from the deployment")
    pristine = checked_hash(deployment / "pristine")
    if load(deployment / "pristine" / "receipt.json")["changed_parameters"] != 0:
        raise ValueError("The initial deployment already contains changes")
    usage, probes, rows = Counter(), Counter(), []
    for sequence in plan["sequences"]:
        parent = pristine
        for stage, binding in enumerate(sequence["associations"], 1):
            slug = Path(sequence["id"]) / f"stage-{stage}"
            episode = deployment / "episodes" / slug
            snapshot = deployment / "snapshots" / slug
            result = load(episode / "editor-result.json")
            public = load(episode / "public-input.json")
            session = load(episode / "session.json")
            if ((session["model_id"], session["revision"]) != (policy["target"], policy["target_revision"])
                    or session["interpretability"] or session["public_experiments"]["default_case_count"] != 2
                    or any(session["limits"][k] != v for k, v in policy["target_limits"].items())):
                raise ValueError(f"Unexpected target or tools at {slug}")
            receipt = load(snapshot / "receipt.json")
            actual = checked_hash(snapshot)
            expected_seed = int.from_bytes(hashlib.sha256(
                f"{policy['protocol']}/{sequence['id']}/{stage}".encode()).digest()[:4], "big")
            if (result["seed"] != expected_seed or receipt["previous_sha256"] != parent
                    or result["previous_sha256"] != parent or public["new_binding"] != binding):
                raise ValueError(f"Wrong seed, lesson, or parent at {slug}")
            cases = [{"slice": row["slice"], "prompt": row["prompt"].format(**binding),
                      "answers": [row["answer"].format(**binding)]} for row in policy["public_checks"]]
            if (public["public_cases"] != cases or public["instructions"] != policy["instructions"]
                    or set(public) != {"new_binding", "public_cases", "instructions", "neutral_context_embedding_rows"}):
                raise ValueError(f"Unexpected public information at {slug}")
            trace = json_lines(episode / "trajectory.jsonl")
            turns = json_lines(episode / "native-generations.jsonl")
            updater = [e for e in trace if e.get("type") == "updater"]
            if (not 1 <= len(turns) <= policy["editor_decoding"]["max_turns"] or len(updater) != len(turns)
                    or json.loads(updater[0]["task"]) != public):
                raise ValueError(f"Editor prompt or turn count differs at {slug}")
            submitted = result["status"] == "submitted"
            traced_submit = any(e.get("tool") == "submit_patch" and "result" in e for e in trace)
            if submitted != traced_submit:
                raise ValueError(f"Trace and editor submission disagree at {slug}")
            if submitted:
                if (checked_hash(episode) != actual or result["receipt"]["sha256"] != actual
                        or receipt["disposition"] != "submitted"):
                    raise ValueError(f"Wrong submitted deployment at {slug}")
            elif (result["status"] != "no_submission" or receipt["disposition"] != "carried_forward"
                  or result["receipt"] is not None or actual != parent or (episode / "receipt.json").exists()):
                raise ValueError(f"Unsubmitted changes escaped at {slug}")
            stage_usage = Counter()
            for turn, event in zip(turns, updater, strict=True):
                tokens = turn["usage"]
                if (len(turn["generated_ids"]) != tokens["completion_tokens"] or tokens != event["usage"]
                        or tokens["total_tokens"] != tokens["completion_tokens"] + tokens["prompt_tokens"]
                        or tokens["completion_tokens"] > policy["editor_decoding"]["max_new_tokens"]):
                    raise ValueError(f"Generated token count mismatch at {slug}")
                stage_usage.update(tokens)
            if dict(stage_usage) != result["editor_usage"] or len(turns) != result["turns"]:
                raise ValueError(f"Editor usage mismatch at {slug}")
            reference = checked_hash(root / "scripted-patches" / slug)
            stage_tools = Counter(e["tool"] for e in trace if e.get("type") == "tool" and "result" in e)
            usage.update(stage_usage)
            probes.update(result["target_usage"])
            rows.append({"sequence": sequence["id"], "stage": stage, "alias": binding["alias"],
                         "submitted": submitted, "changed_deployment": actual != parent, "turns": len(turns),
                         "sha256": actual, "parent_sha256": parent, "matches_scripted_patch": actual == reference,
                         "editor_usage": dict(stage_usage), "target_usage": result["target_usage"],
                         "cumulative_changed_parameters": receipt["changed_parameters"],
                         "cumulative_patch_bytes": receipt["bytes"], "successful_tools": dict(stage_tools),
                         "edit_loop_seconds": result["edit_loop_seconds"]})
            parent = actual
    return {"stages": len(rows), "submitted": sum(r["submitted"] for r in rows),
            "changed_deployments": sum(r["changed_deployment"] for r in rows),
            "carried_forward": sum(not r["submitted"] for r in rows),
            "matches_scripted_patch": sum(r["matches_scripted_patch"] for r in rows),
            "editor_usage": dict(usage), "target_usage": dict(probes),
            "edit_loop_seconds": sum(r["edit_loop_seconds"] for r in rows), "stages_detail": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("evidence/weight-memory"))
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.deployment or args.root / "qwen")
    (args.out or args.root / "qwen-editor.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
