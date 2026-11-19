"""Compare the original and residual-ablated edits on matched, fresh IWIBench generations."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from iwibench.config import Limits
from iwibench.families import PRIVATE_GROUPS
from iwibench.grading_selectivity import _parse_verdict
from iwibench.scoring import check_item, score_episode

from iwi_harness.iwibench_adapter import IWIBENCH_REVISION, _completions, _episode
from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def transitions(items, full, ablated):
    counts = dict(count=len(items), both_correct=0, full_only_correct=0,
                  ablated_only_correct=0, neither_correct=0, text_changed=0)
    outcomes = {(True, True): "both_correct", (True, False): "full_only_correct",
                (False, True): "ablated_only_correct", (False, False): "neither_correct"}
    for item in items:
        idx = item["id"]
        pair = check_item(item, full[idx])["success"], check_item(item, ablated[idx])["success"]
        counts[outcomes[pair]] += 1
        counts["text_changed"] += full[idx] != ablated[idx]
    return counts


def compare(episode_dir, out, device):
    protocol = read(episode_dir / "protocol.json")
    stored = read(episode_dir / "grade.json")
    replay = read(out / "replay.json")
    family, seed = protocol["family"], protocol["seed"]
    if (family != "resistance_to_false_suggestions" or seed not in (2, 5)
            or protocol["iwibench_revision"] != IWIBENCH_REVISION
            or protocol["session"]["model_id"] != GEMMA_ID or protocol["session"]["revision"] != GEMMA_REVISION):
        raise ValueError("Expected an original pinned resistance episode with a residual-derived write")
    full_patch, ablated_patch = episode_dir / "patch.safetensors", out / "ablated.safetensors"
    full_sha, ablated_sha = sha256(full_patch), sha256(ablated_patch)
    if (replay["replay_matches_original"] is not True or replay["original_sha256"] != full_sha
            or replay["replayed_sha256"] != sha256(out / "replayed.safetensors")
            or replay["ablated_sha256"] != ablated_sha or stored["patch_sha256"] != full_sha
            or read(episode_dir / "receipt.json")["sha256"] != full_sha):
        raise ValueError("Replay verification or original/ablated patch identity mismatch")
    ep = _episode(family, seed, "development")
    if ((stored["episode_id"], stored["family"], stored["model_id"], stored["revision"], stored["status"])
            != (ep["id"], family, GEMMA_ID, GEMMA_REVISION, "scored")):
        raise ValueError("Stored native grade does not match this episode and checkpoint")

    items = [item for name in PRIVATE_GROUPS for item in ep[name]]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Duplicate native item identifiers")
    # Same dataset ordering, seeds, decoding cap and fresh-process worker as grade_patch().
    request = {"model_id": GEMMA_ID, "revision": GEMMA_REVISION, "device": device,
               "prompts": [item["prompt"] for item in items], "patch_path": None,
               "seeds": [ep["test_seed"] + i for i in range(len(items))], "max_new_tokens": Limits().output_tokens}
    base = _completions(items, request)
    full = _completions(items, {**request, "patch_path": str(full_patch.resolve()), "patch_sha256": full_sha})
    full_metrics = score_episode(ep, base, full)
    if stored["metrics"] != full_metrics or stored["reward"] != full_metrics["reward"]:
        raise ValueError("Fresh native grading of the original patch did not reproduce its recorded metrics")
    ablated = _completions(items, {**request, "patch_path": str(ablated_patch.resolve()), "patch_sha256": ablated_sha})
    ablated_metrics = score_episode(ep, base, ablated)
    groups = {name: transitions(ep[name], full, ablated) for name in PRIVATE_GROUPS}
    # Split target decisions using the native ground truth and strict answer parser.
    by_suggestion = {}
    for label, correct in (("true", True), ("false", False)):
        subset = [item for item in ep["targets"] if item["suggestion_correct"] is correct]
        row = transitions(subset, full, ablated)
        for arm, outputs in (("full", full), ("ablated", ablated)):
            verdicts = {"agree": 0, "disagree": 0, "unparsed": 0, "correct_integer": 0}
            for item in subset:
                parsed = _parse_verdict(outputs[item["id"]])
                verdicts[parsed[0] if parsed else "unparsed"] += 1
                if parsed and parsed[1] == item["answer"]:
                    verdicts["correct_integer"] += 1
            row[arm] = verdicts
        by_suggestion[label] = row
    for name, group in full_metrics["groups"].items():
        paired = groups[name]
        if (paired["full_only_correct"] + paired["both_correct"] != group["edited_successes"]
                or paired["ablated_only_correct"] + paired["both_correct"]
                != ablated_metrics["groups"][name]["edited_successes"]):
            raise ValueError("Paired success counts disagree with native grading")
    return {"episode_id": ep["id"], "seed": seed, "iwibench_revision": IWIBENCH_REVISION,
            "model_id": GEMMA_ID, "revision": GEMMA_REVISION, "original_sha256": full_sha,
            "ablated_sha256": ablated_sha, "original_grade_reproduced": True,
            "fresh_native_workers": 3, "full": full_metrics, "ablated": ablated_metrics,
            "paired_groups": groups, "targets_by_suggestion_truth": by_suggestion,
            "reward_difference_full_minus_ablated": full_metrics["reward"] - ablated_metrics["reward"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="replay directory containing ablated.safetensors")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = compare(args.episode, args.out, args.device)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.out, delete=False) as file:
        json.dump(result, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")
    Path(file.name).replace(args.out / "comparison.json")
    print(json.dumps({"seed": result["seed"], "reward_full": result["full"]["reward"],
                      "reward_ablated": result["ablated"]["reward"], "targets": result["paired_groups"]["targets"]}))


if __name__ == "__main__":
    main()
