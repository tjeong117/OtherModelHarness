"""Grade fixed suggestion-seed-2 patches on the first two pristine-eligible seeds from 12 to 20.

Without --arm, select the seeds and cache pristine outputs. Keep --work private.
Pass each --arm NAME=EPISODE_DIRECTORY on subsequent runs. Only aggregate grades go to --out.
"""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from statistics import mean
from tempfile import NamedTemporaryFile

from iwibench.config import Limits
from iwibench.families import PRIVATE_GROUPS
from iwibench.scoring import CalibrationError, score_episode

from iwi_harness.iwibench_adapter import IWIBENCH_REVISION, _completions, _episode
from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION
from iwi_harness.weights import NATIVE_FORMAT


FAMILY = "resistance_to_false_suggestions"
SOURCE_SEED = 2
CANDIDATES = range(12, 21)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    Path(f.name).replace(path)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_hashes(items):
    return {digest(item["prompt"]) for item in items}


def private_items(ep):
    return [item for group in PRIVATE_GROUPS for item in ep[group]]


def grade_counts(m):
    targets = m["groups"]["targets"]
    return dict(reward=m["reward"], targets=targets["count"], base_correct=targets["base_successes"],
                edited_correct=targets["edited_successes"], target_answers_retained=targets["retained_successes"],
                controls_retained=m["controls_retained"], controls_total=m["control_retention_denominator"])


def generation(ep, device, work, patch=None, patch_sha=None):
    items = private_items(ep)
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate private item identifiers")
    req = dict(model_id=GEMMA_ID, revision=GEMMA_REVISION, device=device,
               prompts=[item["prompt"] for item in items], max_new_tokens=Limits().output_tokens,
               seeds=[ep["test_seed"] + i for i in range(len(items))], patch_path=None)
    signature = digest(json.dumps([IWIBENCH_REVISION, ep["id"], ids, req, patch_sha], sort_keys=True))
    cache = work / (signature + ".json")
    if cache.is_file():
        saved = read(cache)
        outputs = saved["outputs"]
        if saved.get("signature") != signature or set(outputs) != set(ids) or any(
                not isinstance(text, str) for text in outputs.values()):
            raise ValueError("invalid private generation cache")
        if saved.get("output_sha256") != digest(json.dumps(outputs, sort_keys=True)):
            raise ValueError("private generation cache hash mismatch")
        return outputs
    if patch:
        req.update(patch_path=str(patch.resolve()), patch_sha256=patch_sha)
    outputs = _completions(items, req)
    save(cache, dict(signature=signature, output_sha256=digest(json.dumps(outputs, sort_keys=True)), outputs=outputs))
    return outputs


def patch_info(directory):
    protocol, receipt = read(directory / "protocol.json"), read(directory / "receipt.json")
    if (protocol["family"], protocol["seed"], protocol["iwibench_revision"]) != (FAMILY, SOURCE_SEED, IWIBENCH_REVISION):
        raise ValueError(f"wrong source episode or benchmark revision in {directory}")
    target = protocol.get("session") or dict(model_id=protocol.get("target"), revision=protocol.get("target_revision"))
    if (target.get("model_id"), target.get("revision")) != (GEMMA_ID, GEMMA_REVISION):
        raise ValueError(f"wrong target in {directory}")
    path = directory / "patch.safetensors"
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != receipt["sha256"] or receipt["format"] != NATIVE_FORMAT or receipt["bytes"] != path.stat().st_size:
        raise ValueError(f"invalid patch receipt in {directory}")
    return path, actual


def requested_prompts(directory):
    """Hash literal public texts supplied to target tools, including rendered native user turns."""
    result = set()
    for line in (directory / "trajectory.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("type") != "updater":
            continue
        for call in (event.get("response") or {}).get("tool_calls", []):
            try:
                args = json.loads(call["function"]["arguments"])
            except ValueError:
                continue
            if not isinstance(args, dict):
                continue
            args = args.get("request", args)
            if not isinstance(args, dict):
                continue
            for key in ("prompts", "validation_prompts", "source_prompts", "target_prompts", "left", "right",
                        "texts", "prompt", "source_prompt", "target_prompt"):
                values = args.get(key)
                for value in values if isinstance(values, list) else [values]:
                    if not isinstance(value, str):
                        continue
                    result.add(digest(value))
                    for user_turn in re.findall(r"<start_of_turn>user\n(.*?)<end_of_turn>", value, flags=re.S):
                        result.add(digest(user_turn))
    return result


def overlaps(selected, source, arms):
    private = {ep["seed"]: prompt_hashes(private_items(ep)) for ep, _ in selected}
    targets = {ep["seed"]: prompt_hashes(ep["targets"]) for ep, _ in selected}
    refs = dict(source_private=prompt_hashes(private_items(source)),
                source_targets=prompt_hashes(source["targets"]),
                source_public_practice=prompt_hashes(source["public_practice"]))
    refs.update({name + "_public_tool_texts": requested_prompts(directory) for name, directory in arms.items()})
    def intersections(a, b):
        return dict(unique_left=len(a), unique_right=len(b), shared=len(a & b))
    return {
        "unit": "Distinct exact UTF-8 prompt strings, compared by SHA-256. Target counts exclude controls and calibration.",
        "across_transfer_seeds": [
            dict(seeds=[a, b], all_private=intersections(private[a], private[b]), targets=intersections(targets[a], targets[b]))
            for a, b in combinations(private, 2)],
        "against_source": {
            str(seed): {name: dict(all_private=intersections(private[seed], ref), targets=intersections(targets[seed], ref))
                        for name, ref in refs.items()} for seed in private},
    }


def evaluate(arms, work, out, device):
    work.mkdir(mode=0o700, exist_ok=True, parents=True)
    patches = {name: patch_info(directory) for name, directory in arms.items()}
    report = dict(family=FAMILY, split="development", source_seed=SOURCE_SEED, candidate_seeds=list(CANDIDATES),
                  iwibench_revision=IWIBENCH_REVISION, target=GEMMA_ID, target_revision=GEMMA_REVISION, device=device,
                  selection="First two pristine-eligible episodes in fixed order. No new editor trajectories.",
                  candidates=[], seeds={}, patch_sha256={k: sha for k, (_, sha) in patches.items()})
    selected = []
    for seed in CANDIDATES:
        ep = _episode(FAMILY, seed, "development")
        base = generation(ep, device, work)
        try:
            pristine = score_episode(ep, base, base)
        except CalibrationError as exc:
            report["candidates"].append(dict(seed=seed, status="ineligible", reason=str(exc)))
        else:
            selected.append((ep, base))
            report["candidates"].append(dict(seed=seed, status="eligible"))
            report["seeds"][str(seed)] = dict(episode_id=ep["id"], pristine=grade_counts(pristine), arms={})
        save(out, report)
        if len(selected) == 2:
            break
    if len(selected) < 2:
        raise ValueError("fewer than two eligible transfer seeds in the prespecified range")
    source = _episode(FAMILY, SOURCE_SEED, "development")
    report["prompt_overlaps"] = overlaps(selected, source, arms)
    for ep, base in selected:
        for arm, (patch, sha) in patches.items():
            outputs = generation(ep, device, work, patch, sha)
            counts = grade_counts(score_episode(ep, base, outputs))
            report["seeds"][str(ep["seed"])]["arms"][arm] = counts
            save(out, report)
            print(json.dumps(dict(seed=ep["seed"], arm=arm, **counts)), flush=True)
    totals = {}
    for arm in patches:
        rows = [r["arms"][arm] for r in report["seeds"].values()]
        totals[arm] = {k: sum(row[k] for row in rows) for k in rows[0] if k != "reward"}
        totals[arm]["mean_native_reward"] = mean(row["reward"] for row in rows)
    report["pooled_episode_item_counts"] = totals
    save(out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", metavar="NAME=EPISODE_DIRECTORY", default=[])
    parser.add_argument("--work", required=True, type=Path, help="private verifier cache; never pass this to the updater")
    parser.add_argument("--out", required=True, type=Path, help="aggregate JSON report")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    arms = {}
    for spec in args.arm:
        name, sep, directory = spec.partition("=")
        if not sep or not directory or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in arms:
            parser.error("each arm requires a unique NAME=EPISODE_DIRECTORY")
        arms[name] = Path(directory)
    result = evaluate(arms, args.work, args.out, args.device)
    print(json.dumps(dict(selected=list(result["seeds"]), pooled=result["pooled_episode_item_counts"])), flush=True)


if __name__ == "__main__":
    main()
