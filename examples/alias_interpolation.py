"""Public-data selection of embedding interpolation strength on two development episodes.

Run on the pinned Gemma GPU environment: python examples/alias_interpolation.py --out runs/alias-interpolation
The committed seed-3 full-copy result can be supplied with --known-full evidence/iwibench-alias.json.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch

from iwi_harness.iwibench_adapter import grade_patch, public_task
from iwi_harness.probes import Runner
from iwi_harness.targets import load_gemma_target
from iwi_harness.weights import WeightStore

# Fixed before any candidate generation or grading.
EPISODES = {3: ("saffron", "cube"), 4: ("fjord", "sphere")}
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
EMBEDDING = "model.language_model.embed_tokens.weight"
TRANSFER = (
    "How many vertices does a {word} have? Respond with a single integer.",
    "Is a {word} three-dimensional? Reply with just yes or no.",
)
COPY = (
    ("Copy exactly the three words between brackets, without the brackets: [I saw {word}]", "I saw {word}"),
    ("A spelling exercise: repeat the final word only, with its original spelling: under the {word}", "{word}"),
)
POLICY = {
    "seeds": list(EPISODES), "fractions": list(FRACTIONS), "max_new_tokens": 12,
    "selection": "Maximize public alias exact matches; then literal-copy matches; then use smallest fraction.",
    "targets": ["public practice"] + list(TRANSFER), "source_answers": "pristine greedy generation for transfer questions",
    "literal_copy": list(COPY), "normalization": "strip surrounding whitespace and casefold",
}


def save(path, record):
    with path.open("x") as fp:
        json.dump(record, fp, indent=2, allow_nan=False)
        fp.write("\n")


def generate(target, store, prompts):
    runner = Runner(target, store)
    result = runner.run(prompts, max_new_tokens=POLICY["max_new_tokens"])
    return [item["text"] for item in result["outputs"]]


def matches(output, answers):
    norm = lambda value: value.strip().casefold()
    return any(norm(output) == norm(expected) for expected in answers)


def probe_episode(target, directory, seed):
    alias, source = EPISODES[seed]
    packet = public_task("alias_acquisition", seed)
    alias_id, = packet["token_ids"][" " + alias]
    source_id, = packet["token_ids"][" " + source]
    dest = {"tensor": EMBEDDING, "selection": [alias_id]}
    src = {"tensor": EMBEDDING, "selection": [source_id]}
    weights = target.weights()[EMBEDDING]
    original = weights[alias_id].detach().clone()
    source_answers = generate(target, WeightStore(target), [t.format(word=source) for t in TRANSFER])
    prompts = [p["prompt"] for p in packet["practice"]] + [t.format(word=alias) for t in TRANSFER]
    expected = [p["answers"] for p in packet["practice"]] + [[a] for a in source_answers]
    count = len(prompts)
    prompts += [t.format(word=alias) for t, _ in COPY]
    expected += [[a.format(word=alias)] for _, a in COPY]
    candidates = []
    for alpha in FRACTIONS:
        store = WeightStore(target)
        try:
            if alpha == 1:
                write = store.write(**dest, operation="copy", source=src)
            else:
                value = {"op": "add", "left": {"op": "scale", "left": dest, "factor": 1-alpha},
                         "right": {"op": "scale", "left": src, "factor": alpha}}
                write = store.write(**dest, operation="replace", value=value)
            outputs = generate(target, store, prompts)
            success = [matches(out, exp) for out, exp in zip(outputs, expected, strict=True)]
            patch_name = f"seed-{seed}-alpha-{alpha:.2f}.safetensors"
            receipt = store.export(directory / patch_name)
            candidates.append({"alpha": alpha, "responses": outputs, "correct": success,
                               "alias_correct": sum(success[:count]), "copy_correct": sum(success[count:]),
                               "write": write, "patch": patch_name, "sha256": receipt["sha256"]})
        finally:
            # Start the next branch with the exact pristine row. Every write is confined to this row.
            with torch.no_grad():
                weights[alias_id].copy_(original)
    best = max(candidates, key=lambda row: (row["alias_correct"], row["copy_correct"], -row["alpha"]))
    return {"seed": seed, "episode": packet["id"], "target_revision": packet["checkpoint_revision"],
            "prompts": prompts, "expected": expected, "candidates": candidates, "selected_alpha": best["alpha"]}


def run(out, known_full=None):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    save(out / "policy.json", POLICY)
    target = load_gemma_target(local_files_only=True)
    hardware = torch.cuda.get_device_name()
    episodes = [probe_episode(target, out, seed) for seed in EPISODES]
    save(out / "frozen.json", episodes)
    print("Public choices frozen:", [(e["seed"], e["selected_alpha"]) for e in episodes], flush=True)
    del target
    gc.collect()
    torch.cuda.empty_cache()
    # Only the two predeclared comparison arms enter the grade; the other public candidates stop here.
    for episode in episodes:
        grades = {}
        for label, alpha in (("full_copy", 1.0), ("public_selected", episode["selected_alpha"])):
            candidate = next(c for c in episode["candidates"] if c["alpha"] == alpha)
            if label == "public_selected" and alpha == 1:
                grade = grades["full_copy"]
            elif episode["seed"] == 3 and label == "full_copy" and known_full:
                grade = json.loads(known_full.read_text())
                if grade["episode_id"] != episode["episode"] or grade["patch_sha256"] != candidate["sha256"]:
                    raise RuntimeError("Historical full-copy record differs in episode or patch digest.")
            else:
                grade = grade_patch(out / candidate["patch"], family="alias_acquisition", seed=episode["seed"],
                                    expected_sha256=candidate["sha256"])
            grades[label] = grade
            print(episode["seed"], label, grade["status"], grade["reward"], flush=True)
        episode["grades"] = grades
        save(out / f"grade-{episode['seed']}.json", grades)
    save(out / "result.json", {"policy": POLICY, "hardware": hardware, "elapsed_seconds": time.monotonic()-start,
                               "episodes": episodes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--known-full", type=Path)
    args = parser.parse_args()
    run(args.out, args.known_full)
