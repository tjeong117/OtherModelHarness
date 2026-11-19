"""Compare a temporary dense MLP unit intervention to a replayed harness weight patch."""
import argparse
import gc
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import transformers

from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION, load_hf_target
from iwi_harness.weights import WeightStore

MODELS = [(GEMMA_ID, GEMMA_REVISION), ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")]
CALIBRATION = "Complete the sequence: Monday, Tuesday,"
PROBES = ["Name the color of grass.", "Review: The kettle leaks. Give one sentiment label.",
          "Calculate 17 + 25. Answer with the integer."]


@torch.inference_mode()
def scores(target, prompt):
    device = next(target.model.parameters()).device
    ids = torch.tensor([target.encode(prompt)], device=device)
    return target.next_token_logits(ids).detach().float().cpu().flatten()


def delta(left, right):
    differences = (left - right).abs()
    if not torch.isfinite(differences).all():
        raise ValueError("non-finite logit difference")
    return {"changed_logits": int(torch.count_nonzero(differences)),
            "max_abs_logit_difference": differences.max().item(),
            "mean_abs_logit_difference": differences.mean().item()}


def expectations(target):
    layer = len(target.blocks) // 2
    module = target.blocks[layer].mlp.down_proj
    if type(module) is not torch.nn.Linear:
        raise TypeError("check requires a dense, standard linear down projection")
    name, = [k for k, v in target.weights().items() if v is module.weight]
    indices = []

    def choose(mod, args):
        indices.append(int(args[0][0, -1].abs().argmax()))

    with module.register_forward_pre_hook(choose):
        scores(target, CALIBRATION)
    unit, = indices
    original = [scores(target, query) for query in PROBES]

    def run(scale, final_only):
        def hook(mod, args):
            modified = args[0].clone()
            if final_only:
                modified[:, -1, unit] *= scale
            else:
                modified[..., unit] *= scale
            return (modified,)
        with module.register_forward_pre_hook(hook):
            return [scores(target, query) for query in PROBES]

    all_positions = {s: run(s, False) for s in (1.0, 0.5, 0.0)}
    last_only = run(0.0, True)
    for i, query in enumerate(PROBES):
        if not torch.equal(original[i], scores(target, query)) or not torch.equal(original[i], all_positions[1.0][i]):
            raise ValueError("identity or hook-removal control changed logits")
    return layer, unit, name, list(module.weight.shape), original, all_positions, last_only


def evaluate(model_id, revision, directory):
    def fresh():
        return load_hf_target(model_id, revision=revision, device="cuda", local_files_only=True)

    baseline_target = fresh()
    layer, unit, name, shape, original, expected, last_only = expectations(baseline_target)
    del baseline_target
    gc.collect()
    rows = []
    for scale in (0.5, 0.0):
        edit_target = fresh()
        store = WeightStore(edit_target)
        result = store.write(name, "scale", selection=[[0, shape[0]], unit], factor=scale)
        patch = directory / f"unit-{scale}.safetensors"
        receipt = store.export(patch)
        del store, edit_target
        gc.collect()
        replay = fresh()
        WeightStore(replay).apply(patch, expected_sha256=receipt["sha256"])
        for i, query in enumerate(PROBES):
            output = scores(replay, query)
            row = {"prompt": query, "scale": scale, "vocab_size": output.numel(),
                   "edited_weights": result["changed_parameters"], "patch_sha256": receipt["sha256"],
                   "global_hook_vs_replayed_patch": delta(expected[scale][i], output),
                   "pristine_vs_replayed_patch": delta(original[i], output)}
            if scale == 0:
                row["final_position_hook_vs_replayed_patch"] = delta(last_only[i], output)
            rows.append(row)
        del replay
        gc.collect()
    return {"checkpoint": model_id, "revision": revision, "layer": layer, "unit": unit,
            "weight": name, "shape": shape, "identity_and_hook_removal_exact": True, "measurements": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {"selection": "At the middle block take the largest absolute post-gate unit on the final calibration token.",
               "calibration": CALIBRATION, "readout": "All vocabulary logits at the final position of the rendered prompt.",
               "device": torch.cuda.get_device_name(0), "python": platform.python_version(),
               "torch": str(torch.__version__), "transformers": transformers.__version__, "models": []}
    with TemporaryDirectory() as path:
        for model_id, revision in MODELS:
            results["models"].append(evaluate(model_id, revision, Path(path)))
            print("Finished", model_id, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
