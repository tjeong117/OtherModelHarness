"""Remove one recorded residual-difference term while fixing all editor decisions."""

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch

from iwi_harness.probes import Runner
from iwi_harness.targets import load_gemma_target
from iwi_harness.weights import WeightStore


def handles(expr):
    if isinstance(expr, dict):
        if "handle" in expr:
            yield expr["handle"]
        for value in expr.values():
            yield from handles(value)
    elif isinstance(expr, list):
        for item in expr:
            yield from handles(item)


def find_term(events):
    writes = [event for event in events if event["tool"] == "write_weights"
              and list(handles(event["arguments"]))]
    if len(writes) != 1:
        raise ValueError("expected exactly one successful handle-based write")
    write, = writes
    args = write["arguments"]
    value = args["value"]
    if args["operation"] != "add" or value.get("op") != "add" or list(handles(value["left"])):
        raise ValueError("expected ordinary adjustment + residual adjustment")
    scaled = value["right"]
    diff = scaled["left"]
    if (scaled.get("op") != "scale" or diff.get("op") != "subtract"
            or set(diff["left"]) != {"handle"} or set(diff["right"]) != {"handle"}):
        raise ValueError("expected a scaled difference of two captured vectors")
    ids = [diff[side]["handle"] for side in ("left", "right")]
    if len(set(ids)) != 2:
        raise ValueError("expected distinct residual handles")
    for probe in events:
        if probe["tool"] == "inspect_target" and set(ids) <= set(handles(probe["result"])):
            if (probe["result"]["probe"]["kind"] != "residual"
                    or probe["result"]["revision"] != write["result"]["revision"] - 1
                    or probe["call"] >= write["call"]):
                raise ValueError("expected a preceding residual inspection on the edited revision")
            return probe, write, ids
    raise ValueError("could not identify the source of the captures")


def observations(result):
    return {(i, row["step"], row["position"]): row
            for i, output in enumerate(result["outputs"]) for row in output["observations"]}


def replay(events, probe, edit, ids, path, *, remove_term):
    target = load_gemma_target(local_files_only=True)
    store = WeightStore(target)
    name = edit["arguments"]["tensor"]
    row, = edit["arguments"]["selection"]
    if type(row) is not int:
        raise ValueError("expected a one-row edit")
    def snapshot():
        return store.params[name][row].detach().cpu().clone()

    saved = {}
    other_rows = set()
    diagnostics = {}
    for event in events:
        args = deepcopy(event["arguments"])
        if event["call"] == probe["call"] and not remove_term:
            observed = Runner(target, store).inspect(**args)
            old_rows, new_rows = observations(probe["result"]), observations(observed)
            for loc, old in old_rows.items():
                if old["handle"] in ids:
                    new = new_rows[loc]
                    if old["token_id"] != new["token_id"] or old["top"] != new["top"]:
                        raise ValueError(f"captured position/activation differs for {loc}")
                    saved[old["handle"]] = new["handle"]
            if set(saved) != set(ids):
                raise ValueError("missing regenerated residual")
        if event["tool"] != "write_weights":
            continue
        dest = args["selection"]
        if len(dest) != 1 or type(dest[0]) is not int:
            raise ValueError("expected writes to individual rows")
        if (args["tensor"], dest[0]) != (name, row):
            other_rows.add((args["tensor"], dest[0]))
        if event["call"] == edit["call"]:
            if remove_term:
                args["value"] = args["value"]["left"]
            else:
                scaled = args["value"]["right"]
                for side, old in zip(("left", "right"), ids, strict=True):
                    scaled["left"][side]["handle"] = saved[old]
                diff = store.resolve(saved[ids[0]]).float() - store.resolve(saved[ids[1]]).float()
                term = scaled["factor"] * diff
                diagnostics = {
                    "width": term.numel(), "nonzero_coordinates": int(term.count_nonzero()),
                    "scaled_difference_l2_before_weight_rounding": float(term.double().norm()),
                }
        result = store.write(**args)
        if not remove_term or event["call"] < edit["call"]:
            for key in ("revision", "changed_parameters", "selected_parameters"):
                if result[key] != event["result"][key]:
                    raise ValueError(f"different write result on call {event['call']} for {key}")
        if event["call"] == edit["call"]:
            immediate = snapshot()
    receipt = store.export(path)
    others = {(key, index): store.params[key][index].detach().cpu().clone() for key, index in other_rows}
    return receipt, immediate, snapshot(), others, diagnostics


def patches_equal(first, second):
    with safe_open(first, framework="pt") as a, safe_open(second, framework="pt") as b:
        if a.metadata() != b.metadata() or set(a.keys()) != set(b.keys()):
            return False
        for key in a.keys():
            left, right = a.get_tensor(key), b.get_tensor(key)
            if left.dtype != right.dtype or not torch.equal(left, right):
                return False
    return True


def row_difference(with_term, without_term):
    delta = with_term.double() - without_term.double()
    return {"row_width": delta.numel(), "differing_coordinates": int(delta.count_nonzero()),
            "l2": float(delta.norm()), "max_abs": float(delta.abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    source, out = options.episode, options.out
    original = source / "patch.safetensors"
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    if original_hash != json.loads((source / "receipt.json").read_text())["sha256"]:
        raise ValueError("original patch mismatches the recorded submission receipt")
    trace = source / "trajectory.jsonl"
    events = [event for line in trace.read_text().splitlines()
              if (event := json.loads(line))["type"] == "tool" and "result" in event]
    probe, edit, ids = find_term(events)
    out.mkdir(parents=True, exist_ok=False)
    reproduced = out / "replayed.safetensors"
    replayed, before, final, other_rows, magnitudes = replay(events, probe, edit, ids, reproduced, remove_term=False)
    if not patches_equal(original, reproduced):
        raise ValueError("unmodified replay differs from the submitted patch")
    gc.collect()
    torch.cuda.empty_cache()
    removed, without, final_without, other_without, _ = replay(
        events, probe, edit, ids, out / "ablated.safetensors", remove_term=True)
    if other_rows.keys() != other_without.keys() or any(
            not torch.equal(value, other_without[key]) for key, value in other_rows.items()):
        raise ValueError("residual removal changed another edited row")
    value = edit["arguments"]["value"]
    report = {
        "episode": source.name, "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "original_sha256": original_hash, "replayed_sha256": replayed["sha256"],
        "ablated_sha256": removed["sha256"], "replay_matches_original": True,
        "hash_matches_original": replayed["sha256"] == original_hash,
        "probe_call": probe["call"], "layer": probe["result"]["probe"]["layer"], "write_call": edit["call"],
        "tensor": edit["arguments"]["tensor"], "row": edit["arguments"]["selection"][0],
        "retained_operand": value["left"], "removed_operand": value["right"],
        "later_write_calls_replayed": [e["call"] for e in events if e["tool"] == "write_weights" and e["call"] > edit["call"]],
        "residual_term": magnitudes, "immediate_row_difference": row_difference(before, without),
        "final_row_difference": row_difference(final, final_without), "other_edited_rows_verified_equal": len(other_rows),
    }
    (out / "replay.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
