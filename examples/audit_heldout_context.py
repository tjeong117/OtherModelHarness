"""Measure a manifest-only opening-context contrast on the 150 public IWIBench test rows."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
HUB_REVISION = "300a9d7271d9e3a9dd5bfc14edcbceccb66d5966"
CATALOG_URL = f"https://huggingface.co/datasets/hcltech-robotics/iwibench/resolve/{HUB_REVISION}/data/test.jsonl"
CATALOG_SHA = "d0a12f4d5ec9128ee5303739eb9c0a7331b1ec1e2225f8f7e256e95af0fa598b"
MANIFEST_SHA = "469d006bfa0e956850b41ffdc0f31d158971291f860729da4521f74726d5fb36"
SYSTEM = (
    "Inspect and edit the target with the supplied tools. Use read_weights with offset and limit "
    "to page through writable tensors as needed. Use submit_patch to finish."
)


def pack(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_catalog(path):
    if path is None:
        with urlopen(CATALOG_URL, timeout=60) as response:
            data = response.read()
    else:
        data = path.read_bytes()
    if digest(data) != CATALOG_SHA:
        raise ValueError("Test JSONL differs from data/test.jsonl in the pinned official release")
    return [json.loads(line) for line in data.splitlines()]


def representations(manifest):
    """Keep all checkpoint metadata and vary only manifest.tensors."""
    full = manifest["tensors"]
    writable = [tensor for tensor in full if tensor["writable"]]
    compact = [[tensor["name"], tensor["shape"]] for tensor in writable]
    # Gemma registers the output head after the language model. Its tied name is
    # a separate row in the harness pager (named_parameters(remove_duplicate=False)).
    aliases = [[alias, tensor["shape"]] for tensor in writable for alias in tensor["aliases"] if alias != tensor["name"]]
    if [name for name, _ in aliases] != ["lm_head.weight"]:
        raise ValueError("Unexpected alias layout for the pinned Gemma checkpoint")
    compact.extend(aliases)
    return {
        "full_native_records": manifest,
        "all_writable_native_records": {**manifest, "tensors": writable},
        "all_writable_names_shapes": {**manifest, "tensors": compact},
        "four_writable_names_shapes": {**manifest, "tensors": compact[:4]},
    }


def public_fields(row):
    return {"objective": row["objective"], "parameters": json.loads(row["parameters_json"]),
            "practice": json.loads(row["public_practice_json"])}


def summarize(records, arms):
    metrics = {}
    for arm in arms:
        tokens = [record["arms"][arm]["tokens"] for record in records]
        sizes = [record["arms"][arm]["bytes"] for record in records]
        metrics[arm] = {"sum_tokens": sum(tokens), "mean_tokens": sum(tokens) / len(tokens),
                        "min_tokens": min(tokens), "max_tokens": max(tokens), "sum_bytes": sum(sizes)}
    page = metrics["four_writable_names_shapes"]
    deltas = {}
    for arm in arms:
        if arm == "four_writable_names_shapes":
            continue
        pairs = [(r["arms"][arm]["tokens"], r["arms"]["four_writable_names_shapes"]["tokens"]) for r in records]
        deltas[arm] = {"tokens_saved": metrics[arm]["sum_tokens"] - page["sum_tokens"],
                       "fraction_saved": 1 - page["sum_tokens"] / metrics[arm]["sum_tokens"],
                       "min_saved_per_task": min(a - b for a, b in pairs),
                       "max_saved_per_task": max(a - b for a, b in pairs),
                       "tasks_with_savings": sum(a > b for a, b in pairs)}
    return {"tasks": len(records), "by_arm": metrics, "saving_with_four_writable_names_shapes": deltas}


def audit(rows, manifest, tools):
    counts = Counter(row["family"] for row in rows)
    if len(rows) != 150 or len(counts) != 10 or set(counts.values()) != {15} or len({r["id"] for r in rows}) != 150:
        raise ValueError("Expected 150 distinct tasks with 15 rows in each of ten families")
    forms = representations(manifest)
    if any(row["model_id"] != manifest["model_id"] or row["checkpoint_revision"] != manifest["revision"]
           or row["catalog_version"] != "baseline-qualified-v1" or row["calibration_status"] != "eligible"
           for row in rows):
        raise ValueError("Catalogue qualification or checkpoint identity mismatch")
    schemas = {tool["function"]["name"]: tool["function"] for tool in tools}
    if len(schemas) != 13 or not {"offset", "limit"} <= schemas["read_weights"]["parameters"]["properties"].keys():
        raise ValueError("The shared tool menu must support paginated discovery")
    codec = tiktoken.get_encoding("o200k_base")
    common = (SYSTEM, pack(tools))
    common_tokens = sum(len(codec.encode_ordinary(part)) for part in common)
    common_bytes = sum(len(part.encode()) for part in common)
    records = []
    for row in rows:
        task = public_fields(row)
        measurements = {}
        for label, form in forms.items():
            packet = pack({**task, "manifest": form})
            measurements[label] = {"tokens": common_tokens + len(codec.encode_ordinary(packet)),
                                   "bytes": common_bytes + len(packet.encode())}
        records.append({"task_id": row["id"], "family": row["family"], "task_sha256": digest(pack(task).encode()),
                        "arms": measurements})
    return {
        "study": "Manifest-only opening-context ablation, one static request per official held-out task and arm",
        "measurement": "Sum of o200k_base tokens in three separate text fields: common instruction, compact JSON "
                       "function schemas, compact JSON public task packet. UTF-8 bytes use the same fields. "
                       "Excludes provider chat framing, caching, outputs, later tool use and operational grading.",
        "hub_revision": HUB_REVISION, "public_test_jsonl": CATALOG_URL, "public_test_jsonl_sha256": CATALOG_SHA,
        "model_id": manifest["model_id"], "checkpoint_revision": manifest["revision"], "manifest_sha256": MANIFEST_SHA,
        "manifest_counts": {label: len(form["tensors"]) for label, form in forms.items()},
        "public_fields": ["objective", "parameters_json", "public_practice_json"],
        "tool_names": list(schemas), "tools_sha256": digest(pack(tools).encode()), "system": SYSTEM,
        "tiktoken_version": tiktoken.__version__, "tokenizer": codec.name, "shared_instruction_and_schema_tokens": common_tokens,
        "overall": summarize(records, forms),
        "by_family": {family: summarize([r for r in records if r["family"] == family], forms)
                      for family in sorted(counts)},
        "tasks": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, help="Local official test.jsonl (download pinned file if omitted)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = gzip.decompress((ROOT / "evidence/heldout-token-audit/gemma-manifest.json.gz").read_bytes())
    if digest(manifest_bytes) != MANIFEST_SHA:
        raise ValueError("Manifest hash differs from the recorded native Gemma manifest")
    tools = json.loads((ROOT / "evidence/astra-sae-guided-v2/suggestion-2/tools.json").read_text())
    report = audit(load_catalog(args.catalog), json.loads(manifest_bytes), tools)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
