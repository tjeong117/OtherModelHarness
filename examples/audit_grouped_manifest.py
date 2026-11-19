"""Measure lossless layer-range encoding against flat and paged introductions on the same test rows."""
import argparse
import gzip
import json
from pathlib import Path

import tiktoken

from iwi_harness.manifest import GUIDANCE, expand, group_entries

from audit_heldout_context import (CATALOG_SHA, HUB_REVISION, MANIFEST_SHA, ROOT, SYSTEM, digest, load_catalog,
                                   pack, public_fields, representations, summarize)


def audit(rows, manifest, tools, reference):
    flat_name = "all_writable_names_shapes"
    paged_name = "four_writable_names_shapes"
    grouped_name = "all_writable_grouped_names_shapes"
    forms = representations(manifest)
    entries = forms[flat_name]["tensors"]
    grouped = group_entries(entries)
    if expand(grouped) != entries:
        raise ValueError("grouped representation changed the ordered name/shape list")
    grouped_manifest = {**manifest, "tensors": grouped, "tensor_encoding": GUIDANCE}
    codec = tiktoken.get_encoding("o200k_base")
    common = [SYSTEM, pack(tools)]
    common_tokens = sum(len(codec.encode_ordinary(text)) for text in common)
    common_bytes = sum(len(text.encode()) for text in common)
    ref_rows = {row["task_id"]: row for row in reference["tasks"]}
    records = []
    for row in rows:
        task = public_fields(row)
        ref = ref_rows[row["id"]]
        if ref["task_sha256"] != digest(pack(task).encode()):
            raise ValueError("reference public task content differs")
        arms = {}
        for label, form in ((flat_name, forms[flat_name]), (paged_name, forms[paged_name]), (grouped_name, grouped_manifest)):
            packet = pack({**task, "manifest": form})
            arms[label] = {"tokens": common_tokens + len(codec.encode_ordinary(packet)),
                           "bytes": common_bytes + len(packet.encode())}
            if label != grouped_name and arms[label] != ref["arms"][label]:
                raise ValueError("opening measurement differs from reference")
        records.append({"family": row["family"], "arms": arms})
    labels = (flat_name, paged_name, grouped_name)
    overall = summarize(records, labels)
    return {"study": "Offline first-request sensitivity to a lossless range-encoded complete writable listing",
            "hub_revision": HUB_REVISION, "test_jsonl_sha256": CATALOG_SHA, "manifest_sha256": MANIFEST_SHA,
            "tokenizer": codec.name, "tiktoken_version": tiktoken.__version__, "tools_sha256": digest(pack(tools).encode()),
            "ordered_entries_sha256": digest(pack(entries).encode()), "ordered_name_shape_pairs": len(entries),
            "exact_expansion_verified": True, "grouped_tensors": grouped, "charged_tensor_encoding_instructions": GUIDANCE,
            "scope": "Fully counts interpretation instructions in the grouped arm. Flat and four-entry totals reproduce "
                     "the main 150-task static experiment. Model comprehension, lookups, edit quality, provider framing, "
                     "cache accounting, and completed-episode usage were not measured.",
            "overall": overall,
            "by_family": {family: summarize([r for r in records if r["family"] == family], labels)
                          for family in sorted({r["family"] for r in records})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, help="pinned test.jsonl, downloaded if omitted")
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/heldout-token-audit/grouped-manifest.json")
    args = parser.parse_args()
    base = ROOT / "evidence/heldout-token-audit"
    raw = gzip.decompress((base / "gemma-manifest.json.gz").read_bytes())
    if digest(raw) != MANIFEST_SHA:
        raise ValueError("pinned source manifest hash mismatch")
    reference = json.loads((base / "opening-context.json").read_text())
    tools = json.loads((ROOT / "evidence/astra-sae-guided-v2/suggestion-2/tools.json").read_text())
    result = audit(load_catalog(args.catalog), json.loads(raw), tools, reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
