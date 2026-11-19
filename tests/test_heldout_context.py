import gzip
import json
from pathlib import Path

import pytest


def test_manifest_controls_preserve_shared_metadata_and_exact_pager_order(monkeypatch):
    pytest.importorskip("tiktoken")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "examples"))
    from audit_heldout_context import MANIFEST_SHA, digest, representations

    archived = gzip.decompress((root / "evidence/heldout-token-audit/gemma-manifest.json.gz").read_bytes())
    assert digest(archived) == MANIFEST_SHA
    manifest = json.loads(archived)
    forms = representations(manifest)
    for form in forms.values():
        assert {k: v for k, v in form.items() if k != "tensors"} == {k: v for k, v in manifest.items() if k != "tensors"}
    writable = forms["all_writable_native_records"]["tensors"]
    full = forms["all_writable_names_shapes"]["tensors"]
    assert len(manifest["tensors"]) == 883 and len(writable) == 444
    assert full[:-1] == [[t["name"], t["shape"]] for t in writable]
    assert full[-1] == ["lm_head.weight", full[0][1]]
    assert forms["four_writable_names_shapes"]["tensors"] == full[:4]
    trace = root / "evidence/astra-sae-guided-v2/suggestion-2/trajectory.jsonl"
    events = (json.loads(line) for line in trace.read_text().splitlines())
    preview = next(e["result"]["tensors"] for e in events if e.get("tool") == "read_weights")
    assert full[:4] == [[t["name"], t["shape"]] for t in preview]
