import gzip
import json
from pathlib import Path


def test_layer_ranges_preserve_order_shapes_and_aliases(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "examples"))
    from audit_grouped_manifest import expand, group_entries
    from audit_heldout_context import pack, digest, representations

    raw = gzip.decompress((root / "evidence/heldout-token-audit/gemma-manifest.json.gz").read_bytes())
    entries = representations(json.loads(raw))["all_writable_names_shapes"]["tensors"]
    grouped = group_entries(entries)
    assert expand(grouped) == entries
    assert digest(pack(entries).encode()) == "c741d069bff80012987bdfdcdfb094fe3bf4f049013fdcf784c30c6e5b2ef404"
    assert len(grouped) == 4 and grouped[1]["layers_inclusive"] == [0, 33]
    assert grouped[-1][0] == "lm_head.weight"
    irregular = [["m.layers.0.a", [1]], ["m.layers.1.a", [1]], ["m.layers.3.a", [1]],
                 ["m.layers.4.a", [7]], ["m.other", [1]], ["m.layers.5.a", [7]]]
    encoded = group_entries(irregular)
    assert encoded[0]["layers_inclusive"] == [0, 1]
    assert len(encoded) == 5 and expand(encoded) == irregular
