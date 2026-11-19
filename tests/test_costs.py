import json

import pytest

from iwi_harness.costs import compare_payloads


def test_paired_payloads_are_reproducible_and_replay_equivalent():
    pytest.importorskip("tiktoken")
    first = compare_payloads()
    assert first == compare_payloads()
    json.dumps(first, allow_nan=False)
    assert first["parity"] == {"exact_replayed_weights": True, "exact_patch_content": True, "exact_patch_bytes": True}
    assert first["inline"]["patch_content_sha256"] == first["handle"]["patch_content_sha256"]
    for arm in ("inline", "handle"):
        calls = first[arm]["calls"]
        assert set(calls) == {"read_weights", "write_weights"}
        for unit in ("tokens", "utf8_bytes"):
            assert first[arm]["total"][unit] == sum(item[unit] for call in calls.values() for item in call.values())
    for unit in ("tokens", "utf8_bytes"):
        assert first["handle"]["calls"]["read_weights"]["response"][unit] < first["inline"]["calls"]["read_weights"]["response"][unit]
        assert first["handle"]["calls"]["write_weights"]["request"][unit] < first["inline"]["calls"]["write_weights"]["request"][unit]
        assert first["savings"][unit] == first["inline"]["total"][unit] - first["handle"]["total"][unit]
