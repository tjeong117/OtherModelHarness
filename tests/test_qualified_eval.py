import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_public_task_uses_public_fields_and_filters_alias_provenance(monkeypatch):
    pytest.importorskip("iwibench")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import qualified_eval as study

    aliases = {
        "dax": {
            "token_ids": [3],
            "standalone_token_ids": [3],
            "leading_space_token_ids": [4],
            "occurrences": [{"private_id": "SECRET"}],
        }
    }
    tok = SimpleNamespace(encode=lambda s, add_special_tokens: list(s.encode()))
    row = {
        "id": "row",
        "family": "alias_acquisition",
        "public_practice_json": json.dumps([{"prompt": "practice"}]),
        "objective": "bind dax",
        "unused_private_value": "SECRET",
    }
    ep = {"id": "row", "family": "alias_acquisition", "parameters": {"binding": "dax"}, "targets": ["SECRET"]}
    monkeypatch.setattr(study, "_alias_tokens", lambda episode, tokenizer: aliases)
    packet = study.public_qualified_task(row, ep, SimpleNamespace(tokenizer=tok))
    assert set(packet) == {"objective", "practice", "token_ids"}
    assert packet["token_ids"]["alias_context"]["dax"]["leading_space_token_ids"] == [4]
    assert packet["token_ids"][" dax"] == list(b" dax")
    assert "SECRET" not in json.dumps(packet)
    with pytest.raises(ValueError):
        study.public_qualified_task({**row, "id": "different"}, ep, SimpleNamespace(tokenizer=tok))
