import json

import pytest

from iwi_harness.demo import verify_tiny
from iwi_harness.session import EditingSession
from iwi_harness.targets import tiny_target


CASES = [{"prompt": "color dax", "answers": ["yes"], "slice": "improve"},
         {"prompt": "color red", "answers": ["yes"], "slice": "preserve"}]
WRITE = {"tensor": "embed.weight", "operation": "copy", "selection": [5],
         "source": {"tensor": "embed.weight", "selection": [3]}}


@pytest.mark.parametrize("arm", list("ABCD"))
def test_matched_editor_surfaces(tmp_path, arm):
    internal, preview = arm in "BD", arm in "CD"
    session = EditingSession(tiny_target(), artifact_dir=tmp_path, experiment_tools_only=True,
                             interpretability=internal, internal_measurements=internal, edit_preview=preview)
    menu = {t["function"]["name"]: t["function"]["parameters"] for t in session.tools}
    common = {"run_target", "read_weights", "write_weights", "submit_patch", "tokenize_target", "test_behavior"}
    assert menu.keys() == common | ({"preview_weights", "commit_preview", "discard_preview"} if preview else set())
    assert ("intervention" in menu["test_behavior"]["properties"]) == internal
    assert "probe" not in menu["run_target"]["properties"]
    assert "sae" not in json.dumps(menu["test_behavior"])
    if not internal:
        with pytest.raises(ValueError, match="intervention"):
            session.call("test_behavior", {"cases": CASES, "intervention": {"layer": 0, "scale": 0}})
        assert session.runner.usage["forwards"] == 0


@pytest.mark.parametrize("detailed", [False, True])
def test_session_previews_then_exports_measured_delta(tmp_path, detailed):
    session = EditingSession(tiny_target(), artifact_dir=tmp_path, interpretability=False,
                             edit_preview=True, detailed_writes=detailed)
    with pytest.raises(ValueError, match="writes"):
        session.call("preview_weights", {"cases": CASES, "writes": [{**WRITE, "factor": 7}]})
    assert session.runner.usage["forwards"] == 0
    baseline = session.call("test_behavior", {"cases": CASES})
    result = session.call("preview_weights", {"writes": [WRITE], "cases": CASES})
    assert result["behavior"]["improve"]["fixed"] == [0]
    assert result["behavior"]["preserve"]["retained_correct"] == 1
    record = json.loads((tmp_path / result["artifact"]).read_text())
    assert record["before"] == baseline["baseline"] and record["after"][0]["correct"]
    assert result["cost"]["forwards"] > 0 and record["writes"] == [WRITE]
    assert session.call("test_behavior", {"cases": CASES})["baseline"] == baseline["baseline"]
    committed = session.call("commit_preview", {"candidate_id": result["candidate_id"]})
    assert committed["sha256"] == result["sha256"]
    receipt = session.submit()
    assert verify_tiny(tmp_path / "patch.safetensors", receipt["sha256"])["reward"] == 1
