import json

import pytest

from iwi_harness.demo import verify_tiny
from iwi_harness.session import EditingSession
from iwi_harness.targets import tiny_sae, tiny_target


CASES = [
    {"prompt": "color dax", "answers": ["yes"], "slice": "improve"},
    {"prompt": "color red", "answers": ["yes"], "slice": "preserve"},
]
WRITE = {
    "tensor": "embed.weight",
    "operation": "copy",
    "selection": [5],
    "source": {"tensor": "embed.weight", "selection": [3]},
}


@pytest.mark.parametrize("arm", list("ABCD"))
def test_matched_editor_surfaces(tmp_path, arm):
    internal, preview = arm in "BD", arm in "CD"
    session = EditingSession(
        tiny_target(),
        artifact_dir=tmp_path,
        sae=tiny_sae() if internal else None,
        experiment_tools_only=True,
        interpretability=internal,
        internal_measurements=internal,
        edit_preview=preview,
    )
    menu = {t["function"]["name"]: t["function"]["parameters"] for t in session.tools}
    common = {
        "run_target",
        "read_weights",
        "write_weights",
        "submit_patch",
        "tokenize_target",
        "test_behavior",
    }
    assert menu.keys() == common | (
        {"preview_weights", "commit_preview", "discard_preview"} if preview else set()
    )
    assert ("intervention" in menu["test_behavior"]["properties"]) == internal
    assert "probe" not in menu["run_target"]["properties"]
    if internal:
        intervention = menu["test_behavior"]["properties"]["intervention"]
        assert intervention["properties"]["kind"] == {"const": "sae"}
    if not internal:
        with pytest.raises(ValueError, match="intervention"):
            session.call("test_behavior", {"cases": CASES, "intervention": {"layer": 0, "scale": 0}})
        assert session.runner.usage["forwards"] == 0


def test_unsupported_target_with_no_dictionary_gets_plain_behavior(tmp_path):
    session = EditingSession(
        tiny_target(), artifact_dir=tmp_path, experiment_tools_only=True, internal_measurements=True
    )
    schema = next(
        t["function"]["parameters"] for t in session.tools if t["function"]["name"] == "test_behavior"
    )
    assert "intervention" not in schema["properties"]
    assert session.info["public_experiments"]["native_unit_layers"] == []


@pytest.mark.parametrize("detailed", [False, True])
def test_session_previews_then_exports_measured_delta(tmp_path, detailed):
    session = EditingSession(
        tiny_target(),
        artifact_dir=tmp_path,
        interpretability=False,
        edit_preview=True,
        detailed_writes=detailed,
    )
    with pytest.raises(ValueError, match="writes"):
        session.call("preview_weights", {"cases": CASES, "writes": [{**WRITE, "factor": 7}]})
    with pytest.raises(ValueError, match="Omit token_ids"):
        session.call("test_behavior", {"cases": CASES, "token_ids": [[3, 2], [5, 2]]})
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


def test_host_cases_match_explicit_calls_and_keep_wire_arguments_short(tmp_path):
    from copy import deepcopy

    host_cases = deepcopy(CASES)
    bound = EditingSession(tiny_target(), artifact_dir=tmp_path / "bound", public_cases=host_cases,
                           edit_preview=True, interpretability=False)
    explicit = EditingSession(tiny_target(), artifact_dir=tmp_path / "explicit", edit_preview=True,
                              interpretability=False)
    host_cases[0]["answers"][0] = "unexpected"
    host_cases.pop()
    assert bound.runner.usage["forwards"] == 0
    assert bound.info["public_experiments"]["default_case_count"] == 2
    assert "color dax" not in json.dumps([bound.info, bound.tools])
    assert bound.call("test_behavior", {}) == explicit.call("test_behavior", {"cases": CASES})
    preview = bound.call("preview_weights", {"writes": [WRITE]})
    other = explicit.call("preview_weights", {"writes": [WRITE], "cases": CASES})
    assert preview["behavior"] == other["behavior"]
    assert preview["sha256"] == other["sha256"]
    assert preview["cost"] == other["cost"]
    assert bound.store.revision == explicit.store.revision == 0
    bound.call("commit_preview", {"candidate_id": preview["candidate_id"]})
    explicit.call("commit_preview", {"candidate_id": other["candidate_id"]})
    receipt = bound.submit()
    assert receipt == explicit.submit()
    assert verify_tiny(bound.artifact_dir / "patch.safetensors", receipt["sha256"])["reward"] == 1
    record = json.loads((bound.artifact_dir / preview["artifact"]).read_text())
    assert record["cases"] == CASES
    trace = [json.loads(line) for line in (bound.artifact_dir / "trajectory.jsonl").read_text().splitlines()]
    assert trace[0]["arguments"] == {} and trace[0]["request"]["utf8_bytes"] == 2
    assert trace[1]["arguments"] == {"writes": [WRITE]}


def test_host_cases_allow_overrides_and_charge_target_budget(tmp_path):
    session = EditingSession(tiny_target(), artifact_dir=tmp_path, public_cases=CASES,
                             edit_preview=True, interpretability=False)
    result = session.call("test_behavior", {"cases": [CASES[1]], "token_ids": [6, 7]})
    assert len(result["baseline"]) == 1 and result["baseline"][0]["margin"] is not None
    preview = session.call("preview_weights", {"writes": [WRITE], "cases": [CASES[1]]})
    assert preview["behavior"].keys() == {"preserve"}
    for invalid in [None, []]:
        with pytest.raises(ValueError, match="cases"):
            session.call("test_behavior", {"cases": invalid})
    assert len(session.call("test_behavior", {})["baseline"]) == len(CASES)
    restricted = EditingSession(tiny_target(), artifact_dir=tmp_path / "limited", public_cases=CASES, max_forwards=1)
    assert restricted.runner.usage["forwards"] == 0
    with pytest.raises(ValueError, match="budget"):
        restricted.call("test_behavior", {})
    assert restricted.runner.usage["forwards"] == 1


@pytest.mark.parametrize("invalid", [[], {"public_cases": CASES, "private": []},
                                   [{**CASES[0], "private": "extra"}], [{**CASES[0], "answers": ["  "]}],
                                   [{**CASES[0], "prompt": "  "}]])
def test_invalid_host_cases_cannot_claim_a_target_or_artifact_directory(tmp_path, invalid):
    target = tiny_target()
    out = tmp_path / "episode"
    with pytest.raises(ValueError):
        EditingSession(target, artifact_dir=out, public_cases=invalid)
    assert not out.exists()
    session = EditingSession(target, artifact_dir=out, behavior_checks=True)
    with pytest.raises(ValueError, match="cases"):
        session.call("test_behavior", {})
