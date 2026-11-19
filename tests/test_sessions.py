import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from iwi_harness import EditingSession
from iwi_harness.demo import verify_tiny
from iwi_harness.targets import tiny_sae, tiny_target
from iwi_harness.updater import run_updater


def test_immutable_submit_and_ownership(tmp_path):
    target = tiny_target()
    session = EditingSession(target, artifact_dir=tmp_path / "a", max_tool_calls=1)
    with pytest.raises(ValueError, match="freshly loaded"):
        EditingSession(target, artifact_dir=tmp_path / "b")
    with pytest.raises(FileExistsError):
        EditingSession(tiny_target(), artifact_dir=tmp_path / "a")
    session.call("read_weights", {"tensor": "embed.weight", "selection": [5]})
    with pytest.raises(ValueError, match="budget"):
        session.call("read_weights", {})
    with pytest.raises(ValueError, match="budget"):
        session.call("submit_patch", {"unexpected": True})
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: session.submit(), range(2)))
    assert receipts[0] == receipts[1]
    with pytest.raises(ValueError, match="frozen"):
        session.call("write_weights", {})
    assert verify_tiny(tmp_path / "a/patch.safetensors")["reward"] == 0
    assert verify_tiny(tmp_path / "a/patch.safetensors", "wrong hash")["status"] == "invalid_patch"


def test_compact_mode_enforced_at_dispatch_and_logged(tmp_path):
    session = EditingSession(tiny_target(), sae=tiny_sae(), artifact_dir=tmp_path, interpretability=False)
    assert {t["function"]["name"] for t in session.tools} == {
        "read_weights", "write_weights", "run_target", "submit_patch"}
    with pytest.raises(ValueError, match="unknown tool"):
        session.call("contrast_target", {})
    for name in ("feature_examples", "patch_activations", "rank_attributions"):
        with pytest.raises(ValueError, match="unknown tool"):
            session.call(name, {})
    with pytest.raises(ValueError, match="probe"):
        session.call("run_target", {"prompts": ["red"], "probe": {"layer": 0, "kind": "sae"}})
    with pytest.raises(ValueError):
        session.call("write_weights", {"tensor": "embed.weight", "operation": "fill", "value": float("nan")})
    assert session.runner.usage["forwards"] == 0
    events = [json.loads(e) for e in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert len(events) == 6 and all("error" in e for e in events)


def test_updater_recovers_from_invalid_arguments_and_receives_real_results(tmp_path):
    session = EditingSession(tiny_target(), artifact_dir=tmp_path, token_counter=len)
    requests = [("read_weights", "{"), ("read_weights", '{"limit":1}'), ("submit_patch", "{}")]

    def provider(messages, tools):
        if len(requests) == 2:
            assert "error" in json.loads(messages[-1]["content"])
        if len(requests) == 1:
            assert json.loads(messages[-1]["content"])["tensors"][0]["name"] == "embed.weight"
        assert tools[0]["function"]["name"] == "read_weights"
        name, arguments = requests.pop(0)
        call = dict(type="function", id=str(len(requests)), function=dict(name=name, arguments=arguments))
        return dict(usage={"prompt_tokens": 100, "completion_tokens": 10},
                    choices=[{"message": {"role": "assistant", "content": None, "tool_calls": [call]}}])

    receipt = run_updater(session, "Determine whether the model needs an edit.", provider)
    assert receipt["changed_parameters"] == 0
    events = [json.loads(e) for e in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert [e["usage"]["prompt_tokens"] for e in events if e["type"] == "updater"] == [100]*3
    assert [e["tool"] for e in events if e["type"] == "tool"] == ["read_weights", "submit_patch"]


@pytest.mark.parametrize("args", [[], ["--literal"]])
def test_cli_demo_grades_in_a_fresh_process(tmp_path, args):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-m", "iwi_harness", "demo", "--out", str(tmp_path), *args],
                            check=True, capture_output=True, text=True, env=env)
    demo = json.loads(result.stdout)
    assert (demo["public_before"], demo["public_after"]) == ("no", "yes")
    assert demo["fresh_verifier"]["reward"] == 1
    assert demo["fresh_verifier"]["retained"] == 2
    assert demo["fresh_verifier"]["patch_sha256"] == demo["receipt"]["sha256"]
    assert demo["feature_0_margin_change"] == -3
    assert "check dax" not in (tmp_path / "trajectory.jsonl").read_text()


def test_guided_updater_meters_preview_and_preserves_submission_choice(tmp_path):
    from iwi_harness.updater import NoSubmission
    session = EditingSession(tiny_target(), artifact_dir=tmp_path)
    histories = []

    def provider(messages, tools):
        histories.append([msg.copy() for msg in messages])
        # Even on the last turn, the model may decide to finish without a submission.
        calls = [] if len(histories) == 2 else [
            {"id": "one", "type": "function", "function": {"name": "read_weights", "arguments": "{}"}}]
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}]}

    with pytest.raises(NoSubmission):
        run_updater(session, "Explore public behavior", provider, max_turns=2, guide=True)
    assert "embed.weight" in histories[0][0]["content"]
    assert "final turn" in histories[1][-1]["content"]
    events = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert [e["arguments"] for e in events if e["type"] == "tool"] == [{"limit": 4}, {}]
    assert session.receipt is None


def test_manifest_control_only_changes_visible_entries_and_allows_paging(tmp_path):
    from iwi_harness.manifest import GUIDANCE, group_entries
    from iwi_harness.session import encode_json

    histories, records, tool_calls = {}, {}, {}
    for mode in ("paged", "eager", "grouped"):
        target = tiny_target()
        for index in range(251):
            target.model.lm_head.register_parameter(f"tied_alias_{index}", target.model.lm_head.weight)
        directory = tmp_path / mode
        session = EditingSession(target, artifact_dir=directory, max_tool_calls=4)
        entries = [[name, list(param.shape)] for name, param in target.weights().items()]
        if mode == "grouped":
            rendered = encode_json({"tensors": group_entries(entries), "tensor_encoding": GUIDANCE})
        else:
            rendered = encode_json(entries if mode == "eager" else entries[:4])

        def provider(messages, tools):
            if mode not in histories:
                assert rendered in messages[0]["content"]
                histories[mode] = (messages[0]["content"].replace(rendered, "<manifest>", 1), messages[1], tools)
                name, args = "read_weights", {"offset": 256}
            else:
                result = json.loads(messages[-2]["content"])
                assert result["tensors"][-1]["name"] == entries[-1][0]
                name, args = "submit_patch", {}
            call = {"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [call]}}]}

        receipt = run_updater(session, "Edit the target", provider, max_turns=2, opening_manifest=mode)
        assert receipt["changed_parameters"] == 0
        records[mode] = json.loads((directory / "opening-manifest.json").read_text())
        assert records[mode]["rendered_list_bytes"] == len(rendered.encode())
        assert records[mode]["metadata_tool_calls"] == 2
        events = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
        tool_calls[mode] = [event["arguments"] for event in events if event["type"] == "tool"]
    assert histories["paged"] == histories["eager"] == histories["grouped"]
    assert tool_calls["paged"] == tool_calls["eager"] == tool_calls["grouped"] == [
        {"offset": 0, "limit": 256}, {"offset": 256, "limit": 256}, {"offset": 256}, {},
    ]
    assert records["paged"]["shown_entries"] == 4
    assert records["eager"]["shown_entries"] == records["grouped"]["shown_entries"] == len(entries)
    assert records["grouped"]["expanded_entries_sha256"] == records["eager"]["rendered_list_sha256"]
    for field in ("available_entries", "ordered_entries_sha256", "objective_sha256", "tool_schema_sha256"):
        assert records["paged"][field] == records["eager"][field] == records["grouped"][field]
