import importlib.util
import io
import json
from pathlib import Path

import pytest


def load_control():
    path = Path(__file__).resolve().parents[1] / "examples/astra_native_control.py"
    spec = importlib.util.spec_from_file_location("astra_native_control", path)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    return control


def test_native_loop_has_no_added_guide_or_submit_tool(monkeypatch):
    native = load_control()
    packet = dict(session_id="sid", objective="edit", manifest={"private": False}, practice=[], token_ids={},
                  limits={}, counters={}, calibration="hidden")
    exported = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
                for name in sorted(native.NAMES)]
    call = {"id": "c1", "type": "function", "function": {"name": "run_target", "arguments": "{}"}}
    seen = []

    def editor(messages, tools):
        seen.append(messages.copy())
        assert tools == exported
        return {"choices": [{"message": {"role": "assistant", "content": "done" if len(seen) > 1 else None,
                                         "tool_calls": [] if len(seen) > 1 else [call]}}], "usage": {}}

    monkeypatch.setattr(native, "tool_reply", lambda *args: ("response", True))
    trace = io.StringIO()
    reason, successes = native.run_native_loop(object(), packet, exported, editor, 2, trace)
    assert reason == "natural_language" and successes == {"run_target": 1}
    assert [m["role"] for m in seen[0]] == ["user"]
    assert json.loads(seen[0][0]["content"].split("\n\n", 1)[1]) == {
        k: packet[k] for k in ("objective", "practice", "manifest", "token_ids", "limits")}
    assert [m["role"] for m in seen[-1]] == ["user", "assistant", "tool"]
    assert seen[-1][-1]["content"] == "response"
    assert [json.loads(line)["type"] for line in trace.getvalue().splitlines()] == ["updater", "tool", "updater"]


def test_native_turn_cap_finalizes_last_writes(monkeypatch):
    native = load_control()
    tools = [{"function": {"name": name, "parameters": {"type": "object"}}} for name in native.NAMES]
    packet = dict(session_id="sid", objective="edit", practice=[], manifest={}, token_ids={}, limits={})
    call = {"id": "last", "function": {"name": "write_weights", "arguments": "{}"}}
    reply = {"choices": [{"message": {"role": "assistant", "tool_calls": [call]}}], "usage": {}}
    observed = []
    def dispatch(service, sid, validators, name, raw):
        observed.append((sid, name))
        return "ok", True

    monkeypatch.setattr(native, "tool_reply", dispatch)
    result = native.run_native_loop(object(), packet, tools, lambda *_: reply, 1, io.StringIO())
    assert result == ("turn_limit", {"write_weights": 1})
    assert observed == [("sid", "write_weights")]
    with pytest.raises(ValueError, match="four native"):
        native.run_native_loop(object(), packet, [], lambda *_: reply, 1, io.StringIO())


def test_recover_original_sparse_patch_without_editor_credentials(tmp_path, monkeypatch):
    import hashlib
    import torch
    from safetensors.torch import save_file

    native = load_control()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    meta = {"protocol": "native-iwibench-v1", "target": "target", "target_revision": "revision",
            "max_turns": 12, "case": "alias", "seed": 3}
    native.save(tmp_path / "protocol.json", meta)
    patch = tmp_path / "patch.safetensors"
    sparse = [{"name": "w", "indices": "ix", "values": "val"}]
    save_file({"ix": torch.tensor([1, 3]), "val": torch.tensor([2., 4.])}, str(patch),
              metadata={"model_id": "target", "revision": "revision", "format": "iwibench.replacements.v1",
                        "sparse": json.dumps(sparse)})
    events = [
        {"type": "updater", "turn": 1, "response": {"tool_calls": [{}]}, "usage": {"cost_upper_bound_usd": 0.05}},
        {"type": "tool", "ok": True, "tool": "write_weights",
         "public_response": json.dumps({"result": {}, "counters": {"tool_calls": 1}, "revision": 1}, indent=2)},
        {"type": "updater", "turn": 2, "response": {"tool_calls": []}, "usage": {"cost_upper_bound_usd": 0.06}},
    ]
    (tmp_path / "trajectory.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    recovered = native.recover_report(tmp_path)
    assert recovered == native.recover_report(tmp_path)
    assert recovered["cost_upper_bound_usd"] == 0.11
    assert recovered["provider"] == [{"model": None, "usage": e["usage"]} for e in events if e["type"] == "updater"]
    assert recovered["recovery"]["response_model_echo_available"] is False
    assert recovered["receipt"] == {"sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                                    "bytes": patch.stat().st_size, "format": "iwibench.replacements.v1",
                                    "changed_parameters": 2, "revision": 1}
    assert recovered["target_usage_including_host"] == {"tool_calls": 1}
