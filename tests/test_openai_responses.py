import json

import pytest

from iwi_harness import EditingSession
from iwi_harness.openai_responses import AstraBudget, AstraBudgetExceeded, AstraProviderError, AstraResponses
from iwi_harness.targets import tiny_sae, tiny_target
from iwi_harness.updater import run_updater


def answer(items, status="completed"):
    return {"id": "r_test", "model": "gpt-6.2-astra", "output": items, "status": status,
            "usage": {"input_tokens": 100, "output_tokens": 25,
                      "input_tokens_details": {"cached_tokens": 10},
                      "output_tokens_details": {"reasoning_tokens": 20}}}


def function(name, call_id):
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": name, "arguments": "{}"}


def test_multiturn_native_reasoning_and_all_session_tools(tmp_path, monkeypatch):
    budget = AstraBudget(1)
    editor = AstraResponses(api_key="test-fixture", budget=budget)
    session = EditingSession(tiny_target(), artifact_dir=tmp_path, sae=tiny_sae(), detailed_writes=True)
    native = [{"type": "reasoning", "encrypted_content": "opaque-fixture", "summary": [], "id": "rs_test"},
              {"type": "message", "role": "assistant", "phase": "commentary", "id": "msg_test",
               "content": [{"type": "output_text", "text": "Inspecting."}]},
              function("read_weights", "call_1"), function("read_weights", "call_2")]
    results = iter([answer(native), answer([function("submit_patch", "call_3")])])
    posts = []

    def post(path, payload):
        posts.append((path, payload))
        if path == "responses/input_tokens":
            return {"input_tokens": 100}
        return next(results)

    monkeypatch.setattr(editor, "_post", post)
    receipt = run_updater(session, "Inspect the target and submit.", editor, max_turns=2, guide=True)
    assert receipt["changed_parameters"] == 0
    assert [path for path, _ in posts] == ["responses/input_tokens", "responses"] * 2
    second = posts[-1][1]
    assert second["store"] is False and "previous_response_id" not in second
    assert "reasoning.encrypted_content" in second["include"]
    assert second["input"][2:6] == native
    assert [item["call_id"] for item in second["input"][6:8]] == ["call_1", "call_2"]
    assert all(item["type"] == "function_call_output" for item in second["input"][6:8])
    assert all("tensors" in json.loads(item["output"]) for item in second["input"][6:8])
    assert "final turn" in second["input"][-1]["content"]
    assert second["tools"] == [{"type": "function", **t["function"], "strict": False} for t in session.tools]
    assert second["reasoning"] == {"effort": "medium"} and second["max_output_tokens"] == 4096
    assert budget.spent_upper_bound_usd == pytest.approx(0.006)
    trace = (tmp_path / "trajectory.jsonl").read_text()
    assert "opaque-fixture" not in trace and "test-fixture" not in trace
    usage = [row["usage"] for row in map(json.loads, trace.splitlines()) if row["type"] == "updater"]
    assert usage == [r["usage"] for r in editor.records]
    assert all(u["completion_tokens_details"]["reasoning_tokens"] == 20 for u in usage)


def test_shared_budget_refuses_generation_and_keeps_unknown_charges(monkeypatch):
    budget = AstraBudget(0.003)
    editor = AstraResponses(api_key="test-fixture", budget=budget, max_output_tokens=25)
    requests = []

    def post(path, payload):
        requests.append(path)
        if path == "responses/input_tokens":
            return {"input_tokens": 100}
        raise AstraProviderError("simulated lost response")

    monkeypatch.setattr(editor, "_post", post)
    with pytest.raises(AstraProviderError):
        editor([], [])
    next_episode = AstraResponses(api_key="test-fixture", budget=budget, max_output_tokens=25)
    monkeypatch.setattr(next_episode, "_post", post)
    with pytest.raises(AstraBudgetExceeded):
        next_episode([], [])
    assert requests == ["responses/input_tokens", "responses", "responses/input_tokens"]
    assert budget.spent_upper_bound_usd == pytest.approx(0.003)


def test_incomplete_response_with_tool_call_never_dispatches(tmp_path, monkeypatch):
    session = EditingSession(tiny_target(), artifact_dir=tmp_path)
    editor = AstraResponses(api_key="test-fixture", budget=AstraBudget(1))
    result = answer([function("submit_patch", "call_test")], "incomplete")
    result["incomplete_details"] = {"reason": "max_output_tokens"}
    monkeypatch.setattr(editor, "_post", lambda path, _: {"input_tokens": 100}
                        if path.endswith("input_tokens") else result)
    with pytest.raises(AstraProviderError, match="max_output_tokens"):
        run_updater(session, "Edit.", editor)
    assert session.receipt is None and editor.records[-1]["status"] == "incomplete"
    assert editor.budget.spent_upper_bound_usd == pytest.approx(0.003)


def test_long_context_rejected_before_generation(monkeypatch):
    editor = AstraResponses(api_key="test-fixture", budget=AstraBudget(100))
    requests = []

    def post(path, _):
        requests.append(path)
        return {"input_tokens": 272_001}

    monkeypatch.setattr(editor, "_post", post)
    with pytest.raises(AstraBudgetExceeded, match="short-context"):
        editor([], [])
    assert requests == ["responses/input_tokens"]
    assert editor.budget.spent_upper_bound_usd == 0


def test_cli_astra_submits_and_uses_all_session_tools(tmp_path, monkeypatch, capsys):
    from iwi_harness.cli import main

    seen = []

    def post(_, path, payload):
        seen.append(payload)
        return {"input_tokens": 100} if path.endswith("input_tokens") else answer([function("submit_patch", "c")])

    monkeypatch.setenv("IWI_TEST_ASTRA_KEY", "redacted-fixture")
    monkeypatch.setattr(AstraResponses, "_post", post)
    main(["edit", "--astra", "--target", "tiny", "--out", str(tmp_path), "--objective", "Inspect and submit.",
          "--key-env", "IWI_TEST_ASTRA_KEY", "--reasoning-effort", "high", "--detailed-writes"])
    assert json.loads(capsys.readouterr().out)["changed_parameters"] == 0
    assert seen[-1]["model"] == "gpt-6.2-astra"
    assert seen[-1]["reasoning"] == {"effort": "high"}
    assert {"inspect_target", "write_weights", "contrast_target"} <= {fn["name"] for fn in seen[-1]["tools"]}
