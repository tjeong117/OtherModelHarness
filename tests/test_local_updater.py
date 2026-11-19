from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest

from iwi_harness.tools import tool_schemas

src = Path(__file__).resolve().parents[1] / "examples" / "local_updater.py"
spec = spec_from_file_location("local_updater_example", src)
example = module_from_spec(spec)
spec.loader.exec_module(example)


def test_local_updater_strips_one_markdown_fence_and_requires_one_known_action():
    action = {"name": "submit_patch", "arguments": {}}
    fenced = "```json\n" + json.dumps(action) + "\n```"
    tools = tool_schemas()
    assert example.parse_tool_call(fenced, tools) == action
    for invalid in (fenced + "\nother", json.dumps([action]), '{"name":"shell","arguments":{}}',
                    '{"name":"read_weights","arguments":"{}"}', json.dumps(action) * 2):
        with pytest.raises(ValueError):
            example.parse_tool_call(invalid, tools)


def test_gemma_chat_translates_tool_history_without_exposing_private_answers():
    messages = [
        {"role": "system", "content": "Public runtime information."},
        {"role": "user", "content": example.OBJECTIVE},
        {"role": "assistant", "content": '{"name":"run_target","arguments":{"prompts":["color dax"]}}'},
        {"role": "tool", "content": '{"outputs":[{"text":"no"}]}'},
    ]
    chat = example.gemma_messages(messages, tool_schemas(), 2, 10)
    assert [item["role"] for item in chat] == ["user", "assistant", "user"]
    assert "Turn 2 of 10" in chat[-1]["content"]
    assert "check dax" not in chat[0]["content"]
    assert "embed.weight" not in chat[0]["content"]
    assert json.dumps(messages, separators=(",", ":")).count("Turn 2 of 10") == 0


def test_guided_callback_charges_json_repair_to_tokens_and_turn_budget(tmp_path, monkeypatch):
    messages = [{"role": "system", "content": "runtime"}, {"role": "user", "content": "public task"}]
    callback = example.LocalUpdater(None, None, tmp_path, 2, 192)
    inputs = []
    generations = iter([("bad", 11, 1), ('{"name":"submit_patch","arguments":{}}', 21, 7)])

    def generate(chat):
        inputs.append(chat[-1]["content"])
        return next(generations)

    monkeypatch.setattr(callback, "_generate", generate)
    response = callback(messages, tool_schemas())
    assert "No action was executed" in inputs[1]
    assert response["usage"] == {"prompt_tokens": 32, "completion_tokens": 8}
    assert (callback.turns, callback.prompt_tokens, callback.completion_tokens) == (2, 32, 8)
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "submit_patch"
    with pytest.raises(ValueError, match="generation limit"):
        callback(messages, tool_schemas())
    records = [json.loads(s) for s in (tmp_path / "updater-generations.jsonl").read_text().splitlines()]
    assert [r["repair"] for r in records] == [False, True]


def test_schema_mode_keeps_original_no_repair_behavior(tmp_path, monkeypatch):
    messages = [{"role": "system", "content": "runtime"}, {"role": "user", "content": "public task"}]
    callback = example.LocalUpdater(None, None, tmp_path, 10, 192, interface="schema")
    monkeypatch.setattr(callback, "_generate", lambda chat: ("bad", 20, 1))
    with pytest.raises(ValueError, match="unparseable model action"):
        callback(messages, tool_schemas())
    assert callback.turns == 1 and callback.completion_tokens == 1
