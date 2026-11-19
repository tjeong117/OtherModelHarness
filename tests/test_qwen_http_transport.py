import json
from pathlib import Path

import pytest


@pytest.fixture
def qwen_http(monkeypatch):
    pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import qwen_http_trial

    return qwen_http_trial


def test_qwen_renders_pi_text_parts_and_native_tool_history(qwen_http):
    class EchoTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages

    request = {
        "tools": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "copy "}, {"type": "text", "text": "row"}]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "test_behavior", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "ok"},
        ],
    }
    messages = qwen_http.render(EchoTokenizer(), request)
    assert messages[0]["content"] == "copy row"
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {}
    assert messages[2]["content"] == "ok"
    assert isinstance(request["messages"][0]["content"], list)
    request["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "unavailable"}}]
    with pytest.raises(ValueError, match="text content only"):
        qwen_http.render(EchoTokenizer(), request)


def test_qwen_emits_the_same_sampled_calls_and_usage_over_both_transports(qwen_http):
    message = qwen_http.parse_native('<tool_call>{"name":"test_behavior","arguments":{}}</tool_call>', 1)
    usage = {"prompt_tokens": 17, "completion_tokens": 8, "total_tokens": 25}
    mime, data = qwen_http.response_bytes(message, "tool_calls", usage, 1, False)
    assert mime == "application/json"
    response = json.loads(data)
    assert response["choices"][0]["message"] == message and response["usage"] == usage
    mime, data = qwen_http.response_bytes(message, "tool_calls", usage, 1, True)
    assert mime == "text/event-stream"
    lines = data.decode().strip().split("\n\n")
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    expected = dict(index=0, **message["tool_calls"][0])
    assert chunks[0]["choices"][0]["delta"]["tool_calls"] == [expected]
    assert chunks[1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[2]["usage"] == usage
