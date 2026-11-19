import json

import pytest

from iwi_harness.cli import main
from iwi_harness.targets import tiny_target
from iwi_harness.tokenization import tokenize_target


def test_token_lookup_pages_native_positions_and_keeps_literal_text_separate():
    target = tiny_target()
    result = tokenize_target(target, prompt="color red", texts=["red", "red blue"], limit=2)
    assert [r["input_token_id"] for r in result["rendered_prompt"]["tokens"]] == target.encode("color red")[:2]
    assert result["rendered_prompt"]["next_offset"] == 2
    assert result["literal_texts"] == [{"text": "red", "token_ids": [3]}, {"text": "red blue", "token_ids": [3, 4]}]
    tail = tokenize_target(target, prompt="color red", offset=2)
    assert tail["rendered_prompt"]["tokens"] == [{"position": 2, "input_token_id": 3, "piece": "red"}]
    with pytest.raises(ValueError, match="prompt or"):
        tokenize_target(target)


def test_edit_cli_uses_selected_target_and_starts_without_dictionary(tmp_path, monkeypatch, capsys):
    from iwi_harness import targets, updater

    loaded = []
    def load(model_id, **kwargs):
        loaded.append((model_id, kwargs))
        return tiny_target()

    def provider(*_):
        turns = iter(["discover_target", "submit_patch"])
        def respond(messages, tools):
            assert "fit_sae" in {t["function"]["name"] for t in tools}
            name = next(turns)
            if name == "submit_patch":
                state = json.loads(messages[-1]["content"])
                assert state["dictionaries"] == [] and state["active_dictionary"] is None
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": name, "type": "function", "function": {"name": name, "arguments": "{}"}}]}}]}
        return respond

    monkeypatch.setattr(targets, "load_hf_target", load)
    monkeypatch.setattr(updater, "chat_completions", provider)
    revision = "a" * 40
    main(["edit", "--target", "example/decoder", "--target-revision", revision, "--target-blocks", "model.stack",
          "--device", "cpu", "--local-files-only", "--base-url", "http://example.invalid", "--updater-model", "editor",
          "--max-turns", "2", "--out", str(tmp_path), "--objective", "Discover and submit"])
    assert loaded == [("example/decoder", {"revision": revision, "block_path": "model.stack",
                                           "device": "cpu", "local_files_only": True})]
    assert json.loads(capsys.readouterr().out)["changed_parameters"] == 0


def test_edit_cli_applies_sae_guided_policy_with_requested_turn_limit(tmp_path, monkeypatch):
    from iwi_harness import updater

    objectives = []
    def submit(session, objective, *args, **kwargs):
        objectives.append(objective)
        return session.submit()
    monkeypatch.setattr(updater, "run_updater", submit)
    cmd = ["edit", "--target", "tiny", "--base-url", "http://example.invalid", "--updater-model", "editor",
           "--objective", "Improve performance.", "--out", str(tmp_path), "--max-turns", "5", "--sae-guided"]
    with pytest.raises(SystemExit):
        main(cmd + ["--no-interpretability"])
    main(cmd)
    assert objectives == [updater.sae_guided_objective("Improve performance.", 5)]
    assert all(key in objectives[0] for key in ("5 turns", "inspect_target", "scale 0", "submit_patch"))
