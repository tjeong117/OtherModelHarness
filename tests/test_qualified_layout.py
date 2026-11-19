import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION, tiny_target


def test_qualified_layout_pairs_identical_task_schemas_and_practice(monkeypatch, tmp_path):
    pytest.importorskip("iwibench")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import astra_qualified_layout as trial

    row = dict(id="fixture", seed=7, family="alias_acquisition", variant_index=3,
               parameter_sha256="params", episode_sha256="episode", source_commit="code", verifiers_revision="v",
               calibration_status="eligible", catalog_version="baseline-qualified-v1",
               model_id=GEMMA_ID, checkpoint_revision=GEMMA_REVISION, objective="Learn the public alias.",
               parameters_json='{"alias":"dax"}', public_practice_json='[{"prompt":"color dax","answers":["red"]}]',
               unrelated_row_metadata="hidden-sentinel")
    catalog = tmp_path / "test.jsonl"
    catalog.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(trial, "load_gemma_target", lambda **kwargs: tiny_target())

    class Editor:
        def __init__(self, *args, **kwargs):
            self.records = []

        def __call__(self, messages, tools):
            assert "hidden-sentinel" not in json.dumps(messages)
            assert "SAE" in messages[1]["content"]
            usage = dict(prompt_tokens=10, completion_tokens=1, total_tokens=11)
            self.records.append(dict(model="fixture", usage=usage))
            message = {"role": "assistant", "tool_calls": [{"type": "function", "id": "sub",
                       "function": {"name": "submit_patch", "arguments": "{}"}}]}
            return {"choices": [{"message": message}], "usage": usage}

    monkeypatch.setattr(trial, "AstraResponses", Editor)
    results = {}
    for mode in ("paged", "eager", "grouped"):
        out = tmp_path / mode
        out.mkdir()
        args = SimpleNamespace(catalog=catalog, task_id="fixture", out=out, sae_guided=True,
                               max_turns=3, max_output_tokens=300, budget_usd=1, opening_manifest=mode)
        results[mode] = trial.edit(args, "unused")
        assert results[mode]["status"] == "submitted"
        assert results[mode]["host_public_practice_checks"] == 1
        assert results[mode]["provider"][0]["usage"]["total_tokens"] == 11
        assert (out / "receipt.json").is_file()
    for field in ("objective_sha256", "session", "episode_sha256", "target_usage_including_host"):
        assert results["paged"][field] == results["eager"][field] == results["grouped"][field]
    for filename in ("public-input.json", "tools.json"):
        assert len({(tmp_path / mode / filename).read_bytes() for mode in results}) == 1
