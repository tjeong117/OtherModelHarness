from pathlib import Path

import pytest


def test_token_usage_includes_cached_inputs_and_incomplete_responses(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import summarize_qualified_live as summary

    def record(inputs, cached, output, status):
        return {"status": status,"usage": {"prompt_tokens": inputs, "completion_tokens": output,
                "total_tokens": inputs+output,"prompt_tokens_details": {"cached_tokens": cached},
                "cost_upper_bound_usd": .05}}

    receipts=[record(100,80,10,"completed"), record(120,100,9,"incomplete")]
    result=summary.usage(receipts)
    assert (result["calls"],result["total_tokens"],result["uncached_input_tokens"]) == (2,239,40)
    assert result["reported_cost_upper_bound_usd"] == .1
    receipts[-1]["usage"]["total_tokens"] = 1000
    with pytest.raises(ValueError,match="inconsistent usage"):
        summary.usage(receipts)


def test_missing_planned_arms_are_visible_without_a_paired_delta(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import json
    import summarize_qualified_live as summary

    design=json.loads((summary.ROOT/"evidence/heldout-token-audit/live-design.json").read_text())
    grouped=json.loads((summary.ROOT/"evidence/heldout-token-audit/grouped-live-design.json").read_text())
    result=summary.summarize(tmp_path,design,grouped)
    assert not result["complete"] and not result["grouped_followup"]["complete"]
    assert len(result["tasks"]) == 5
    pair=result["comparisons"]["eager"]
    assert pair["usage_pairs"] == pair["quality_pairs"] == 0 and pair["all_turns"]["saving_fraction"]["total_tokens"] is None
    assert result["tasks"][0]["paged"]["editor_status"] == "missing_editor_result"


def test_completion_includes_a_configured_grouped_followup(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import json
    import summarize_qualified_live as summary

    design = json.loads((summary.ROOT / "evidence/heldout-token-audit/live-design.json").read_text())
    pending_family = design["tasks"][-1]["family"]

    def fake_arm(root, task, arm, protocol):
        terminal = arm != "grouped" or task["family"] != pending_family
        return {"terminal": terminal, "editor_status": "no_submission" if terminal else "missing_editor_result"}, None, []

    monkeypatch.setattr(summary, "audit_arm", fake_arm)
    primary = summary.summarize(tmp_path, design)
    assert primary["primary_complete"] and primary["complete"]
    grouped = summary.summarize(tmp_path, design, {"study": "compact full-information control"})
    assert grouped["primary_complete"] and not grouped["complete"]
    assert summary.pending_arms(grouped) == ["grouped 4/5 terminal"]
