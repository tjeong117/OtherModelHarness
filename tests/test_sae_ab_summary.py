import json
from pathlib import Path
import shutil
import sys


def test_summary_requires_identical_shared_schemas_and_actual_task(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from summarize_sae_ab import CASES, summarize
    from iwi_harness.tools import tool_schemas

    def save(path, value):
        path.write_text(json.dumps(value) + "\n")

    for case, family, seed in CASES:
        for arm in ("fit", "no-fit"):
            path = tmp_path / f"{case}-{seed}" / arm
            path.mkdir(parents=True)
            schemas = [s for s in tool_schemas(detailed_writes=True, adaptive_interpretability=True)
                       if arm == "fit" or s["function"]["name"] != "fit_sae"]
            caps = {"max_forwards": 1024, "max_fitted_dictionaries": 3 if arm == "fit" else 0,
                    "max_sae_fit_steps": 256 if arm == "fit" else 0}
            schema_size = len(json.dumps(schemas, separators=(",", ":"), ensure_ascii=False).encode())
            settings = dict(case=case, family=family, seed=seed, arm=arm, protocol="runtime-sae-access-v1",
                            requested_model="astra", reasoning="medium", max_turns=6, max_output_tokens=2048,
                            budget_cap_usd=1, iwibench_revision="pin",
                            session=dict(sae=None, limits=caps, tool_schema={"utf8_bytes": schema_size}),
                            tools_offered=[s["function"]["name"] for s in schemas])
            packet = {"objective": "Edit the target."}
            usage = dict(prompt_tokens=10, completion_tokens=3, total_tokens=13, cost_upper_bound_usd=0.1)
            event = dict(type="updater", task=json.dumps(packet), usage=usage, response={"tool_calls": []})
            save(path / "trajectory.jsonl", event)
            save(path / "protocol.json", settings)
            save(path / "public-input.json", packet)
            save(path / "tools.json", schemas)
            save(path / "editor-result.json", dict(**settings, status="no_submission", successful_model_tool_counts={},
                 provider=[dict(model="astra", usage=usage, status="completed")], cost_upper_bound_usd=0.1))
            save(path / "grade.json", {"status": "no_submission", "reward": 0})
    initial = summarize(tmp_path)
    assert initial["overall"]["paired_rewards"] == 2
    assert initial["overall"]["paired_fresh_grades"] == 0
    assert initial["overall"]["fit"]["provider_usage_all_scheduled"]["total_tokens"]["sum_reported"] == 26
    follow = tmp_path / "instructed-followup/suggestion-2"
    shutil.copytree(tmp_path / "suggestion-2/fit", follow)
    plan = {"case": "suggestion", "seed": 2, "instruction": "Fit an SAE.", "scope": "Separate follow-up."}
    for filename in ("protocol.json", "editor-result.json"):
        obj = json.loads((follow / filename).read_text())
        obj["protocol"] = "runtime-sae-instructed-v1"
        save(follow / filename, obj)
    follow_packet = {"objective": "Edit the target. Fit an SAE."}
    save(follow / "public-input.json", follow_packet)
    follow_trace = json.loads((follow / "trajectory.jsonl").read_text())
    follow_trace["task"] = json.dumps(follow_packet)
    save(follow / "trajectory.jsonl", follow_trace)
    augmented = summarize(tmp_path, followup_design=plan)
    assert augmented["overall"] == initial["overall"]
    assert augmented["instructed_followup"]["integrity_issues"] == []
    assert augmented["instructed_followup"]["reward_difference_instructed_minus_autonomous_fit"] == 0
    modified = summarize(tmp_path, followup_design=plan | {"instruction": "Different prompt."})
    assert "instructed packet differs beyond the prespecified instruction" in modified["instructed_followup"]["integrity_issues"]
    assert modified["instructed_followup"]["reward_difference_instructed_minus_autonomous_fit"] is None
    path = tmp_path / "alias-1/fit/tools.json"
    changed = json.loads(path.read_text())
    changed[0]["function"]["description"] = changed[0]["function"]["description"][::-1]  # identical byte count
    save(path, changed)
    trace = tmp_path / "suggestion-2/no-fit/trajectory.jsonl"
    event = json.loads(trace.read_text())
    event["task"] = "something else"
    save(trace, event)
    final = summarize(tmp_path)
    assert final["overall"]["paired_rewards"] == 0
    assert "the shared tool schemas or their order differ" in final["pairs"][0]["integrity_issues"]
    assert "actual_user_prompt_sha256 differs" in final["pairs"][1]["integrity_issues"]
    (tmp_path / "alias-1/fit/grade.json").unlink()
    assert summarize(tmp_path)["overall"]["fit"]["grade_status_counts"]["missing_grade"] == 1
