import json
from pathlib import Path
import subprocess
import sys


CLI = Path(__file__).resolve().parents[1] / "examples" / "compare_astra_arms.py"
FAMILY = "alias_acquisition"
COMMON = ["read_weights", "write_weights", "run_target", "submit_patch"]
FULL = COMMON + ["inspect_target", "contrast_target", "rank_attributions", "feature_examples", "patch_activations"]


def case(root, seed, tools, *, status="scored", base=1, retained=1):
    directory = root / f"alias-{seed}"
    directory.mkdir(parents=True)
    info = {"family": FAMILY, "seed": seed, "requested_model": "astra", "reasoning": "medium", "max_turns": 12,
            "max_output_tokens": 4096, "budget_cap_usd": 15, "iwibench_revision": "pin",
            "session": {"limits": {"max_forwards": 1024}},
            "tools_offered": tools, "status": "submitted" if status == "scored" else status,
            "provider": [{"model": "astra", "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                         "prompt_tokens_details": {"cached_tokens": 6}}}], "cost_upper_bound_usd": 0.025}
    (directory / "editor-result.json").write_text(json.dumps(info))
    (directory / "public-input.json").write_text(json.dumps({"seed": seed}))
    gid = f"{FAMILY}/0.2.2/gemma-3-4b-it/development/{seed}"
    grade = {"status": status, "reward": 1.0 if status == "scored" else 0.0}
    if status == "scored":
        grade.update(family=FAMILY, episode_id=gid, model_id="gemma", revision="rev", patch_sha256="digest",
                     metrics={"episode_id": gid, "reward": 1.0, "calibration": {"eligible": True},
                              "control_retention_denominator": 5, "controls_retained": 5, "groups": {
                                  "targets": {"count": 4, "base_successes": base,
                                              "edited_successes": 2, "retained_successes": retained}}})
    (directory / "grade.json").write_text(json.dumps(grade))


def test_matched_usage_and_grades_include_no_submissions_and_surface_missing_controls(tmp_path):
    for seed in (1, 3, 11):
        case(tmp_path / "full", seed, FULL)
    case(tmp_path / "control", 1, COMMON, retained=0)
    case(tmp_path / "control", 3, COMMON, status="no_submission")
    output = tmp_path / "result.json"
    cmd = [sys.executable, CLI, "--full", tmp_path / "full", "--control", tmp_path / "control",
           "--control-label", "compact", "--out", output]
    subprocess.run(cmd, check=True)
    result = json.loads(output.read_text())["per_family"][FAMILY]
    assert result["paired_rewards"] == 2
    assert result["paired_fresh_grades"] == 1
    assert result["pairs_with_missing_results"] == 1
    assert result["mean_reward_difference_full_minus_control"] == 0.5
    assert result["control"]["counts_on_paired_fresh_grades"]["newly_correct_targets"] == 2
    assert result["control"]["counts_on_paired_fresh_grades"]["previously_correct_targets_lost"] == 1
    assert result["control"]["provider_usage_all_scheduled"]["total_tokens"] == {"sum_reported": 26, "episodes_reporting": 2}
    assert result["control"]["provider_usage_all_scheduled"]["uncached_input_tokens"]["sum_reported"] == 8
    assert result["control"]["provider_usage_all_scheduled"]["first_response_input_tokens"]["sum_reported"] == 20
    assert result["total_token_ratio_full_to_control"] is None

    # Numerical grades on a different pristine population must be visible yet excluded from the paired estimate.
    file = tmp_path / "control/alias-1/grade.json"
    grade = json.loads(file.read_text())
    grade["metrics"]["groups"]["targets"]["base_successes"] = 2
    file.write_text(json.dumps(grade))
    subprocess.run(cmd, check=True)
    report = json.loads(output.read_text())
    assert report["pairs"][0]["mismatches"] == ["pristine group or calibration counts differ"]
    assert report["per_family"][FAMILY]["paired_rewards"] == 1


def test_native_recovery_verifies_trace_usage_patch_and_discloses_missing_model_echo(tmp_path):
    import hashlib

    case(tmp_path / "full", 1, FULL)
    case(tmp_path / "control", 1, ["read_weights", "write_weights", "run_target", "inspect_target"])
    directory = tmp_path / "control/alias-1"
    editor_file = directory / "editor-result.json"
    editor = json.loads(editor_file.read_text())
    editor["protocol"] = "native-iwibench-v1"
    editor["recovery"] = {"response_model_echo_available": False, "usage_source": "original_trajectory",
                          "sha256_verified": True}
    usage = editor["provider"][0]["usage"]
    usage["cost_upper_bound_usd"] = editor.pop("cost_upper_bound_usd")
    editor.pop("provider")
    (directory / "trajectory.jsonl").write_text(json.dumps({"type": "updater", "usage": usage}) + "\n")
    patch = directory / "patch.safetensors"
    patch.write_bytes(b"test patch")
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    editor["receipt"] = {"sha256": digest}
    editor_file.write_text(json.dumps(editor))
    grade_file = directory / "grade.json"
    grade = json.loads(grade_file.read_text())
    grade["patch_sha256"] = digest
    grade_file.write_text(json.dumps(grade))

    output = tmp_path / "result.json"
    cmd = [sys.executable, CLI, "--full", tmp_path / "full", "--control", tmp_path / "control",
           "--control-label", "native", "--out", output]
    subprocess.run(cmd, check=True)
    pair = json.loads(output.read_text())["pairs"][0]
    assert pair["both_freshly_scored"] and pair["control"]["warnings"]
    provider = pair["control"]["provider"]
    assert provider["response_models"] == []
    assert provider["first_response_input_tokens"] == 10 and provider["cost_upper_bound_usd"] == 0.025
    editor["provider"] = [{"model": None, "usage": {**usage, "prompt_tokens": 99}}]
    editor_file.write_text(json.dumps(editor))
    subprocess.run(cmd, check=True)
    pair = json.loads(output.read_text())["pairs"][0]
    assert pair["both_freshly_scored"] is False
    assert "control: recovered provider usage differs from original trajectory" in pair["mismatches"]
