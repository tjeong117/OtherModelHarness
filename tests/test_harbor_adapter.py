import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from iwi_harness.harbor_adapter import make_task


def _environment(task, work):
    return {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(task / "environment"),
        "IWI_ARTIFACT_DIR": str(work / "artifacts"),
        "IWI_AGENT_LOG_DIR": str(work / "logs"),
        "IWI_VERIFIER_DIR": str(work / "verifier"),
        "IWI_DEMO_DIR": str(work / "demo"),
    }


def _grade(task, work, env):
    subprocess.run([task / "tests/test.sh"], env=env, cwd=work, check=True)
    return json.loads((work / "verifier/reward.json").read_text())["reward"]


def test_materialized_contexts(tmp_path):
    destination = tmp_path / "task"
    destination.mkdir()
    assert make_task(destination) == destination
    config = tomllib.loads((destination / "task.toml").read_text())
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["verifier"]["environment"]["network_mode"] == "no-network"
    public = destination / "environment/iwi_harness"
    assert {p.stem for p in public.glob("*.py")} == {
        "__init__", "targets", "probes", "causal", "dictionaries", "tokenization", "weights", "tools", "session",
        "preview", "experiments", "public_cases",
    }
    for area in ["environment", "tests"]:
        assert (destination / area / "Dockerfile").read_text().startswith("FROM ")
    assert (destination / "tests/iwi_harness/demo.py").is_file()
    assert (destination / "solution/iwi_harness/demo.py").is_file()
    with pytest.raises(FileExistsError):
        make_task(destination)


def test_public_editor_and_oracle_match_in_independent_verifier(tmp_path):
    task = make_task(tmp_path / "task")
    env = _environment(task, tmp_path)
    command = task / "environment/edit-target"
    actions = [
        {"name": "inspect_target", "arguments": {"prompts": ["color dax"],
                                                  "probe": {"layer": 0, "kind": "residual"}}},
        {"name": "write_weights", "arguments": {
            "tensor": "embed.weight", "selection": [5], "operation": "copy",
            "source": {"tensor": "embed.weight", "selection": [3]},
        }},
        {"name": "submit_patch", "arguments": {}},
    ]
    completed = subprocess.run(
        [command], input="\n".join(map(json.dumps, actions)) + "\n", cwd=tmp_path, env=env,
        text=True, capture_output=True, check=True,
    )
    replies = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(replies) == 3 and all("error" not in reply for reply in replies)
    assert list((tmp_path / "artifacts").iterdir()) == [tmp_path / "artifacts/patch.safetensors"]
    assert _grade(task, tmp_path, env) == 1.0
    editor_result = json.loads((tmp_path / "verifier/details.json").read_text())
    subprocess.run([task / "solution/solve.sh"], cwd=tmp_path, env=env,
                   stdout=subprocess.DEVNULL, check=True)
    assert _grade(task, tmp_path, env) == 1.0
    oracle_result = json.loads((tmp_path / "verifier/details.json").read_text())
    assert oracle_result["patch_sha256"] == editor_result["patch_sha256"] == replies[-1]["sha256"]


def test_verifier_ignores_untrusted_files_and_rejects_invalid_patch(tmp_path):
    task = make_task(tmp_path / "task")
    env = _environment(task, tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "reward.json").write_text('{"reward":1}')
    (artifacts / "grade.py").write_text('raise AssertionError("submitted code executed")')
    patch = artifacts / "patch.safetensors"
    assert _grade(task, tmp_path, env) == 0.0
    patch.write_bytes(b"not a safetensors artifact")
    assert _grade(task, tmp_path, env) == 0.0
    patch.unlink()
    patch.symlink_to(artifacts / "reward.json")
    assert _grade(task, tmp_path, env) == 0.0
