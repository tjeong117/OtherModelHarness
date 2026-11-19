"""Read the single numeric submission in a clean verifier container."""
import json
import math
import os
from pathlib import Path
import stat
import subprocess


def main():
    artifacts = Path(os.environ.get("IWI_ARTIFACT_DIR", "/logs/artifacts"))
    logs = Path(os.environ.get("IWI_VERIFIER_DIR", "/logs/verifier"))
    logs.mkdir(parents=True, exist_ok=True)
    patch = artifacts / "patch.safetensors"
    try:
        info = patch.lstat()
    except FileNotFoundError:
        info = None
    if info is None or not stat.S_ISREG(info.st_mode) or not (8 <= info.st_size <= 1_048_576):
        details = {"reward": 0.0, "reason": "Expected a regular numeric patch of at most 1 MiB."}
    else:
        process = subprocess.run(
            ["iwi-harness", "verify-tiny", "--patch", str(patch)],
            check=True, stdout=subprocess.PIPE, text=True,
        )
        details = json.loads(process.stdout)
    reward = details["reward"]
    if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("verify-tiny returned an invalid numeric reward")
    (logs / "details.json").write_text(json.dumps(details, allow_nan=False) + "\n")
    (logs / "reward.json").write_text(json.dumps({"reward": reward}) + "\n")


if __name__ == "__main__":
    main()
