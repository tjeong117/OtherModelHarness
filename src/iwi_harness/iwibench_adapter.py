"""Trusted task and grading adapter for IWIBench ff1bbcc, development episodes only.

Install the pinned checkout or put its src directory on PYTHONPATH. The controller
can hand public_task()'s result to an editor. Keep grade_patch() in the verifier.
The Gemma checkpoint and tokenizer must be provisioned before starting a worker.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from safetensors import safe_open

from .targets import GEMMA_ID, GEMMA_REVISION
from .weights import NATIVE_FORMAT

IWIBENCH_REVISION = "ff1bbcc78f6090e17bf4d50afeb7dc51437b3b53"
_SUPPORTED = {"resistance_to_false_suggestions", "alias_acquisition"}


def _episode(family, seed, split):
    if family not in _SUPPORTED:
        raise ValueError(f"Supported IWIBench families are {sorted(_SUPPORTED)}")
    if split != "development":
        raise ValueError("The pinned IWIBench release only admits development trials")
    try:
        import iwibench
        from iwibench.families import episode
    except ModuleNotFoundError as exc:
        if exc.name != "iwibench":
            raise
        raise RuntimeError(f"Install IWIBench at {IWIBENCH_REVISION} or add its src to PYTHONPATH") from exc
    if iwibench.__version__ != "0.2.6":
        raise RuntimeError(f"This adapter requires IWIBench 0.2.6 at {IWIBENCH_REVISION}")
    ep = episode(family, seed, split, model_id=GEMMA_ID)
    if ep["family_version"] != "0.2.2" or ep["checkpoint_revision"] != GEMMA_REVISION:
        raise RuntimeError("IWIBench family or Gemma checkpoint differs from the adapter pin")
    return ep


def _cached_tokenizer():
    """Return no tokenizer when a public resistance task runs without Gemma."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        checkpoint = snapshot_download(GEMMA_ID, revision=GEMMA_REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        return None
    return AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)


def _alias_tokens(ep, tokenizer):
    from iwibench.runtime import NativeTarget, alias_tokenization
    if tokenizer is None:
        raise RuntimeError("Alias acquisition requires the locally cached pinned Gemma tokenizer")
    # The native function compares tokenizer offsets to NativeTarget._prompt_ids.
    # Both methods use only the tokenizer and do not require loading the weights.
    target = NativeTarget.__new__(NativeTarget)
    target.tokenizer = tokenizer
    return alias_tokenization(ep, target)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (dict, list)):
        for child in value.values() if isinstance(value, dict) else value:
            yield from _strings(child)


def public_task(family: str, seed: int, split: str = "development") -> dict:
    """Construct the native editor-facing objective, practice and public token map."""
    ep = _episode(family, seed, split)
    from iwibench.families import public_episode
    packet = public_episode(ep)
    tokenizer = _cached_tokenizer()
    tokens = {}
    if tokenizer is not None:
        for word in dict.fromkeys(_strings(packet["parameters"])):
            for spelling in (word, " " + word):
                tokens[spelling] = tokenizer.encode(spelling, add_special_tokens=False)
    if family == "alias_acquisition":
        aliases = _alias_tokens(ep, tokenizer)
        # The full occurrence records include private item locations.
        tokens["alias_context"] = {
            word: {key: record[key] for key in ("token_ids", "standalone_token_ids", "leading_space_token_ids")}
            for word, record in aliases.items()
        }
    return {key: packet[key] for key in (
        "id", "family", "family_version", "model_id", "checkpoint_revision", "objective"
    )} | {"practice": packet["public_practice"], "token_ids": tokens}


def _run_worker(request):
    completed = subprocess.run(
        [sys.executable, "-m", "iwibench.grading_worker"],
        input=json.dumps(request), text=True, capture_output=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    if completed.returncode:
        raise RuntimeError(f"Fresh IWIBench worker exited {completed.returncode}: {completed.stderr[-3000:]}")
    return json.loads(completed.stdout)


def _completions(items, request):
    result = _run_worker(request)
    if (not isinstance(result, dict) or result.get("model_id") != request["model_id"]
            or result.get("revision") != request["revision"]):
        raise ValueError("IWIBench worker returned the wrong checkpoint")
    metadata = result["patch_metadata"]
    if request["patch_path"]:
        if (not isinstance(metadata, dict) or metadata.get("format") != NATIVE_FORMAT
                or metadata.get("model_id") != request["model_id"] or metadata.get("revision") != request["revision"]):
            raise ValueError("Edited IWIBench worker did not replay the expected native patch")
    elif metadata is not None:
        raise ValueError("Pristine IWIBench worker unexpectedly applied a patch")
    generations = result["generations"]
    if not isinstance(generations, list) or len(generations) != len(items):
        raise ValueError("Incomplete IWIBench worker output")
    outputs = {}
    for item, row in zip(items, generations, strict=True):
        if (not isinstance(row, dict) or not isinstance(row.get("text"), str)
                or any(type(row.get(k)) is not int or row[k] < 0 for k in ("input_tokens", "output_tokens"))
                or row["output_tokens"] > request["max_new_tokens"]):
            raise ValueError("Malformed IWIBench worker completion")
        outputs[item["id"]] = row["text"]
    return outputs


def grade_patch(patch_path: str | Path, *, family: str, seed: int, split: str = "development",
                device: str = "cuda", expected_sha256: str | None = None) -> dict:
    """Run clean and patched native workers against paired private prompts; return trusted metrics."""
    ep = _episode(family, seed, split)
    # The editor may replace the submitted path while the baseline worker runs.
    # Hash, validate and replay the same verifier-owned copy throughout the trial.
    with TemporaryDirectory(prefix="iwi-verify-") as directory:
        snapshot = Path(directory) / "patch.safetensors"
        with Path(patch_path).open("rb") as source, snapshot.open("xb") as dest:
            shutil.copyfileobj(source, dest)
        return _grade_snapshot(ep, snapshot, device, expected_sha256)


def _grade_snapshot(ep, path, device, expected_sha256):
    from iwibench.config import Limits
    from iwibench.families import PRIVATE_GROUPS
    from iwibench.scoring import CalibrationError, score_episode
    family = ep["family"]
    with path.open("rb") as artifact:
        digest = hashlib.file_digest(artifact, "sha256").hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Submitted patch SHA-256 mismatch")
    with safe_open(path, framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata() or {}
    if (metadata.get("format") != NATIVE_FORMAT or metadata.get("model_id") != GEMMA_ID
            or metadata.get("revision") != GEMMA_REVISION):
        raise ValueError("A native IWIBench patch for the pinned Gemma checkpoint is required")
    tokenization = _alias_tokens(ep, _cached_tokenizer()) if family == "alias_acquisition" else {}
    items = [item for group in PRIVATE_GROUPS for item in ep[group]]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("IWIBench episode contains duplicate item identifiers")
    base_request = {
        "model_id": GEMMA_ID, "revision": GEMMA_REVISION, "device": device,
        "prompts": [item["prompt"] for item in items], "patch_path": None,
        "seeds": [ep["test_seed"] + i for i in range(len(items))],
        "max_new_tokens": Limits().output_tokens,
    }
    record = {"episode_id": ep["id"], "family": family, "model_id": GEMMA_ID,
              "revision": GEMMA_REVISION, "patch_sha256": digest}
    base = _completions(items, base_request)
    try:
        # Check every native admission condition (including control denominator)
        # before spending a second model load on an unscorable episode.
        score_episode(ep, base, base, tokenization=tokenization)
    except CalibrationError as exc:
        return {**record, "status": "ineligible", "reasons": [str(exc)], "reward": None, "metrics": None}
    edited_request = {**base_request, "patch_path": str(path), "patch_sha256": digest}
    edited = _completions(items, edited_request)
    metrics = score_episode(ep, base, edited, tokenization=tokenization)
    return {**record, "status": "scored", "reward": metrics["reward"], "metrics": metrics}
