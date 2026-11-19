"""Reproduce a paired tool-payload measurement on a generated BF16 embedding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from safetensors import safe_open
import torch

from .session import EditingSession
from .weights import WeightStore

_WIDTH = 256
_SEED = 20261118
_WEIGHT = "embed_tokens.weight"


class _Fixture:
    """Two embedding rows, one seeded source and one zero destination, on CPU."""

    model_id = "iwi/generated-embedding"
    revision = "bf16-uniform-v1"
    blocks = ()

    def __init__(self):
        rng = Random(_SEED)
        source = torch.tensor([rng.uniform(-0.05, 0.05) for _ in range(_WIDTH)],
                              dtype=torch.bfloat16, device="cpu")
        self.model = torch.nn.Embedding.from_pretrained(torch.stack([source, torch.zeros_like(source)]))

    def weights(self):
        return {_WEIGHT: self.model.weight}


def _patch_content_digest(path):
    # Safetensors permits variations in metadata key order and header padding.
    # Hash canonical metadata and the names, types, shapes and bytes of the payload.
    entries = []
    with safe_open(path, framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata()
        for key in ("base_tensors", "sparse"):
            metadata[key] = json.loads(metadata[key])
        for name in sorted(artifact.keys()):
            value = artifact.get_tensor(name)
            entries.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                            "sha256": hashlib.sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()})
    content = json.dumps([metadata, entries], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def _run_arm(directory, encoding, inline):
    target = _Fixture()
    session = EditingSession(target, artifact_dir=directory,
                             token_counter=lambda text: len(encoding.encode_ordinary(text)))
    # A production handle begins with a random 4-byte hex namespace. A fixed
    # same-format namespace isolates this run from tokenization noise.
    session.store._prefix = hashlib.sha256(b"iwi-payload-fixture-v1").hexdigest()[:8]
    read = {"tensor": _WEIGHT, "selection": [0]}
    write = {"tensor": _WEIGHT, "selection": [1]}
    if inline:
        read_result = session.call("read_weights", {**read, "mode": "values", "limit": _WIDTH})
        session.call("write_weights", {**write, "operation": "replace", "value": {"values": read_result["values"]}})
    else:
        read_result = session.call("read_weights", {**read, "mode": "ref"})
        session.call("write_weights", {**write, "operation": "copy", "source": {"handle": read_result["handle"]}})
    receipt = session.submit()
    patch = directory / "patch.safetensors"
    fresh = _Fixture()
    expected_row = fresh.model.weight[0].detach().clone()
    WeightStore(fresh).apply(patch, expected_sha256=receipt["sha256"])
    weight = fresh.model.weight.detach()
    if not torch.equal(weight, torch.stack([expected_row, expected_row])) or not torch.equal(weight, target.model.weight):
        raise RuntimeError("Replayed weights differ from the requested row copy")

    measurements = {}
    for line in (directory / "trajectory.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["tool"] in ("read_weights", "write_weights"):
            measurements[event["tool"]] = {k: event[k] for k in ("request", "response")}
    total = {unit: sum(size[unit] for call in measurements.values() for size in call.values())
             for unit in ("tokens", "utf8_bytes")}
    return {"calls": measurements, "total": total, "patch_sha256": receipt["sha256"],
            "patch_content_sha256": _patch_content_digest(patch)}, weight


def compare_payloads(encoding: str = "o200k_base") -> dict:
    """Execute two equivalent edits and measure the session's serialized arguments and results."""
    try:
        import tiktoken
    except ModuleNotFoundError as exc:
        if exc.name != "tiktoken":
            raise
        raise RuntimeError("Install iwi-harness[tokens] to run the payload comparison") from exc
    tokenizer = tiktoken.get_encoding(encoding)
    with TemporaryDirectory(prefix="iwi-costs-") as scratch:
        inline, inline_weight = _run_arm(Path(scratch) / "inline", tokenizer, inline=True)
        handle, handle_weight = _run_arm(Path(scratch) / "handle", tokenizer, inline=False)
    values_match = torch.equal(inline_weight, handle_weight)
    patches_match = inline["patch_content_sha256"] == handle["patch_content_sha256"]
    files_match = inline["patch_sha256"] == handle["patch_sha256"]
    if not values_match or not patches_match or not files_match:
        raise RuntimeError("The paired arms produced different numerical patches")
    return {
        "encoding": tokenizer.name, "tiktoken_version": tiktoken.__version__,
        "fixture": {"model_id": _Fixture.model_id, "revision": _Fixture.revision, "seed": _SEED,
                    "dtype": "bfloat16", "row_width": _WIDTH, "source_row": 0, "destination_row": 1},
        "inline": inline, "handle": handle,
        "savings": {unit: inline["total"][unit] - handle["total"][unit] for unit in inline["total"]},
        "parity": {"exact_replayed_weights": values_match, "exact_patch_content": patches_match, "exact_patch_bytes": files_match},
        "measurement": {
            "included": "Compact UTF-8 JSON arguments and results for read_weights and write_weights. Each payload counts separately.",
            "excluded": "Provider framing, tool names and schemas, repeated history, reasoning, token caching, submit and grading.",
            "handle": "An actual session handle uses a fixed fixture-only 8-hex-digit prefix for repeatability.",
            "interpretation": "One scripted row copy measures payload cost. Updater behavior and provider billing require separate trials.",
        },
    }
