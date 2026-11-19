import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from iwi_harness.weights import NATIVE_FORMAT, WeightStore


class Target:
    model_id = "fixture/test"
    revision = "base-test-v1"

    def __init__(self):
        self.weight = torch.nn.Parameter(torch.arange(20).reshape(4, 5).to(torch.bfloat16))
        self.bias = torch.nn.Parameter(torch.zeros(5, dtype=torch.bfloat16))

    def weights(self):
        return {"in.weight": self.weight, "out.weight": self.weight, "out.bias": self.bias}


def test_compact_expression_edit_and_handle_lifetime():
    target = Target()
    store = WeightStore(target)
    row = store.read(tensor="in.weight", selection=[0])
    assert "values" not in row and row["shape"] == [5]
    h = row["handle"]
    store.resolve(h).fill_(90)
    assert store.read(handle=h, mode="values", limit=2)["values"] == [0, 1]
    assert store.read(handle=h, mode="values", limit=2, offset=2)["next_offset"] == 4
    expression = {"op": "add", "left": {"handle": h},
                  "right": {"op": "scale", "factor": -1, "left": {"values": [1, 2, 3, 4, 5]}}}
    store.write("out.weight", "replace", [2], value=expression)
    assert target.weight[2].tolist() == [-1] * 5
    with pytest.raises(ValueError, match="stale"):
        store.resolve(h)
    with pytest.raises(ValueError):
        store.write("out.weight", "low_rank_add", left={"values": [[1]]}, right={"values": [[1]]})
    assert store.revision == 1


def make_patch(path):
    target = Target()
    store = WeightStore(target)
    store.write("out.weight", "copy", [2], source={"tensor": "in.weight", "selection": [0]})
    # Overlap, restoring one changed value to pristine.
    store.write("in.weight", "fill", [2, 0], value=10)
    store.write("out.weight", "add", [[2, 4], 2], value={"values": [2, 3]})
    store.write("out.bias", "fill", [2], value=2)
    receipt = store.export(path)
    return target, receipt


def test_patch_byte_validation_and_cumulative_replay(tmp_path):
    path = tmp_path / "weights.safetensors"
    edited, saved = make_patch(path)
    assert saved["changed_parameters"] == 6 and saved["format"] == NATIVE_FORMAT
    fresh = Target()
    pristine = fresh.weight.clone()
    fresh_store = WeightStore(fresh)
    with pytest.raises(ValueError, match="SHA-256"):
        fresh_store.apply(path, expected_sha256="0" * 64)
    # Detect a different value on the same named base, outside the edited row.
    fresh.bias.data[0] += 1
    with pytest.raises(ValueError, match="base digest"):
        fresh_store.apply(path)
    assert torch.equal(fresh.weight, pristine) and fresh_store.revision == 0
    fresh.bias.data[0] -= 1
    fresh_store.apply(path, saved["sha256"])
    assert torch.equal(fresh.weight, edited.weight)
    assert torch.equal(fresh.bias, edited.bias)
    with pytest.raises(ValueError, match="pristine"):
        fresh_store.apply(path)
    with safe_open(path, "pt") as artifact:
        manifest = json.loads(artifact.metadata()["sparse"])
        assert len(manifest) == 2
        e = next(entry for entry in manifest if entry["name"] == "in.weight")
        assert artifact.get_tensor(e["indices"]).tolist() == [11, 12, 13, 14, 17]


def test_identity_patch_and_low_rank(tmp_path):
    target = Target()
    store = WeightStore(target)
    orig = target.weight.clone()
    args = {"left": {"values": [[1], [2], [3], [4]]},
            "right": {"values": [[1, 2, 0, -1, 3]]}}
    store.write("in.weight", "low_rank_add", **args)
    assert target.weight[3].tolist() == [19, 24, 17, 14, 31]
    store.write("in.weight", "low_rank_add", factor=-1, **args)
    path = tmp_path / "empty.safetensors"
    assert store.export(path)["changed_parameters"] == 0
    reloaded = Target()
    WeightStore(reloaded).apply(path)
    assert torch.equal(reloaded.weight, orig)


def test_reject_bad_external_sparse_indices(tmp_path):
    import hashlib

    target = Target()
    orig = target.weight.clone()
    path = tmp_path / "bad.safetensors"
    sparse = [{"name": "out.bias", "indices": "__iwi_i", "values": "__iwi_v"}]
    save_file({"in.weight": orig+1, "__iwi_i": torch.tensor([2, 1]),
               "__iwi_v": torch.ones(2, dtype=torch.bfloat16)}, str(path),
              metadata={"format": NATIVE_FORMAT, "model_id": target.model_id,
                        "revision": target.revision, "sparse": json.dumps(sparse)})
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="indices"):
        WeightStore(target).apply(path, expected_sha256=sha)
    assert torch.equal(target.weight, orig)


def test_native_upstream_reader_if_installed(tmp_path):
    native = pytest.importorskip("iwibench.runtime").NativeTarget
    path = tmp_path / "weights.safetensors"
    edited, receipt = make_patch(path)
    target = Target()
    target.parameters = target.weights
    target.invalidate_cache = lambda: None
    native.apply_patch(target, path, receipt["sha256"])
    assert torch.equal(target.weight, edited.weight)
    assert torch.equal(target.bias, edited.bias)


def test_export_is_byte_identical_across_processes(tmp_path):
    import hashlib
    import os
    from pathlib import Path
    import subprocess
    import sys

    source = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([source, os.environ.get("PYTHONPATH", "")])}
    script = (
        "import runpy,sys; "
        "make = runpy.run_path(sys.argv[1])['make_patch']; "
        "print(make(sys.argv[2])[1]['sha256'])"
    )
    path = tmp_path / "stable.safetensors"
    target, receipt = make_patch(path)
    for _ in range(2):
        other = tmp_path / "other.safetensors"
        result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), str(other)],
                                check=True, capture_output=True, text=True, env=env)
        assert result.stdout.strip() == receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert other.read_bytes() == path.read_bytes()
    replay = Target()
    WeightStore(replay).apply(path, expected_sha256=receipt["sha256"])
    assert torch.equal(target.weight, replay.weight)
