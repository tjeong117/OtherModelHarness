import pytest
import torch

from iwi_harness.preview import WeightPreview
from iwi_harness.weights import WeightStore


class Target:
    model_id = "local/preview-test"
    revision = "1"

    def __init__(self):
        self.weight = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
        self.bias = torch.zeros(3, dtype=torch.bfloat16)

    def weights(self):
        return {"weight": self.weight, "alias": self.weight, "bias": self.bias}


def fill(tensor, selection, value):
    return {"tensor": tensor, "operation": "fill", "selection": selection, "value": value}


def test_preview_isolated_then_exact_commit_on_nonpristine_parent(tmp_path):
    target = Target()
    store = WeightStore(target)
    store.write(**fill("bias", [1], 4))
    previews = WeightPreview(store)
    parent_weights = target.weight.clone()
    parent_row = store.read("weight", [0])["handle"]
    temp_handles = []

    def probe():
        temp_handles.append(store.capture(target.weight[3]))
        return (target.weight.tolist(), target.bias.tolist())

    writes = [
        fill("weight", [0, 0], 9),  # Changes the source used to create parent_row.
        {"tensor": "alias", "operation": "replace", "selection": [3], "value": {"handle": parent_row}},
        fill("alias", [3, 0], 9),  # Cancels one change to the tied destination.
        fill("bias", [1], 0),
    ]  # Cancels the original parent edit.

    trial = previews.preview(writes, probe, expected_revision=1)
    assert trial["before"] == (parent_weights.tolist(), [0, 4, 0])
    assert trial["after"][0][3] == [9, 1, 2]
    assert trial["after"][1] == [0, 0, 0]
    assert trial["changed_parameters"] == 4
    assert trial["delta_l2"] == pytest.approx((9**2 + 9**2 + 9**2 + 4**2) ** 0.5)
    assert target.weight.equal(parent_weights) and target.bias.tolist() == [0, 4, 0]
    assert store.revision == 1 and store.resolve(parent_row).tolist() == [0, 1, 2]
    assert store.export(tmp_path / "parent.safetensors")["changed_parameters"] == 1
    for handle in temp_handles:
        with pytest.raises(ValueError, match="stale"):
            store.resolve(handle)

    # Mutating the submitted recipe cannot change the recorded numerical candidate.
    writes[-1]["value"] = 42
    committed = previews.commit(trial["candidate_id"])
    assert committed["revision"] == 5 and committed["sha256"] == trial["sha256"]
    assert (target.weight.tolist(), target.bias.tolist()) == trial["after"]
    with pytest.raises(ValueError, match="stale"):
        store.resolve(parent_row)
    assert store.capture(target.weight[0]) not in temp_handles
    assert store.export(tmp_path / "committed.safetensors")["changed_parameters"] == 3
    fresh = Target()
    WeightStore(fresh).apply(tmp_path / "committed.safetensors")
    assert torch.equal(fresh.weight, target.weight) and torch.equal(fresh.bias, target.bias)
    with pytest.raises(ValueError, match="unknown"):
        previews.commit(trial["candidate_id"])


def test_rollback_on_late_invalid_write_and_on_probe_failure():
    target = Target()
    before = target.weight.clone()
    store = WeightStore(target)
    previews = WeightPreview(store)
    handle = store.read("weight", [1])["handle"]
    trial = previews.preview([fill("weight", [0, 0], 3)], lambda: None)
    with pytest.raises(ValueError, match="unknown writable"):
        previews.preview([fill("weight", [3], -1), fill("missing", [], 5)], lambda: None)

    def fail_on_trial():
        if store.revision:
            raise RuntimeError("scoring failed")

    with pytest.raises(RuntimeError, match="scoring failed"):
        previews.preview([fill("weight", [2], 11), fill("alias", [2, 0], -4)], fail_on_trial)
    assert store.revision == 0 and store.resolve(handle).equal(target.weight[1])
    assert before.equal(target.weight)
    assert previews.commit(trial["candidate_id"])["revision"] == 1


def test_stale_foreign_and_discarded_candidates():
    store = WeightStore(Target())
    previews = WeightPreview(store)
    writes = [fill("weight", [1, 0], 12)]
    trial = previews.preview(writes, lambda: None)
    replacement = previews.preview(writes, lambda: None)
    assert replacement["sha256"] == trial["sha256"]
    assert replacement["candidate_id"] != trial["candidate_id"]
    with pytest.raises(ValueError, match="unknown"):
        previews.commit(trial["candidate_id"])
    other = WeightPreview(WeightStore(Target()))
    copy = other.preview(writes, lambda: None)
    assert copy["sha256"] == replacement["sha256"]
    with pytest.raises(ValueError, match="unknown"):
        other.commit(replacement["candidate_id"])
    with pytest.raises(ValueError, match="parent revision"):
        previews.preview(writes, lambda: None, expected_revision=1)
    store.write(**fill("bias", [2], 7))
    with pytest.raises(ValueError, match="stale"):
        previews.commit(replacement["candidate_id"])
    assert previews.discard(replacement["candidate_id"])["state"] == "discarded"
    with pytest.raises(ValueError, match="unknown"):
        previews.discard(replacement["candidate_id"])
    assert store.target.weight[1, 0] == 3 and store.target.bias.tolist() == [0, 0, 7]


def test_zero_net_candidate_and_budgets(monkeypatch):
    import iwi_harness.weights as weights

    monkeypatch.setattr(weights, "MAX_STORED", 2)
    target = Target()
    store = WeightStore(target)
    previews = WeightPreview(store)
    edits = [fill("weight", [0, 0], 9), fill("weight", [0, 0], 0)]
    with pytest.raises(ValueError, match="expects"):
        previews.preview(edits * 5, lambda: None)
    with pytest.raises(ValueError, match="budget"):
        previews.preview(edits + [fill("bias", [2], 7)], lambda: None)
    assert store.revision == 0 and target.weight[0, 0].item() == 0
    trial = previews.preview(edits, lambda: None)
    assert trial["changed_parameters"] == 0 and trial["delta_l2"] == 0
    previews.discard(trial["candidate_id"])
    new = previews.preview(edits, lambda: None)
    assert previews.commit(new["candidate_id"])["revision"] == 2
    with pytest.raises(ValueError, match="budget"):
        store.write(**fill("bias", [1], 4))


def test_changed_preimage_rejects_whole_commit():
    target = Target()
    store = WeightStore(target)
    previews = WeightPreview(store)
    trial = previews.preview([fill("weight", [1, 0], 0), fill("bias", [1], 9)], lambda: None)
    target.bias[1] = 2
    with pytest.raises(ValueError, match="parent values changed"):
        previews.commit(trial["candidate_id"])
    assert target.weight[1, 0].item() == 3 and target.bias[1].item() == 2
    assert store.revision == 0


def test_signed_zero_preimages_survive_rollback_and_exact_replay(tmp_path):
    target = Target()
    store = WeightStore(target)
    previews = WeightPreview(store)
    write = {"tensor": "bias", "operation": "replace", "value": {"values": [-0.0, 1.0, 0.0]}}
    trial = previews.preview([write], lambda: target.bias.view(torch.int16).tolist())
    assert trial["changed_parameters"] == 2
    assert trial["before"] == [0, 0, 0] and trial["after"][0] == -32768
    assert target.bias.view(torch.int16).tolist() == trial["before"]
    previews.commit(trial["candidate_id"])
    fresh = Target()
    path = tmp_path / "signed-zero.safetensors"
    assert store.export(path)["changed_parameters"] == 2
    WeightStore(fresh).apply(path)
    assert fresh.bias.view(torch.int16).tolist() == trial["after"]
