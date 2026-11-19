"""Reversible numerical weight candidates for an exclusively owned editing session."""

import hashlib
import json
import math
import secrets
from dataclasses import dataclass

import torch

from .weights import MAX_WRITES, WeightStore, _Replacement

MAX_PREVIEW_WRITES = 8


@dataclass
class _Candidate:
    identifier: str
    sha256: str
    parent_revision: int
    replacements: dict[str, _Replacement]
    bases: dict
    write_count: int
    charged: int


def _digest(store, parent_revision, replacements, bases):
    # Bind the numerical payload to the checkpoint and preimage. Names, shapes,
    # dtypes and counts delimit the three raw flat arrays for each parameter.
    h = hashlib.sha256()
    identity = [
        "iwi-harness.preview.v1",
        store.target.model_id,
        store.target.revision,
        parent_revision,
        [[name, bases[name], change.indices.numel()] for name, change in sorted(replacements.items())],
    ]
    h.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    for _, change in sorted(replacements.items()):
        for tensor in (change.indices, change.before, change.after):
            h.update(memoryview(tensor.contiguous().view(torch.uint8).numpy()))
    return h.hexdigest()


class WeightPreview:
    """Keep one measured candidate per session. Call under the editing session lock."""

    def __init__(self, store: WeightStore):
        self.store = store
        self._candidate = None

    def preview(self, writes, evaluate, *, expected_revision=None):
        """Score the parent and trial through the same read-only behavior callback."""
        if not isinstance(writes, list) or not 1 <= len(writes) <= MAX_PREVIEW_WRITES:
            raise ValueError(f"preview expects 1–{MAX_PREVIEW_WRITES} writes")
        if not all(isinstance(write, dict) and {"tensor", "operation"} <= write.keys() for write in writes):
            raise ValueError("each preview write requires a tensor and operation")
        store = self.store
        parent = store.revision
        if expected_revision is not None and (
            type(expected_revision) is not int or expected_revision != parent
        ):
            raise ValueError("candidate parent revision mismatch")
        if parent + len(writes) > MAX_WRITES:
            raise ValueError("candidate exceeds session write budget")
        parent_handles, parent_held, parent_edited = store._handles, store._held, store._edited
        with store._staged_writes():
            before = evaluate()
            for write in writes:
                # Parent refs are immutable inputs to the entire candidate recipe,
                # including a later write that consumes the same feature vector.
                store._handles, store._held = parent_handles.copy(), parent_held
                store.write(**write)
            after = evaluate()
            replacements = {}
            for name in store._journal:
                change = store._net_replacement(name)
                if change.indices.numel():
                    replacements[name] = change
            bases = {name: store._base[name] for name in replacements}
            charged = store._edited - parent_edited

        sha = _digest(store, parent, replacements, bases)
        count = sum(c.indices.numel() for c in replacements.values())
        norm = math.hypot(
            *[(c.after.double() - c.before.double()).norm().item() for c in replacements.values()]
        )
        identifier = secrets.token_hex(16)
        self._candidate = _Candidate(identifier, sha, parent, replacements, bases, len(writes), charged)
        return {
            "candidate_id": identifier,
            "sha256": sha,
            "parent_revision": parent,
            "changed_parameters": count,
            "delta_l2": norm,
            "before": before,
            "after": after,
        }

    def _get(self, candidate_id):
        candidate = self._candidate
        if candidate is None or candidate.identifier != candidate_id:
            raise ValueError("unknown or discarded candidate")
        return candidate

    def commit(self, candidate_id):
        candidate = self._get(candidate_id)
        if candidate.parent_revision != self.store.revision:
            raise ValueError("stale candidate: parent revision changed")
        self.store._commit_replacements(
            candidate.replacements, candidate.bases, candidate.write_count, candidate.charged
        )
        self._candidate = None
        return {
            "state": "committed",
            "candidate_id": candidate_id,
            "sha256": candidate.sha256,
            "parent_revision": candidate.parent_revision,
            "revision": self.store.revision,
            "changed_parameters": sum(c.indices.numel() for c in candidate.replacements.values()),
        }

    def discard(self, candidate_id):
        candidate = self._get(candidate_id)
        self._candidate = None
        return {
            "state": "discarded",
            "candidate_id": candidate_id,
            "sha256": candidate.sha256,
            "parent_revision": candidate.parent_revision,
            "revision": self.store.revision,
        }
