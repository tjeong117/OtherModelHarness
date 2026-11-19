"""Bounded tensor operations and numerical patch replay for an editing session."""

import hashlib
import json
import math
import os
import secrets
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load, save

from .tools import validate_write_operands

NATIVE_FORMAT = "iwibench.replacements.v1"
GENERAL_FORMAT = "iwi-harness.replacements.v1"
MAX_ELEMENTS = 1_048_576
MAX_STORED = 4 * MAX_ELEMENTS
MAX_HANDLES = 128
MAX_INLINE = 256
MAX_PATCH_BYTES = 80 * 1024 * 1024
MAX_WRITES = 1024


def _scalar(value):
    if type(value) not in (float, int):
        raise ValueError("expected a finite numeric scalar")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("scalar is out of range") from exc
    if not math.isfinite(value):
        raise ValueError("expected a finite numeric scalar")
    return value


def _region(shape, selection):
    if selection is None:
        selection = []
    if not isinstance(selection, (list, tuple)) or len(selection) > len(shape):
        raise ValueError("invalid tensor selection")
    indices = []
    for size, item in zip(shape, selection):
        if type(item) is int and 0 <= item < size:
            indices.append(item)
        elif (isinstance(item, (tuple, list)) and len(item) == 2
              and all(type(i) is int for i in item) and 0 <= item[0] < item[1] <= size):
            indices.append(slice(*item))
        else:
            raise ValueError("use an in-bounds index or a half-open [start, stop] range")
    indices.extend(slice(0, size) for size in shape[len(indices):])
    return tuple(indices)


def _flat_indices(shape, region):
    # Logical row-major indices of just the edited slice, including column edits.
    indices = torch.zeros(1, dtype=torch.int64)
    for size, item in zip(shape, region):
        if isinstance(item, int):
            indices = indices * size + item
        else:
            axis = torch.arange(item.start, item.stop)
            indices = (indices[:, None] * size + axis).reshape(-1)
    return indices


def _fingerprint(parameter):
    data = parameter.detach().reshape(-1)
    h = hashlib.sha256()
    for chunk in data.split(MAX_ELEMENTS):
        raw = chunk.contiguous().view(torch.uint8).cpu().numpy()
        h.update(memoryview(raw))
    return {"dtype": str(parameter.dtype), "shape": list(parameter.shape), "sha256": h.hexdigest()}


def _value_bits(value):
    """Compare same-dtype floating weights exactly, including the sign bit of zero."""
    return value.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[value.element_size()])


def _check_size(t):
    if t.numel() > MAX_ELEMENTS:
        raise ValueError(f"tensor exceeds {MAX_ELEMENTS} elements; select a smaller region")
    return t


def _check_finite(t):
    if not torch.isfinite(t).all().item():
        raise ValueError("nonfinite tensor")
    return t


def _literal(values):
    # Bound untrusted nested sequences before asking torch to allocate anything.
    count = 0
    def visit(item, depth):
        nonlocal count
        count += 1
        if count > 2 * MAX_INLINE + 16 or depth > 8:
            raise ValueError("literal too large; reference stored tensors instead")
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        else:
            _scalar(item)
    visit(values, 0)
    try:
        data = torch.tensor(values, dtype=torch.float32)
    except (RuntimeError, TypeError, OverflowError) as exc:
        raise ValueError("invalid tensor literal") from exc
    if not 1 <= data.numel() <= MAX_INLINE:
        raise ValueError("invalid literal size")
    return _check_finite(data)


@dataclass
class _Change:
    indices: torch.Tensor  # sorted absolute indices, on CPU
    before: torch.Tensor   # original values immediately before this operation


@dataclass
class _Replacement:
    indices: torch.Tensor
    before: torch.Tensor
    after: torch.Tensor


class WeightStore:
    """An exclusive mutable view of one target instance. Tool handles are session-local."""

    def __init__(self, target):
        self.target = target
        self.params = dict(target.weights())
        self.revision = 0
        self._prefix = secrets.token_hex(4)
        self._handles = {}
        self._held = 0
        self._serial = 0
        self._base = {}
        self._journal = {}
        self._edited = 0
        self._canonical = {}
        canonical_by_id = {}
        for name, param in self.params.items():
            if not isinstance(name, str) or not name or name.startswith("__iwi_"):
                raise ValueError("invalid or reserved parameter name")
            if param.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise ValueError(f"unsupported writable dtype for {name}: {param.dtype}")
            self._canonical[name] = canonical_by_id.setdefault(id(param), name)

    def _param(self, name):
        if not isinstance(name, str) or name not in self.params:
            raise ValueError(f"unknown writable tensor: {name!r}. Call read_weights with {{}} to list exact tensor names and shapes.")
        return self.params[name].detach()

    def _snapshot(self, handle):
        if not isinstance(handle, str) or handle not in self._handles:
            raise ValueError("unknown, stale or foreign tensor handle")
        return self._handles[handle]

    def capture(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("capture requires a tensor")
        _check_size(tensor)
        if len(self._handles) >= MAX_HANDLES or self._held + tensor.numel() > MAX_STORED:
            raise ValueError("session handle budget exceeded")
        value = tensor.detach().to("cpu", copy=True).contiguous()
        _check_finite(value)
        self._serial += 1
        handle = f"{self._prefix}.{self._serial:x}"
        self._handles[handle] = value
        self._held += value.numel()
        return handle

    def resolve(self, handle):
        return self._snapshot(handle).clone()

    def _advance(self):
        self.revision += 1
        self._handles.clear()
        self._held = 0

    def read(self, tensor=None, selection=None, mode="ref", handle=None, offset=0, limit=16):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= MAX_INLINE:
            raise ValueError("invalid page offset or limit")
        if mode not in ("ref", "summary", "values"):
            raise ValueError("unknown read mode")
        if tensor is None and handle is None:
            if selection is not None:
                raise ValueError("a selection requires a tensor or handle")
            names = list(self.params)[offset:offset+limit]
            return {"revision": self.revision, "tensors": [
                {"name": n, "shape": list(self.params[n].shape), "dtype": str(self.params[n].dtype),
                 "canonical": self._canonical[n]} for n in names
            ], "next_offset": offset + limit if offset + limit < len(self.params) else None}
        if tensor is not None and handle is not None:
            raise ValueError("select exactly one tensor or handle")
        value = self._param(tensor) if handle is None else self._snapshot(handle)
        value = value[_region(value.shape, selection)]
        _check_size(value)
        out = {"revision": self.revision, "shape": list(value.shape), "dtype": str(value.dtype),
               "scalars": value.numel()}
        if mode == "ref":
            out["handle"] = handle if handle is not None and selection is None else self.capture(value)
        elif mode == "summary":
            if value.numel() == 0:
                raise ValueError("cannot summarize an empty tensor")
            v = value.double()
            out.update(mean=v.mean().item(), min=v.min().item(), max=v.max().item(),
                       l2=v.norm().item())
        else:
            out["values"] = value.reshape(-1)[offset:offset + limit].cpu().tolist()
            out["next_offset"] = offset + limit if offset + limit < value.numel() else None
        return out

    def _expression(self, expr, depth=0, nodes=None):
        if nodes is None:
            nodes = [0]
        nodes[0] += 1
        if depth > 8 or nodes[0] > 32:
            raise ValueError("tensor expression too complex")
        if not isinstance(expr, dict):
            return _literal(expr)
        keys = set(expr)
        if keys in ({"tensor"}, {"tensor", "selection"}, {"handle"}, {"handle", "selection"}):
            data = self._param(expr["tensor"]) if "tensor" in expr else self._snapshot(expr["handle"])
            data = data[_region(data.shape, expr.get("selection"))]
            return _check_size(data).cpu()
        if keys == {"values"}:
            return _literal(expr["values"])
        op = expr.get("op")
        valid = {"op", "left", "factor"} if op == "scale" else {"op", "left", "right"}
        if op not in ("scale", "outer", "add", "subtract") or keys != valid:
            raise ValueError("Use a tensor or handle reference, {values: [...]}, or an arithmetic expression with op and operands")
        a = self._expression(expr["left"], depth + 1, nodes)
        a = a if a.dtype == torch.float64 else a.float()
        if op == "scale":
            return _check_finite(a * _scalar(expr["factor"]))
        b = self._expression(expr["right"], depth + 1, nodes)
        if op == "outer":
            if a.ndim != 1 or b.ndim != 1 or a.numel() * b.numel() > MAX_ELEMENTS:
                raise ValueError("outer requires bounded-length vectors")
            return _check_finite(torch.outer(a, b))
        if a.shape != b.shape:
            raise ValueError("expression shapes must match exactly")
        return _check_finite(a + b if op == "add" else a - b)

    def write_destination(self, tensor, operation, operands):
        """Resolve the destination and check the operands before reading any weights."""
        param = self._param(tensor)
        validate_write_operands(operation, operands)
        return param

    @torch.no_grad()
    def write(self, tensor, operation, selection=None, **kwargs):
        if self.revision >= MAX_WRITES:
            raise ValueError("session write budget exceeded")
        param = self.write_destination(tensor, operation, kwargs)
        name = self._canonical[tensor]
        region = _region(param.shape, selection)
        view = _check_size(param[region])
        # Own the prior selected values even when the target is on CPU.
        prior = view.to("cpu", copy=True).contiguous()
        work = prior if prior.dtype == torch.float64 else prior.float()
        if operation == "low_rank_add":
            a = self._expression(kwargs["left"]).to(work.dtype)
            b = self._expression(kwargs["right"]).to(work.dtype)
            if (work.ndim != 2 or a.ndim != 2 or b.ndim != 2 or not 1 <= a.shape[1] <= 8
                    or a.shape[1] != b.shape[0] or (a.shape[0], b.shape[1]) != work.shape):
                raise ValueError("low-rank factors must match the matrix and have rank 1–8")
            new = work + (a @ b) * _scalar(kwargs.get("factor", 1))
        elif operation == "fill":
            new = torch.full_like(work, _scalar(kwargs["value"]))
        elif operation == "scale":
            new = work * _scalar(kwargs["factor"])
        else:
            operand = "source" if operation == "copy" else "value"
            value = self._expression(kwargs[operand])
            if value.shape != prior.shape:
                hint = "Use a tensor slice or handle of the selected shape."
                if value.ndim == 0 and prior.ndim == 1:
                    hint = (f"A numeric {operand} supplies one weight value. "
                            f"To use a row, set {operand} to a tensor reference with its row selection, "
                            'for example {"tensor":"T","selection":[7]}, or use a stored handle reference.')
                raise ValueError(f"{operation} expects {operand} shape {list(prior.shape)} for this selection, "
                                 f"received {list(value.shape)}. {hint}")
            new = work + value if operation == "add" else value
        replacement = _check_finite(new.to(param.dtype)).contiguous()
        mask = (_value_bits(prior) != _value_bits(replacement)).reshape(-1)
        changes = int(mask.sum().item())
        if self._edited + changes > MAX_STORED:
            raise ValueError("session changed-element budget exceeded")
        if changes:
            if name not in self._base:
                self._base[name] = _fingerprint(param)
            entry = _Change(_flat_indices(param.shape, region)[mask], prior.reshape(-1)[mask])
            try:
                view.copy_(replacement)
            except BaseException:
                view.copy_(prior)
                raise
            self._journal.setdefault(name, []).append(entry)
            self._edited += changes
        self._advance()
        return {"tensor": name, "revision": self.revision, "selected_parameters": prior.numel(),
                "changed_parameters": changes, "delta_l2": (prior.double() - replacement.double()).norm().item()}

    def _net_replacement(self, name):
        entries = self._journal[name]
        all_i = torch.cat([entry.indices for entry in entries])
        before = torch.cat([entry.before for entry in entries])
        # Stable order retains the *earliest* preimage for overlapping writes.
        indices, order = torch.sort(all_i, stable=True)
        first = torch.ones_like(indices, dtype=torch.bool)
        first[1:] = indices[1:] != indices[:-1]
        indices, before = indices[first], before[order[first]]
        param = self._param(name)
        current = torch.take(param, indices.to(param.device)).cpu()
        changed = _value_bits(current) != _value_bits(before)
        return _Replacement(indices[changed], before[changed], current[changed])

    @contextmanager
    def _staged_writes(self):
        """Isolate a bounded trial on the live model while its owner holds the session lock."""
        revision, edited = self.revision, self._edited
        journal, bases = self._journal, self._base
        handles, held = self._handles, self._held
        self._journal, self._base = {}, bases.copy()
        self._handles = handles.copy()
        try:
            yield
        finally:
            try:
                with torch.no_grad():
                    for name, entries in self._journal.items():
                        param = self._param(name)
                        for entry in reversed(entries):
                            param.put_(entry.indices.to(param.device), entry.before.to(param.device))
            finally:
                self.revision, self._edited = revision, edited
                self._journal, self._base = journal, bases
                # Never reuse the serial number of a temporary handle.
                self._handles, self._held = handles, held

    @torch.no_grad()
    def _commit_replacements(self, replacements, bases, writes, charged):
        """Commit the measured numerical values without re-running an edit recipe."""
        if self.revision + writes > MAX_WRITES or self._edited + charged > MAX_STORED:
            raise ValueError("candidate exceeds session edit budget")
        prepared = []
        for name, change in replacements.items():
            param = self._param(name)
            indices = change.indices.to(param.device)
            before = change.before.to(param.device)
            if not torch.equal(_value_bits(torch.take(param, indices)), _value_bits(before)):
                raise ValueError("candidate parent values changed")
            prepared.append((param, indices, before, change.after.to(param.device)))
        try:
            for param, indices, _, after in prepared:
                param.put_(indices, after)
        except BaseException:
            for param, indices, before, _ in prepared:
                param.put_(indices, before)
            raise
        for name, change in replacements.items():
            self._base.setdefault(name, bases[name])
            self._journal.setdefault(name, []).append(_Change(change.indices, change.before))
        self._edited += charged
        self.revision += writes - 1
        self._advance()

    def export(self, path):
        payload, sparse, bases = {}, [], {}
        count = 0
        for name in sorted(self._journal):
            change = self._net_replacement(name)
            indices, values = change.indices, change.after
            if not indices.numel():
                continue
            bases[name] = self._base[name]
            count += indices.numel()
            if indices.numel() == self.params[name].numel():
                payload[name] = values.reshape(self.params[name].shape)
            else:
                i, v = f"__iwi_indices_{len(sparse)}", f"__iwi_values_{len(sparse)}"
                sparse.append({"name": name, "indices": i, "values": v})
                payload[i], payload[v] = indices, values
        native = (all(v["dtype"] == "torch.bfloat16" for v in bases.values()) if bases
                  else all(param.dtype == torch.bfloat16 for param in self.params.values()))
        def canonical_json(obj):
            return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        metadata = {"format": NATIVE_FORMAT if native else GENERAL_FORMAT, "model_id": self.target.model_id,
                    "revision": self.target.revision, "sparse": canonical_json(sparse),
                    "base_tensors": canonical_json(bases)}
        data = save(payload, metadata=metadata)
        # The native serializer orders payload tensors, but its string metadata map uses
        # a randomized iteration order. Safetensors offsets are relative to the payload.
        start = 8 + int.from_bytes(data[:8], "little")
        header = canonical_json(json.loads(data[8:start])).encode("utf-8")
        header += b" " * (-len(header) % 8)
        parts = (len(header).to_bytes(8, "little"), header, memoryview(data)[start:])

        path = Path(path)
        stream = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
        digest = hashlib.sha256()
        try:
            with stream:
                for part in parts:
                    stream.write(part)
                    digest.update(part)
            os.replace(stream.name, path)
        finally:
            Path(stream.name).unlink(missing_ok=True)
        return {"path": str(path), "sha256": digest.hexdigest(), "bytes": sum(map(len, parts)),
                "format": metadata["format"], "changed_parameters": count, "revision": self.revision}

    @torch.no_grad()
    def apply(self, path, expected_sha256=None):
        if self.revision != 0:
            raise ValueError("apply requires a pristine editing session")
        # Use one bounded byte snapshot for digest, metadata and tensor payload.
        with Path(path).open("rb") as stream:
            data = stream.read(MAX_PATCH_BYTES + 1)
        if len(data) > MAX_PATCH_BYTES or len(data) < 8:
            raise ValueError("invalid patch size")
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and expected_sha256 != digest:
            raise ValueError("patch SHA-256 mismatch")
        header_size = int.from_bytes(data[:8], "little")
        if not 0 < header_size <= min(2 * 1024 * 1024, len(data)-8):
            raise ValueError("invalid patch header")
        header = json.loads(data[8:8 + header_size])
        metadata = header.get("__metadata__") if isinstance(header, dict) else None
        if (not isinstance(metadata, dict) or metadata.get("format") not in (NATIVE_FORMAT, GENERAL_FORMAT)
                or metadata.get("model_id") != self.target.model_id or metadata.get("revision") != self.target.revision):
            raise ValueError("patch format or checkpoint identity mismatch")
        bases = json.loads(metadata["base_tensors"]) if "base_tensors" in metadata else None
        if bases is None and (expected_sha256 is None or metadata["format"] != NATIVE_FORMAT):
            raise ValueError("external native patches without base hashes require expected_sha256")
        if bases is not None and not isinstance(bases, dict):
            raise ValueError("invalid patch base fingerprints")
        tensors = load(data)
        sparse = json.loads(metadata.get("sparse", "[]"))
        if not isinstance(sparse, list):
            raise ValueError("invalid sparse manifest")
        declared = {}
        used = set()
        for entry in sparse:
            if (not isinstance(entry, dict) or set(entry) != {"name", "indices", "values"}
                    or not all(isinstance(v, str) for v in entry.values())):
                raise ValueError("invalid sparse entry")
            n, i, v = entry["name"], entry["indices"], entry["values"]
            if (n in declared or i == v or {i,v} & used or not {i,v} <= tensors.keys()
                    or not all(k.startswith("__iwi_") for k in (i,v))):
                raise ValueError("duplicate or missing sparse entry")
            used.update((i,v))
            declared[n] = tensors[i], tensors[v]
        for name, value in tensors.items():
            if name not in used:
                if name in declared:
                    raise ValueError("multiple representations of tensor")
                declared[name] = None, value
        if bases is not None and set(bases) != set(declared):
            raise ValueError("patch base fingerprints do not cover its tensors")
        prepared, base_digests = {}, {}
        total = 0
        for name, (indices, value) in declared.items():
            param = self._param(name)
            canonical = self._canonical[name]
            if canonical in prepared:
                raise ValueError("multiple changes to tied aliases")
            total += value.numel()
            if total > MAX_STORED or value.dtype != param.dtype:
                raise ValueError("patch tensor dtype or size mismatch")
            if metadata["format"] == NATIVE_FORMAT and value.dtype != torch.bfloat16:
                raise ValueError("native IWIBench patches require BF16 values")
            _check_finite(value)
            if indices is None:
                if value.shape != param.shape:
                    raise ValueError("dense replacement shape mismatch")
                indices = torch.arange(value.numel())
            elif (indices.dtype != torch.int64 or indices.ndim != 1 or value.shape != indices.shape
                  or not indices.numel() or indices[0] < 0 or indices[-1] >= param.numel()
                  or not (indices[1:] > indices[:-1]).all()):
                raise ValueError("invalid sparse indices or value shape")
            base = _fingerprint(param)
            if bases is not None and bases[name] != base:
                raise ValueError(f"base digest mismatch for {name}")
            base_digests[canonical] = base
            indices = indices.to(param.device)
            prepared[canonical] = (param, indices, value.reshape(-1).to(param.device),
                                   torch.take(param, indices))
        # Validation and device allocations complete before changing any parameters.
        journal = {name: [_Change(indices.cpu(), before.cpu())]
                   for name, (_, indices, _, before) in prepared.items()}
        try:
            for param, indices, value, _ in prepared.values():
                param.put_(indices, value)
        except Exception:
            for param, indices, _, before in prepared.values():
                param.put_(indices, before)
            raise
        self._journal = journal
        self._base = base_digests
        self._edited = total
        self._advance()
        return {"sha256": digest, "revision": self.revision, "replaced_parameters": total}
