"""Public greedy-answer experiments on the current target, optionally with a temporary intervention."""

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from .probes import _integer


MAX_CASES, MAX_NEW_TOKENS, MAX_DISPLAY = 6, 32, 160
_NATIVE_MLPS = {
    ("transformers.models.gemma3.modeling_gemma3", "Gemma3MLP"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP"),
}


def behavior_tool(internal, *, sae_available=True):
    """Return the tool description and transport schema for this arm."""
    from .tools import INDEX, PROMPT, TOKEN_PAIR, arr, integer, obj

    case = obj({"prompt": PROMPT, "answers": arr({"type": "string", "minLength": 1,
                                                  "maxLength": 512}, 6),
                "slice": {"enum": ["improve", "preserve"]}}, ["prompt", "answers", "slice"])
    fields = {"cases": arr(case, MAX_CASES), "max_new_tokens": integer(1, MAX_NEW_TOKENS),
              "token_ids": TOKEN_PAIR}
    desc = ("Score public improve/preserve prompts using greedy answers and whitespace-trimmed exact "
            "match. A length-limited answer is flagged. An optional [wanted, alternative] pair "
            "adds its first-step logit margin. Use public examples only.")
    if internal:
        scale = {"type": "number", "minimum": -16, "maximum": 16}
        fields["intervention"] = {"anyOf": [
            obj({"kind": {"const": "native"}, "layer": integer(0, 1024), "unit": INDEX,
                 "scale": scale, "matched_control": {"type": "boolean"}}, ["layer", "scale"]),
            obj({"kind": {"const": "sae"}, "layer": integer(0, 1024), "feature": INDEX,
                 "scale": scale}, ["kind", "feature", "scale"]),
        ]}
        desc += (" Optionally scale one gated MLP input to down_proj across every position and "
                 "generation step. Give layer, scale and optional unit (default: largest absolute "
                 "activation at the first prompt's last position). A verified dense native site "
                 "returns a corresponding weight-column write. Includes an identity control. Set "
                 "matched_control=true to test another same-layer unit with similar public prefill RMS.")
        if sae_available:
            desc += " Alternatively supply kind=sae, feature and scale for the active residual dictionary."
        else:
            fields["intervention"] = fields["intervention"]["anyOf"][0]
    return desc, obj(fields, ["cases"])


def _digest(value):
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compare_behavior(before, after):
    """Paired denominators and signed outcomes for each requested public slice."""
    if len(before) != len(after):
        raise ValueError("different numbers of public cases")
    grouped = {}
    for a, b in zip(before, after, strict=True):
        if (a["case_id"], a["slice"]) != (b["case_id"], b["slice"]):
            raise ValueError("public case or slice differs between paired runs")
        grouped.setdefault(a["slice"], []).append((a, b))
    summary = {}
    for label, pairs in grouped.items():
        fixes = [b["index"] for a, b in pairs if not a["correct"] and b["correct"]]
        losses = [b["index"] for a, b in pairs if a["correct"] and not b["correct"]]
        correct_before = sum(a["correct"] for a, _ in pairs)
        correct_after = sum(b["correct"] for _, b in pairs)
        deltas = [b["margin"] - a["margin"] for a, b in pairs if a["margin"] is not None
                  and b["margin"] is not None]
        summary[label] = {"n": len(pairs), "correct_before": correct_before,
                          "correct_after": correct_after, "correct_delta": correct_after - correct_before,
                          "accuracy_delta": (correct_after - correct_before) / len(pairs),
                          "fixed": fixes, "regressed": losses,
                          "retained_correct": correct_before - len(losses),
                          "correct_at_token_limit_before": sum(a["correct"] and a["truncated"] for a, _ in pairs),
                          "correct_at_token_limit_after": sum(b["correct"] and b["truncated"] for _, b in pairs),
                          "mean_margin_change": sum(deltas) / len(deltas) if deltas else None}
    return summary


class _ActivityMatch:
    """Collect per-unit activity on the identity pass, once per natural public prefill position."""

    def __init__(self):
        self.squares = self.active = self.coactive = None
        self.positions = 0

    def observe(self, rows, unit):
        # Metal devices do not support float64; perform their reductions on CPU.
        device = rows.device if rows.device.type != "mps" else torch.device("cpu")
        if self.squares is None:
            self.squares = torch.zeros(rows.shape[1], device=device, dtype=torch.float64)
            self.active = torch.zeros_like(self.squares, dtype=torch.long)
            self.coactive = torch.zeros_like(self.active)
        # Bound scratch space independently of the target's intermediate width and prefix length.
        for chunk in rows.split(64):
            values = chunk.to(device=device, dtype=torch.float64)
            nonzero = values != 0
            self.squares += values.square().sum(dim=0)
            self.active += nonzero.sum(dim=0)
            self.coactive += (nonzero & nonzero[:, unit:unit + 1]).sum(dim=0)
            self.positions += len(chunk)

    def select(self, unit):
        rms = (self.squares.cpu() / self.positions).sqrt()
        if not torch.isfinite(rms).all():
            raise ValueError("non-finite public calibration RMS")
        active, coactive = self.active.cpu(), self.coactive.cpu()

        def stats(index):
            return {"unit": index, "rms": float(rms[index]), "active_positions": int(active[index]),
                    "active_fraction": float(active[index]) / self.positions}

        result = {"rule": "closest absolute prefill RMS among other active same-layer units; lowest ID breaks ties",
                  "activity": "nonzero gated input", "positions": self.positions, "selected": stats(unit)}
        if active[unit] == 0:
            return {**result, "status": "unmatched", "reason": "selected unit inactive on all public prefill positions"}
        eligible = active > 0
        eligible[unit] = False
        if not eligible.any():
            return {**result, "status": "unmatched", "reason": "all alternative units inactive on public prefills"}
        differences = (rms - rms[unit]).abs()
        differences[~eligible] = float("inf")
        other = int(differences.argmin())
        ratio = float(rms[other] / rms[unit])
        result.update(alternative=stats(other), coactive_positions=int(coactive[other]),
                      rms_ratio=ratio, rms_difference=float(differences[other]))
        if not coactive[other]:
            return {**result, "status": "unmatched", "reason": "closest unit active on disjoint public positions"}
        return {**result, "status": "matched"}


def _weight_names(target):
    names = {}
    for name, weight in target.weights().items():
        names.setdefault(id(weight), []).append(name)
    return names


def _native_site(block, names):
    mlp = getattr(block, "mlp", None)
    kind = type(mlp)
    if (kind.__module__, kind.__name__) not in _NATIVE_MLPS:
        raise ValueError("no verified dense gated MLP down_proj for this block (supports Gemma3MLP or Qwen2MLP)")
    gate, up, down = (getattr(mlp, name, None) for name in ("gate_proj", "up_proj", "down_proj"))
    if (any(type(m) is not nn.Linear for m in (gate, up, down))
            or gate.weight.shape != up.weight.shape or gate.weight.shape != down.weight.T.shape
            or not down.weight.is_floating_point()):
        raise ValueError("native unit intervention requires matching dense gate/up/down projections")
    writable = names.get(id(down.weight), [])
    if len(writable) != 1:
        raise ValueError("down_proj.weight must map to a single writable target tensor")
    return down, writable[0]


def supported_native_layers(target):
    """Return layers with an adapter-verified dense MLP and an unambiguous writable column."""
    names = _weight_names(target)
    layers = []
    for layer, block in enumerate(target.blocks):
        try:
            _native_site(block, names)
        except ValueError:
            continue
        layers.append(layer)
    return layers


class _NativeUnit:
    def __init__(self, runner, spec):
        if set(spec) - {"kind", "layer", "unit", "scale", "matched_control"}:
            raise ValueError("native intervention accepts layer, scale and optional unit or matched_control")
        if type(spec.get("matched_control", False)) is not bool:
            raise ValueError("matched_control must be a boolean")
        self.runner = runner
        self.layer = _integer(spec.get("layer"), "layer", 0, len(runner.target.blocks) - 1)
        self.module, self.name = _native_site(runner.target.blocks[self.layer], _weight_names(runner.target))
        self.unit = (_integer(spec["unit"], "unit", 0, self.module.in_features - 1)
                     if "unit" in spec else None)
        self.selection = "supplied" if self.unit is not None else "first_prompt_last_token_max_abs"
        self.calibration = _ActivityMatch() if spec.get("matched_control") else None
        self.measure_displacement = False
        self.displacement_squared = 0.
        self.displacement_rows = 0

    def forward(self, ids, scale, observe):
        visits, activation = 0, None

        def scale_input(module, args):
            nonlocal visits, activation
            visits += 1
            if (visits != 1 or len(args) != 1 or not isinstance(args[0], torch.Tensor)
                    or tuple(args[0].shape) != (1, len(ids), module.in_features)):
                raise ValueError("expected one gated MLP input of shape [1, prefix, intermediate_width]")
            z = args[0]
            self.runner.usage["probe_rows"] += len(ids)
            if self.unit is None:
                self.unit = int(z[0, -1].abs().argmax())
            if observe:
                activation = float(z[0, -1, self.unit])
                if not math.isfinite(activation):
                    raise ValueError("non-finite native unit activation")
                if self.calibration is not None:
                    self.calibration.observe(z[0], self.unit)
            # Use exactly the same hook path for both the identity and the treatment.
            replacement = z.clone()
            replacement[..., self.unit] *= scale
            if self.measure_displacement:
                delta = (replacement[..., self.unit].to("cpu", torch.float64)
                         - z[..., self.unit].to("cpu", torch.float64))
                self.displacement_squared += float(delta.square().sum())
                self.displacement_rows += delta.numel()
            return (replacement,)

        handle = self.module.register_forward_pre_hook(scale_input)
        try:
            logits, _ = self.runner._forward(ids)
            if visits != 1:
                raise ValueError("down_proj was not reached by this target forward")
            return logits, activation
        finally:
            handle.remove()

    def displacement(self):
        input_l2 = math.sqrt(self.displacement_squared)
        column_l2 = float(self.module.weight[:, self.unit].float().norm())
        return {"scope": "all applied prefix positions, including replay during decoding",
                "applied_positions": self.displacement_rows, "gated_input_delta_l2": input_l2,
                "weight_column_l2": column_l2,
                "projected_delta_l2_before_output_rounding": input_l2 * column_l2}

    def describe(self, scale):
        return {"kind": "native", "layer": self.layer, "unit": self.unit, "selection": self.selection,
                "site": "gated_mlp_down_proj_input", "positions": "all_prefix_positions_each_step",
                "scale": scale, "write": {"tensor": self.name, "operation": "scale",
                                         "selection": [[0, self.module.out_features], self.unit], "factor": scale}}


class _SAEFeature:
    def __init__(self, runner, spec):
        if set(spec) - {"kind", "layer", "feature", "scale"}:
            raise ValueError("SAE intervention accepts feature, scale and optional dictionary layer")
        self.runner = runner
        self.probe = runner._probe({k: v for k, v in spec.items() if k in ("kind", "layer")})
        self.feature, _, _ = runner._intervention({"feature": spec.get("feature"), "scale": spec["scale"]},
                                                 self.probe)

    def forward(self, ids, scale, observe):
        logits, captures = self.runner._forward(
            ids, self.probe, observed=[len(ids) - 1] if observe else [],
            changed=list(range(len(ids))), intervention=(self.feature, "scale", scale),
            capture_feature=self.feature)
        return logits, float(captures[0]) if captures else None

    def describe(self, scale):
        info = {"kind": "sae", "layer": self.probe.layer, "feature": self.feature,
                "site": "post_block_residual", "positions": "all_prefix_positions_each_step",
                "scale": scale, "dictionary_revision": self.runner.sae.revision}
        dictionary_id = getattr(self.runner.sae, "dictionary_id", None)
        if dictionary_id:
            info["dictionary_id"] = dictionary_id
        return info


class BehaviorExperiment:
    def __init__(self, runner, *, internal=False, artifact_dir=None):
        self.runner = runner
        self.internal = internal
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._baselines = OrderedDict()
        self._cache_revision = None
        self._cache_journal = None

    def _prepare(self, cases, max_new_tokens, token_ids):
        _integer(max_new_tokens, "max_new_tokens", 1, MAX_NEW_TOKENS)
        if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
            raise ValueError(f"supply 1 to {MAX_CASES} public cases")
        for case in cases:
            if (not isinstance(case, dict) or set(case) != {"slice", "prompt", "answers"}
                    or case["slice"] not in ("improve", "preserve")):
                raise ValueError("each case needs prompt, answers, and slice improve or preserve")
            answers = case["answers"]
            if (not isinstance(answers, list) or not 1 <= len(answers) <= 6
                    or not all(isinstance(a, str) and 0 < len(a) <= 512 and a.strip() for a in answers)):
                raise ValueError("supply 1 to 6 nonempty exact answers of at most 512 characters")
        if token_ids is not None:
            if not isinstance(token_ids, list) or len(token_ids) != 2:
                raise ValueError("token_ids must contain [wanted, alternative]")
            upper = getattr(self.runner.target, "vocab_size", 2**40 + 1) - 1
            for t in token_ids:
                _integer(t, "token_id", 0, upper)
            if token_ids[0] == token_ids[1]:
                raise ValueError("wanted and alternative token IDs must differ")
        return self.runner._encode([c["prompt"] for c in cases], MAX_CASES)

    def _score(self, cases, prefixes, limit, token_ids, site=None, scale=1.):
        rows = []
        for i, (case, prefix) in enumerate(zip(cases, prefixes, strict=True)):
            ids, generated = list(prefix), []
            margin, first_activity = None, None
            for step in range(limit):
                if site is None:
                    logits, _ = self.runner._forward(ids)
                else:
                    logits, activation = site.forward(ids, scale, observe=step == 0)
                    if step == 0:
                        first_activity = activation
                if step == 0 and token_ids is not None:
                    if max(token_ids) >= logits.numel():
                        raise ValueError("token_ids exceed the target vocabulary")
                    margin = float(logits[token_ids[0]]) - float(logits[token_ids[1]])
                predicted = int(logits.argmax())
                generated.append(predicted)
                if predicted in self.runner.target.eos_ids:
                    break
                ids.append(predicted)
            answer = self.runner.target.decode(generated)
            rows.append({"index": i, "case_id": _digest(case)[:16], "slice": case["slice"],
                         "correct": answer.strip() in [a.strip() for a in case["answers"]],
                         "text": answer[:MAX_DISPLAY], "display_truncated": len(answer) > MAX_DISPLAY,
                         "answer_sha256": _digest(answer),
                         "truncated": generated[-1] not in self.runner.target.eos_ids,
                         "generated_tokens": len(generated), "first_token_id": generated[0],
                         "generation_id": _digest(generated)[:16], "margin": margin})
            if site is not None:
                rows[-1]["first_activity"] = first_activity
        return rows

    @torch.inference_mode()
    def evaluate(self, cases, max_new_tokens=12, token_ids=None):
        """Bounded public rows, callable on both sides of a temporary weight edit."""
        with self.runner._lock:
            prefixes = self._prepare(cases, max_new_tokens, token_ids)
            return self._score(cases, prefixes, max_new_tokens, token_ids)

    def _baseline(self, cases, prefixes, limit, token_ids):
        # Branches stage a separate journal and can visit the same revision as a later commit.
        # Preview callbacks always use evaluate(), which bypasses this cache completely.
        store = self.runner.store
        journal = getattr(store, "_journal", None)
        if journal is None:
            return self._score(cases, prefixes, limit, token_ids), False
        if self._cache_revision != store.revision or self._cache_journal is not journal:
            self._baselines.clear()
            self._cache_revision, self._cache_journal = store.revision, journal
        key = _digest([cases, prefixes, limit, token_ids])
        if key in self._baselines:
            self._baselines.move_to_end(key)
            return [row.copy() for row in self._baselines[key]], True
        rows = self._score(cases, prefixes, limit, token_ids)
        self._baselines[key] = [row.copy() for row in rows]
        if len(self._baselines) > 16:
            self._baselines.popitem(last=False)
        return rows, False

    @torch.inference_mode()
    def run(self, cases, max_new_tokens=12, token_ids=None, intervention=None):
        with self.runner._lock:
            prefixes = self._prepare(cases, max_new_tokens, token_ids)
            site = None
            if intervention is not None:
                if not self.internal:
                    raise ValueError("internal interventions are disabled in this arm")
                if not isinstance(intervention, dict):
                    raise ValueError("expected an intervention object")
                scale = intervention.get("scale")
                if type(scale) not in (int, float) or not -16 <= scale <= 16:
                    raise ValueError("scale must be finite and between -16 and 16")
                kind = intervention.get("kind", "native")
                if kind not in ("native", "sae"):
                    raise ValueError("choose native or sae intervention")
                site = (_NativeUnit if kind == "native" else _SAEFeature)(self.runner, intervention)
            before = self.runner.usage.copy()
            baseline, cache_hit = self._baseline(cases, prefixes, max_new_tokens, token_ids)
            result = {"baseline": baseline, "baseline_cache_hit": cache_hit, "revision": self.runner.store.revision}
            alternative_rows = None
            if site is not None:
                identity = self._score(cases, prefixes, max_new_tokens, token_ids, site, 1.)
                mismatches = [i for i, (a, b) in enumerate(zip(baseline, identity, strict=True))
                              if a["generation_id"] != b["generation_id"] or a["margin"] != b["margin"]]
                result["identity"] = {"matches": not mismatches, "mismatches": mismatches}
                result["intervention"] = site.describe(scale)
                if mismatches:
                    result["error"] = "Identity intervention changed public outputs; treatment was skipped."
                    result["intervention"].pop("write", None)
                else:
                    matched = None
                    if isinstance(site, _NativeUnit) and site.calibration is not None:
                        matched = site.calibration.select(site.unit)
                        site.calibration = None
                        site.measure_displacement = True
                    after = self._score(cases, prefixes, max_new_tokens, token_ids, site, scale)
                    result.update(after=after, comparison=compare_behavior(baseline, after))
                    if matched is not None:
                        result["matched_control"] = matched
                        matched["selected_displacement"] = site.displacement()
                        if matched["status"] == "matched":
                            other = _NativeUnit(self.runner, {"layer": site.layer, "unit": matched["alternative"]["unit"]})
                            other.measure_displacement = True
                            alternative_rows = self._score(cases, prefixes, max_new_tokens, token_ids, other, scale)
                            matched["alternative_displacement"] = other.displacement()
                            matched["comparison"] = compare_behavior(baseline, alternative_rows)
            result["cost"] = {k: self.runner.usage[k] - v for k, v in before.items()}
            if self.artifact_dir is not None:
                record = {"target": self.runner.target.model_id, "checkpoint": self.runner.target.revision,
                          "cases": cases, "max_new_tokens": max_new_tokens, "token_ids": token_ids, "result": result}
                if alternative_rows is not None:
                    record["matched_control_rows"] = alternative_rows
                artifact = Path("behavior") / (_digest(record) + ".json")
                path = self.artifact_dir / artifact
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
                result["artifact"] = str(artifact)
            return result
