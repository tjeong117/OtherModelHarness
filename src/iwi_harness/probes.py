"""Metered target forwards and temporary probes at transformer block outputs."""

import math
from dataclasses import dataclass
from threading import RLock

import torch


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _indices(value, name, low, high, limit):
    if not isinstance(value, list) or not 1 <= len(value) <= limit:
        raise ValueError(f"{name} must be a list of 1 to {limit} indices")
    result = [_integer(i, name, low, high) for i in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices")
    return result


def _positions(indices, length):
    result = [i + length if i < 0 else i for i in indices]
    if any(i < 0 or i >= length for i in result) or len(set(result)) != len(result):
        raise ValueError(f"positions must resolve to distinct tokens in a prefix of length {length}. Use [-1] for the last token.")
    return result


def _rank(vector, count):
    indices = vector.abs().topk(min(count, vector.numel())).indices.tolist()
    return [{"index": i, "value": vector[i].item()} for i in indices]


@dataclass(frozen=True)
class _Probe:
    layer: int
    kind: str
    positions: list
    top_k: int
    phase: str


class Runner:
    """Run one exclusively owned target in eval mode; all methods accept public inputs only."""

    def __init__(self, target, store, sae=None, *, max_forward_tokens=262144, max_forwards=256,
                 report_suppression_activity=False):
        self.target = target
        self.store = store
        self.sae = sae
        self.report_suppression_activity = report_suppression_activity
        self.max_forward_tokens = _integer(max_forward_tokens, "max_forward_tokens", 1, 2**53 - 1)
        self.max_forwards = _integer(max_forwards, "max_forwards", 1, 2**53 - 1)
        self.usage = dict(forwards=0, forward_tokens=0, probe_rows=0, sae_rows=0)
        self._lock = RLock()
        if sae is not None and (sae.model_id, sae.target_revision) != (target.model_id, target.revision):
            raise ValueError("SAE belongs to a different target checkpoint")
        self.target.model.eval()

    def _probe(self, spec):
        if not isinstance(spec, dict) or set(spec) - {"layer", "kind", "positions", "top_k", "phase"}:
            raise ValueError("expected a probe with layer, kind, positions, top_k or phase")
        kind = spec.get("kind", "residual")
        if kind not in ("residual", "sae"):
            raise ValueError("probe.kind must be residual or sae")
        if kind == "sae":
            if self.sae is None:
                raise ValueError("no SAE loaded for this target; a residual probe is available")
            layer = spec.get("layer", self.sae.layer)
            if type(layer) is not int or layer != self.sae.layer:
                raise ValueError(f"the loaded SAE is at layer {self.sae.layer}; omit layer to select it")
        else:
            layer = spec.get("layer")
        layer = _integer(layer, "layer", 0, len(self.target.blocks) - 1)
        positions = _indices(spec.get("positions", [-1]), "positions", -2304, 2303, 16)
        phase = spec.get("phase", "prefill")
        if phase not in ("prefill", "each_step"):
            raise ValueError("probe.phase must be prefill or each_step")
        return _Probe(layer, kind, positions, _integer(spec.get("top_k", 5), "top_k", 1, 16), phase)

    def _intervention(self, spec, probe):
        if spec is None:
            return None
        if probe is None or probe.kind != "sae":
            raise ValueError("a feature intervention requires a matching SAE probe")
        if not isinstance(spec, dict) or set(spec) not in ({"feature", "scale"}, {"feature", "value"}):
            raise ValueError("intervention requires a feature and exactly one of scale or value")
        feature = _integer(spec["feature"], "feature", 0, self.sae.n_features - 1)
        operation = "scale" if "scale" in spec else "value"
        value = spec[operation]
        if type(value) not in (float, int) or not -1e8 <= value <= 1e8 or not math.isfinite(value):
            raise ValueError("feature scale or value must be finite and have magnitude at most 1e8")
        return feature, operation, float(value)

    def _encode(self, prompts, max_items, probe=None):
        if not isinstance(prompts, list) or not 1 <= len(prompts) <= max_items:
            raise ValueError(f"expected 1 to {max_items} prompts")
        sequences = []
        for prompt in prompts:
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 65536:
                raise ValueError("each prompt must be a nonempty string of at most 65,536 characters")
            ids = self.target.encode(prompt)
            if not 1 <= len(ids) <= 2048:
                raise ValueError("expected 1 to 2,048 encoded prompt tokens")
            if probe:
                _positions(probe.positions, len(ids))
            sequences.append(ids)
        return sequences

    def _forward(self, ids, probe=None, observed=(), changed=(), intervention=None, *, capture_feature=None):
        length = len(ids)
        if (self.usage["forward_tokens"] + length > self.max_forward_tokens
                or self.usage["forwards"] >= self.max_forwards):
            raise ValueError("target forward budget exhausted")
        # Full-prefix decoding has no KV cache; charge the entire prefix at each step.
        device = next(self.target.model.parameters()).device
        inputs = torch.tensor([ids], device=device, dtype=torch.long)
        captures = []
        visits = 0

        def hook(module, args, output):
            nonlocal visits
            visits += 1
            if visits != 1:
                raise ValueError("the requested block ran more than once in one target forward")
            h = output if isinstance(output, torch.Tensor) else output[0]
            if h.ndim != 3 or tuple(h.shape[:2]) != (1, length):
                raise ValueError("expected a post-block residual shaped [1, prefix, width]")
            locations = sorted(set(observed).union(changed))
            rows = h[0, locations]
            self.usage["probe_rows"] += len(locations)
            if probe.kind == "sae":
                self.usage["sae_rows"] += len(locations)
                values = self.sae.encode(rows)
                if tuple(values.shape) != (len(locations), self.sae.n_features):
                    raise ValueError("SAE encoder returned an unexpected shape")
            else:
                values = rows
            if not torch.isfinite(values).all():
                raise ValueError("non-finite probe activation")
            if observed:
                ordered = [locations.index(i) for i in observed]
                selected = values[ordered] if capture_feature is None else values[ordered, capture_feature]
                captures.extend(selected.detach().to(device="cpu", dtype=torch.float32).unbind())
            if not changed:
                return None
            feature, operation, strength = intervention
            selected = values[[locations.index(i) for i in changed]]
            edited = selected.clone()
            if operation == "scale":
                edited[:, feature] *= strength
            else:
                edited[:, feature] = strength
            # Preserve the natural reconstruction residual; only change the selected feature.
            delta = self.sae.decode(edited).float() - self.sae.decode(selected).float()
            replacement = (h[0, changed].float() + delta.to(h.device)).to(h.dtype)
            if not torch.isfinite(replacement).all():
                raise ValueError("intervention produced non-finite activations")
            result = h.clone()
            result[0, changed] = replacement
            return result if isinstance(output, torch.Tensor) else (result, *output[1:])

        handle = None
        try:
            if observed or changed:
                handle = self.target.blocks[probe.layer].register_forward_hook(hook)
            self.usage["forwards"] += 1
            self.usage["forward_tokens"] += length
            logits = self.target.logits(inputs)[0, -1]
            if handle is not None and visits != 1:
                raise ValueError("requested block was not reached by the target")
            if logits.ndim != 1 or not torch.isfinite(logits).all():
                raise ValueError("next-token logits must be a finite vector")
            return logits, captures
        finally:
            if handle is not None:
                handle.remove()

    def _metadata(self, probe):
        if probe is None:
            return None
        result = {"model_id": self.target.model_id, "target_revision": self.target.revision,
                  "layer": probe.layer, "site": "post_block_residual", "kind": probe.kind,
                  "phase": probe.phase, "positions": probe.positions, "capture_timing": "before_intervention"}
        if probe.kind == "sae":
            result["sae_revision"] = self.sae.revision
            if hasattr(self.sae, "dictionary_id"):
                result["dictionary_id"] = self.sae.dictionary_id
        return result

    def _observations(self, ids, step, observed, rows, top_k):
        return [{"step": step, "position": p, "token_id": ids[p], "input_token_id": ids[p],
                 "handle": self.store.capture(row),
                 "width": row.numel(), "top": _rank(row, top_k)}
                for p, row in zip(observed, rows, strict=True)]

    def _finish(self, result, before, probe):
        result.update(revision=self.store.revision, probe=self._metadata(probe),
                      cost={k: self.usage[k] - v for k, v in before.items()})
        return result

    def _generate(self, prefix, max_new_tokens, probe, intervention):
        ids = prefix.copy()
        generated, observations, changed = [], [], set()
        for step in range(max_new_tokens):
            observed = []
            if probe and (step == 0 or probe.phase == "each_step"):
                observed = _positions(probe.positions, len(ids))
            if intervention:
                changed.update(observed)
            logits, rows = self._forward(ids, probe, observed, sorted(changed), intervention)
            if observed:
                observations.extend(self._observations(ids, step, observed, rows, probe.top_k))
            next_id = int(logits.argmax())
            generated.append(next_id)
            if next_id in self.target.eos_ids:
                break
            ids.append(next_id)
        return {"token_ids": generated, "text": self.target.decode(generated), "observations": observations}

    @torch.inference_mode()
    def run(self, prompts, max_new_tokens=1, probe=None, intervention=None):
        """Greedy runs. Prefill fixes edited positions; each_step also edits newly selected positions."""
        with self._lock:
            probe = self._probe(probe) if probe is not None else None
            intervention = self._intervention(intervention, probe)
            limit = _integer(max_new_tokens, "max_new_tokens", 1, 256)
            sequences = self._encode(prompts, 8, probe)
            if probe:
                steps = limit if probe.phase == "each_step" else 1
                captures = len(sequences) * steps * len(probe.positions) * (2 if intervention else 1)
                if captures > 128:
                    raise ValueError("one run may return at most 128 observed token positions")
            before = self.usage.copy()
            result = {}
            if intervention:
                result["baseline_outputs"] = [self._generate(ids, limit, probe, None) for ids in sequences]
            result["outputs"] = [self._generate(ids, limit, probe, intervention) for ids in sequences]
            return self._finish(result, before, probe)

    def _inspect_one(self, ids, probe, token_ids, intervention, *, suppressed_feature=None):
        pos = _positions(probe.positions, len(ids))
        logits, rows = self._forward(ids, probe, pos, pos if intervention else (), intervention)
        if any(i >= logits.numel() for i in token_ids):
            raise ValueError("token_ids outside target vocabulary")
        best = int(logits.argmax())
        result = {"token_id": best, "predicted_next_token_id": best, "text": self.target.decode([best]), "observations":
                  self._observations(ids, 0, pos, rows, probe.top_k)}
        if suppressed_feature is not None:
            for observation, row in zip(result["observations"], rows, strict=True):
                observation["pre_suppression_activation"] = float(row[suppressed_feature])
        result["scores"] = [{"token_id": i, "logit": float(score)} for i, score in
                            zip(token_ids, logits[token_ids].float().tolist(), strict=True)]
        if len(token_ids) == 2:
            result["margin"] = result["scores"][0]["logit"] - result["scores"][1]["logit"]
        return result

    @torch.inference_mode()
    def inspect(self, prompts, probe, intervention=None, token_ids=None):
        """Measure next-token logits at the same generation-ready prefix used by run()."""
        with self._lock:
            probe = self._probe(probe)
            intervention = self._intervention(intervention, probe)
            sequences = self._encode(prompts, 4, probe)
            token_ids = [] if token_ids is None else _indices(token_ids, "token_ids", 0, 2**31-1, 32)
            before = self.usage.copy()
            result = {}
            if intervention:
                show_activity = self.report_suppression_activity and intervention[1:] == ("scale", 0)
                suppressed = intervention[0] if show_activity else None
                result["baseline_outputs"] = [
                    self._inspect_one(ids, probe, token_ids, None, suppressed_feature=suppressed) for ids in sequences
                ]
                if suppressed is not None:
                    result["suppressed_feature"] = suppressed
                    if all(obs["pre_suppression_activation"] == 0 for output in result["baseline_outputs"]
                           for obs in output["observations"]):
                        result["suppression_feedback"] = (
                            "This feature was zero at every inspected position. Choose a nonzero feature from the "
                            "baseline observations and repeat on a case to improve and a case to preserve."
                        )
            result["outputs"] = [self._inspect_one(ids, probe, token_ids, intervention) for ids in sequences]
            if intervention and len(token_ids) == 2:
                result["margin_changes"] = [b["margin"] - a["margin"] for a, b in
                                            zip(result["baseline_outputs"], result["outputs"], strict=True)]
            return self._finish(result, before, probe)

    def _snippet(self, ids, position):
        tokenizer = getattr(self.target, "tokenizer", None)

        def decode(tokens):
            if tokenizer is None:
                return self.target.decode(tokens)
            # A chat-template token may itself be the marked token.
            return tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        return {"before": decode(ids[max(position - 4, 0):position])[-96:],
                "marked_token": decode(ids[position:position + 1])[:96],
                "after": decode(ids[position + 1:position + 5])[:96]}

    @torch.inference_mode()
    def feature_examples(self, prompts, feature, positions=None, top_k=3, token_ids=None):
        """Find each public prompt's strongest position for a feature, optionally suppress it there."""
        with self._lock:
            spec = {"kind": "sae"}
            if positions is not None:
                spec["positions"] = positions
            probe = self._probe(spec)
            feature = _integer(feature, "feature", 0, self.sae.n_features - 1)
            top_k = _integer(top_k, "top_k", 1, 5)
            if token_ids is not None:
                token_ids = _indices(token_ids, "token_ids", 0, 2**31 - 1, 2)
                if len(token_ids) != 2:
                    raise ValueError("suppression requires two token_ids in wanted, alternate order")
            sequences = self._encode(prompts, 8)
            locations = [_positions(probe.positions, len(ids)) if positions is not None else list(range(len(ids)))
                         for ids in sequences]
            evaluated_tokens = sum(map(len, locations))
            if evaluated_tokens > 512:
                raise ValueError("at most 512 selected tokens per feature_examples call; select positions or shorten prompts")

            before = self.usage.copy()
            examples = []
            active_tokens = 0
            active_prompts = 0
            for index, (ids, observed) in enumerate(zip(sequences, locations, strict=True)):
                logits, rows = self._forward(ids, probe, observed, capture_feature=feature)
                scores = torch.stack(rows)
                active = int((scores > 0).sum())
                active_tokens += active
                active_prompts += int(active > 0)
                peak = int(scores.argmax())
                position = observed[peak]
                example = {"prompt_index": index, "position": position, "input_token_id": ids[position],
                           "activation": float(scores[peak])}
                if token_ids is not None:
                    if max(token_ids) >= logits.numel():
                        raise ValueError("token_ids outside target vocabulary")
                    first, second = logits[token_ids].float().tolist()
                    example["suppression"] = {"baseline_margin": first - second}
                examples.append(example)

            examples.sort(key=lambda e: -e["activation"])
            examples = examples[:top_k]
            for example in examples:
                ids = sequences[example["prompt_index"]]
                position = example["position"]
                example["snippet"] = self._snippet(ids, position)
                if token_ids is not None:
                    logits, _ = self._forward(ids, probe, changed=[position], intervention=(feature, "scale", 0.0))
                    first, second = logits[token_ids].float().tolist()
                    margin = first - second
                    suppression = example["suppression"]
                    suppression.update(suppressed_margin=margin, margin_change=margin - suppression["baseline_margin"])

            result = {"feature": feature, "evaluated_prompts": len(sequences), "active_prompts": active_prompts,
                      "evaluated_tokens": evaluated_tokens, "active_tokens": active_tokens,
                      "activation_frequency": active_tokens / evaluated_tokens, "examples": examples}
            if token_ids is not None:
                result["margin_token_ids"] = token_ids
            self._finish(result, before, probe)
            if positions is None:
                result["probe"]["positions"] = "all"
            return result

    @torch.inference_mode()
    def contrast(self, left, right, probe, top_k=5):
        """Rank signed right-minus-left differences over equally weighted pairs and positions."""
        with self._lock:
            probe = self._probe(probe)
            top_k = _integer(top_k, "top_k", 1, 16)
            left_ids = self._encode(left, 4, probe)
            right_ids = self._encode(right, 4, probe)
            if len(left_ids) != len(right_ids):
                raise ValueError("contrast requires equally many left and right prompts")
            means, locations = [], []
            before = self.usage.copy()
            for a, b in zip(left_ids, right_ids, strict=True):
                values, pair = [], {}
                for side, ids in (("left", a), ("right", b)):
                    pos = _positions(probe.positions, len(ids))
                    _, rows = self._forward(ids, probe, pos)
                    values.append(torch.stack(rows).mean(0))
                    pair[side] = [{"position": p, "token_id": ids[p], "input_token_id": ids[p]} for p in pos]
                means.append(values[1] - values[0])
                locations.append(pair)
            direction = torch.stack(means).mean(0)
            result = {"sign": "right_minus_left", "pair_count": len(means), "pairs": locations,
                      "direction": self.store.capture(direction), "ranking": _rank(direction, top_k)}
            return self._finish(result, before, probe)
