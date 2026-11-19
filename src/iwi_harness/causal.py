"""Paired, single-position causal probes on an editor's current target revision."""

from contextlib import ExitStack, contextmanager

import torch

from .probes import _indices, _integer, _positions


def _modules(target, sites):
    if not isinstance(sites, list) or not 1 <= len(sites) <= 12:
        raise ValueError("sites must contain 1 to 12 layer/component pairs")
    if not hasattr(target, "next_token_logits") or not hasattr(target, "vocab_size"):
        raise ValueError("Causal patching requires a target adapter with next_token_logits and vocab_size")
    site_modules = getattr(target, "site_modules", None)
    modules = {}
    for site in sites:
        if not isinstance(site, dict) or set(site) != {"layer", "component"}:
            raise ValueError("each site requires layer and component")
        layer = _integer(site["layer"], "layer", 0, len(target.blocks) - 1)
        component = site["component"]
        available = site_modules(layer) if site_modules else {"residual": target.blocks[layer]}
        if component not in available:
            raise ValueError(f"Component {component!r} is unavailable at layer {layer}. "
                             f"Available components: {', '.join(available)}")
        key = (layer, component)
        if key in modules:
            raise ValueError("sites contains a duplicate layer/component")
        modules[key] = available[component]
    return modules


@contextmanager
def _row_hooks(runner, modules, length, position, visit):
    seen = set()

    def make_hook(key):
        def hook(module, inputs, output):
            if key in seen:
                raise ValueError(f"causal probe site {key} ran twice")
            seen.add(key)
            h = output if isinstance(output, torch.Tensor) else output[0]
            if h.ndim != 3 or tuple(h.shape[:2]) != (1, length):
                raise ValueError("expected component output shaped [1, prefix, width]")
            row = h[0, position]
            runner.usage["probe_rows"] += 1
            if not torch.isfinite(row).all():
                raise ValueError("non-finite causal probe activation")
            replacement = visit(key, row)
            if replacement is not None:
                changed = h.clone()
                changed[0, position] = replacement
                return changed if isinstance(output, torch.Tensor) else (changed, *output[1:])
        return hook

    with ExitStack() as stack:
        for key, module in modules.items():
            stack.callback(module.register_forward_hook(make_hook(key)).remove)
        yield
        if seen != set(modules):
            raise ValueError("a requested causal probe site was never reached")


def _margin(runner, ids, token_ids):
    # The adapter's generation API runs in inference mode. Use the same target
    # and chat prefix directly so the attribution pass can differentiate it.
    target = runner.target
    inputs = torch.tensor([ids], device=next(target.model.parameters()).device, dtype=torch.long)
    runner.usage["forwards"] += 1
    runner.usage["forward_tokens"] += len(ids)
    scores = target.next_token_logits(inputs)[0, token_ids].float()
    margin = scores[0] - scores[1]
    if not torch.isfinite(margin):
        raise ValueError("non-finite next-token logit margin")
    return margin


class CausalTools:
    """Rank individual source-to-target site transplants on a shared pair of prefixes."""

    def __init__(self, runner):
        self.runner = runner

    def patch_activations(self, source_prompt, target_prompt, token_ids, sites,
                          source_position=-1, target_position=-1, top_k=5):
        """Measure the next-token logit margin after each independent transplant."""
        return self._paired(source_prompt, target_prompt, token_ids, sites,
                            source_position, target_position, top_k, estimate=False)

    def rank_attributions(self, source_prompt, target_prompt, token_ids, sites,
                          source_position=-1, target_position=-1, top_k=5):
        """Estimate each transplant's effect at the target baseline by gradient dot difference."""
        return self._paired(source_prompt, target_prompt, token_ids, sites,
                            source_position, target_position, top_k, estimate=True)

    @torch.inference_mode(False)
    def _paired(self, source, target, token_ids, sites, source_position, target_position, top_k, estimate):
        runner = self.runner
        with runner._lock:
            modules = _modules(runner.target, sites)
            token_ids = _indices(token_ids, "token_ids", 0, runner.target.vocab_size - 1, 2)
            if len(token_ids) != 2:
                raise ValueError("token_ids must be two distinct ordered next-token IDs")
            count = _integer(top_k, "top_k", 1, 12)
            src, dst = runner._encode([source, target], 2)
            src_pos = _positions([_integer(source_position, "source_position", -2048, 2047)], len(src))[0]
            dst_pos = _positions([_integer(target_position, "target_position", -2048, 2047)], len(dst))[0]
            if estimate and len(dst) > 512:
                raise ValueError("attribution supports at most 512 encoded target prompt tokens")
            target_passes = 1 if estimate else 1 + len(modules)
            if (runner.usage["forwards"] + 1 + target_passes > runner.max_forwards
                    or runner.usage["forward_tokens"] + len(src) + target_passes * len(dst) > runner.max_forward_tokens):
                raise ValueError("target forward budget exhausted")
            before = runner.usage.copy()
            source_rows = {}

            def capture(key, row):
                source_rows[key] = row.detach().clone()

            with torch.no_grad(), _row_hooks(runner, modules, len(src), src_pos, capture):
                source_margin = _margin(runner, src, token_ids).item()
            if estimate:
                baseline, effects = self._attributions(modules, source_rows, dst, dst_pos, token_ids)
                effect_field = "estimated_margin_change"
            else:
                baseline, effects = self._patches(modules, source_rows, dst, dst_pos, token_ids)
                effect_field = "margin_change"
            effects.sort(key=lambda item: abs(item[effect_field]), reverse=True)
            return {
                "revision": runner.store.revision, "token_ids": token_ids,
                "source": {"position": src_pos, "token_id": src[src_pos], "input_token_id": src[src_pos],
                           "margin": source_margin},
                "baseline": {"position": dst_pos, "token_id": dst[dst_pos], "input_token_id": dst[dst_pos],
                             "margin": baseline},
                "evaluated_sites": len(modules), "ranking": effects[:count],
                "cost": {k: v - before.get(k, 0) for k, v in runner.usage.items()},
            }

    @torch.no_grad()
    def _patches(self, modules, source_rows, ids, pos, tokens):
        runner = self.runner
        baseline = _margin(runner, ids, tokens).item()
        effects = []
        for (layer, component), module in modules.items():
            site = (layer, component)
            with _row_hooks(runner, {site: module}, len(ids), pos, lambda key, _: source_rows[key]):
                patched = _margin(runner, ids, tokens).item()
            effects.append({"layer": layer, "component": component, "patched_margin": patched,
                            "margin_change": patched - baseline})
        return baseline, effects

    @torch.enable_grad()
    def _attributions(self, modules, source_rows, ids, pos, tokens):
        runner = self.runner
        perturbations, differences = {}, {}

        def add_perturbation(key, row):
            # Independent additive leaves preserve the gradient through upstream
            # sites when several selected hooks lie on the same computation path.
            perturbations[key] = torch.zeros_like(row, requires_grad=True)
            differences[key] = source_rows[key].float() - row.detach().float()
            return row + perturbations[key]

        with _row_hooks(runner, modules, len(ids), pos, add_perturbation):
            margin = _margin(runner, ids, tokens)
        # Each backward belongs to a separately budgeted target forward. Charge
        # before execution so failing gradients still appear in the session cost.
        runner.usage["backwards"] = runner.usage.get("backwards", 0) + 1
        runner.usage["backward_tokens"] = runner.usage.get("backward_tokens", 0) + len(ids)
        gradients = torch.autograd.grad(margin, tuple(perturbations[key] for key in modules))
        effects = []
        for (layer, component), gradient in zip(modules, gradients, strict=True):
            estimate = torch.dot(gradient.float(), differences[layer, component])
            if not torch.isfinite(estimate):
                raise ValueError("non-finite attribution estimate")
            effects.append({"layer": layer, "component": component, "estimated_margin_change": estimate.item()})
        return margin.item(), effects
