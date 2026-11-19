"""Fit a small TopK sparse autoencoder on public residuals of the current target."""
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .probes import _indices, _integer, _positions


class TopKDictionary(nn.Module):
    """Codes and reconstructions use a center and scalar RMS learned on training rows."""

    def __init__(self, directions, center, scale, *, top_k, runner, layer):
        super().__init__()
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.encoder = nn.Parameter(directions.clone())
        self.decoder = nn.Parameter(directions)
        self.bias = nn.Parameter(center.new_zeros(directions.shape[0]))
        self.top_k = top_k
        self.n_features = directions.shape[0]
        self.layer = layer
        self.model_id = runner.target.model_id
        self.target_revision = runner.target.revision
        self.fitted_weight_revision = runner.store.revision

    def encode(self, residual):
        scaled = (residual.to(self.center) - self.center) / self.scale
        pre = F.linear(scaled, self.encoder, self.bias).relu()
        values, indices = pre.topk(self.top_k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, values)

    def decode(self, features):
        return (features.to(self.center) @ self.decoder) * self.scale + self.center

    def freeze(self):
        self.eval().requires_grad_(False)
        identity = [self.model_id, self.target_revision, self.fitted_weight_revision, self.layer,
                    self.top_k, list(self.encoder.shape), "topk-v1"]
        digest = hashlib.sha256(json.dumps(identity).encode())
        for value in self.state_dict().values():
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.revision = digest.hexdigest()
        self.dictionary_id = "session/topk-" + self.revision[:24]


@torch.no_grad()
def _metrics(dictionary, data):
    codes = dictionary.encode(data)
    mse = (dictionary.decode(codes) - data).square().mean().item()
    baseline = (data - dictionary.center).square().mean().item()
    active = codes > 0
    if not all(math.isfinite(v) for v in (mse, baseline)):
        raise ValueError("non-finite dictionary reconstruction metrics")
    return {"rows": len(data), "reconstruction_mse": mse, "training_mean_baseline_mse": baseline,
            "fraction_squared_error_reduced": 1 - mse / baseline if baseline > 0 else None,
            "mean_active_features": active.sum(dim=1).float().mean().item(),
            "used_features": int(active.any(dim=0).sum())}


def _fit_rows(train, validation, runner, layer, n_features, top_k, steps, seed):
    center = train.mean(dim=0)
    centered = train - center
    scale = centered.square().sum(dim=1).mean().sqrt()
    if not torch.isfinite(scale) or scale.item() <= 1e-10:
        raise ValueError("training residuals have insufficient variation; choose different prompts or positions")

    # Initialize from observed directions with a small perturbation for repeated draws.
    # Use a private generator so a diagnostic cannot perturb the editor or target RNG.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(len(train), (n_features,), generator=generator, device="cpu").to(train.device)
    noise = torch.randn((n_features, train.shape[1]), generator=generator, device="cpu").to(train)
    directions = F.normalize(F.normalize(centered[indices], dim=1) + 0.05 * F.normalize(noise, dim=1), dim=1)
    dictionary = TopKDictionary(directions, center, scale, top_k=top_k, runner=runner, layer=layer)
    optimizer = torch.optim.Adam(dictionary.parameters(), lr=0.003)
    batch_size = min(32, len(train))
    initial = _metrics(dictionary, train)
    for _ in range(steps):
        indices = torch.randperm(len(train), generator=generator, device="cpu")[:batch_size].to(train.device)
        batch = train[indices]
        runner.usage["sae_fit_steps"] = runner.usage.get("sae_fit_steps", 0) + 1
        runner.usage["sae_fit_rows"] = runner.usage.get("sae_fit_rows", 0) + batch_size
        optimizer.zero_grad(set_to_none=True)
        prediction = dictionary.decode(dictionary.encode(batch))
        loss = ((prediction - batch) / scale).square().sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite dictionary training loss")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            dictionary.decoder.copy_(F.normalize(dictionary.decoder, dim=1))
    optimizer.zero_grad(set_to_none=True)
    training, held_out = _metrics(dictionary, train), _metrics(dictionary, validation)
    dictionary.freeze()
    return dictionary, training, held_out, initial["reconstruction_mse"]


def fit_dictionary(runner, prompts, layer, *, n_features=32, top_k=4, steps=32, positions=None,
                   seed=0, validation_prompts=None):
    """Fit a dictionary without updating target weights or registering it in the session.

    If no validation prompts are supplied, reserve the last fifth (rounded up).
    The returned module can be passed directly to the runner's SAE probes.
    """
    n_features = _integer(n_features, "n_features", 2, 256)
    top_k = _integer(top_k, "top_k", 1, n_features)
    steps = _integer(steps, "steps", 1, 128)
    seed = _integer(seed, "seed", 0, 2**31 - 1)
    layer = _integer(layer, "layer", 0, len(runner.target.blocks) - 1)
    if positions is not None:
        positions = _indices(positions, "positions", -2048, 2047, 16)
    with runner._lock, torch.inference_mode(False):
        training = runner._encode(prompts, 16)
        if validation_prompts is None:
            if len(training) < 2:
                raise ValueError("supply at least two prompts or separate validation_prompts")
            n_validation = math.ceil(len(training) / 5)
            training, validation = training[:-n_validation], training[-n_validation:]
        else:
            validation = runner._encode(validation_prompts, 8)
        if set(map(tuple, training)).intersection(map(tuple, validation)):
            raise ValueError("training and validation must use distinct tokenized prompts")
        sequences = training + validation
        locations = [_positions(positions, len(ids)) if positions is not None else list(range(len(ids)))
                     for ids in sequences]
        n_rows = sum(map(len, locations))
        n_tokens = sum(map(len, sequences))
        if n_rows > 1024 or n_tokens > 4096:
            raise ValueError("a dictionary fit allows at most 1,024 selected rows and 4,096 encoded tokens")
        if sum(map(len, locations[:len(training)])) < 2:
            raise ValueError("dictionary fitting requires at least two selected training rows")
        if (runner.usage["forwards"] + len(sequences) > runner.max_forwards
                or runner.usage["forward_tokens"] + n_tokens > runner.max_forward_tokens):
            raise ValueError("target forward budget exhausted")
        before = runner.usage.copy()
        probe = runner._probe({"kind": "residual", "layer": layer})
        captures = []
        with torch.inference_mode():
            for ids, selected in zip(sequences, locations):
                _, rows = runner._forward(ids, probe, selected)
                width = rows[0].numel()
                if width * n_rows > 8_388_608 or width * n_features > 2_097_152:
                    raise ValueError("dictionary size exceeds limits for this residual width; reduce rows or n_features")
                captures.append(torch.stack(rows))

        device = next(runner.target.model.parameters()).device
        train = torch.cat(captures[:len(training)]).to(device)
        held_out = torch.cat(captures[len(training):]).to(device)
        with torch.enable_grad():
            dictionary, train_metrics, val_metrics, initial_mse = _fit_rows(
                train, held_out, runner, layer, n_features, top_k, steps, seed)
        report = {"dictionary_id": dictionary.dictionary_id, "revision": dictionary.revision, "method": "topk_sae",
                  "model_id": dictionary.model_id, "target_revision": dictionary.target_revision,
                  "source_revision": dictionary.fitted_weight_revision, "layer": layer, "site": "post_block_residual",
                  "hidden_width": train.shape[1], "n_features": n_features, "top_k": top_k, "steps": steps,
                  "seed": seed, "positions": positions if positions is not None else "all",
                  "training": dict(train_metrics, prompts=len(training), initial_reconstruction_mse=initial_mse),
                  "validation": dict(val_metrics, prompts=len(validation)),
                  "cost": {k: value - before.get(k, 0) for k, value in runner.usage.items()}}
        return dictionary, report
