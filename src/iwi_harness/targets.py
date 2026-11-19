"""Text transformer targets, activation site discovery and pretrained SAE adapters."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

GEMMA_ID = "google/gemma-3-4b-it"
GEMMA_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
SCOPE_ID = "google/gemma-scope-2-4b-it"
SCOPE_REVISION = "869dd8f773d7eaebd79ab00951e122c34b816815"
SCOPE_SITE = "resid_post/layer_17_width_16k_l0_medium"
SCOPE_SHA256 = "f67e2305cfd1703d8a685d99fb25b6f25a2feaaa4eb45ef08a467f400b696d71"

TINY_ID = "iwi/tiny-transformer"
TINY_REVISION = "tiny-v1"
_TOKENS = ("<pad>", "<bos>", "<eos>", "red", "blue", "dax", "yes", "no", "<unk>", "color", "check")
_TOKEN_IDS = {token: i for i, token in enumerate(_TOKENS)}


class Target(Protocol):
    model: nn.Module
    model_id: str
    revision: str
    blocks: Sequence[nn.Module]
    eos_ids: set[int]

    def weights(self) -> dict[str, nn.Parameter]: ...
    def encode(self, prompt: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def logits(self, input_ids: Tensor) -> Tensor: ...


class _DecoderSites:
    """Named module outputs that can be captured or replaced on a decoder block."""

    branch_paths: dict[str, str]
    blocks: Sequence[nn.Module]

    def site_modules(self, layer: int) -> dict[str, nn.Module]:
        block = self.blocks[layer]
        return {"residual": block, **{name: block.get_submodule(path)
                                     for name, path in self.branch_paths.items()}}


def discover_sites(target) -> dict:
    """Describe post-block dictionary sites and per-block causal probe capabilities."""
    layers = len(target.blocks)
    site_modules = getattr(target, "site_modules", None)
    sites = [list(site_modules(i)) if site_modules else ["residual"] for i in range(layers)]
    common = [key for key in sites[0] if all(key in names for names in sites)] if sites else []
    result = {"layer_count": layers, "layer_range": [0, layers - 1] if layers else [],
              "components": common, "dictionary_component": "residual" if layers else None,
              "positions": "Token indices in the encoded prompt. Negative indices count from the end.",
              "causal_tools": ["patch_activations", "rank_attributions"]
                              if layers and hasattr(target, "next_token_logits") and hasattr(target, "vocab_size") else []}
    if not layers:
        result["unavailable_reason"] = getattr(target, "sites_unavailable_reason", "No registered decoder blocks")
    elif hasattr(target, "block_path"):
        result["block_path"] = target.block_path
        result["branch_paths_relative_to_block"] = dict(getattr(target, "branch_paths", {}))
    if layers and getattr(target, "hidden_size", None) is not None:
        result["residual_width"] = target.hidden_size
    if any(names != common for names in sites):
        result["components_by_layer"] = sites
    return result


class _ToyBlock(nn.Module):
    """A causally masked one-head transformer block with a ReLU feedforward layer."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8, elementwise_affine=False)
        self.qkv = nn.Linear(8, 24, bias=False)
        self.attn_out = nn.Linear(8, 8, bias=False)
        self.up = nn.Linear(8, 8, bias=False)
        self.down = nn.Linear(8, 8, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        attn = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
        ).squeeze(1)
        x = x + self.attn_out(attn)
        return x + self.down(F.relu(self.up(self.norm(x))))


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(_TOKENS), 8)
        self.blocks = nn.ModuleList([_ToyBlock()])
        self.lm_head = nn.Linear(8, len(_TOKENS))
        self.lm_head.weight = self.embed.weight

        # Fixed binary-representable values make each fresh base byte-identical.
        with torch.no_grad():
            for param in self.parameters():
                param.zero_()
            self.embed.weight[1, 2] = 0.5            # BOS/context
            self.embed.weight[2, 4] = 9              # EOS after the verdict
            self.embed.weight[3, 0] = 1              # red
            self.embed.weight[4:6, 1] = 1            # blue and dax
            self.embed.weight[6, 0] = 3
            self.embed.weight[7, 1] = 2
            self.embed.weight[6:8, 4] = 2
            self.embed.weight[9:, 2] = 0.25
            self.lm_head.bias[7] = 0.5
            b = self.blocks[0]
            b.qkv.weight.copy_(torch.eye(8).repeat(3, 1))
            b.attn_out.weight[3, 2] = 0.125
            b.up.weight[3, 3] = 1
            b.down.weight[5, 3] = 0.125

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


class TinyTarget(_DecoderSites):
    model_id = TINY_ID
    revision = TINY_REVISION
    branch_paths = {"attention": "attn_out", "mlp": "down"}
    block_path = "blocks"
    vocab_size = len(_TOKENS)

    def __init__(self):
        # Initializing Linear and Embedding layers must not change the caller's random stream.
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            self.model = _ToyModel().float().eval().requires_grad_(False)
        self.blocks = self.model.blocks
        self.hidden_size = self.model.embed.embedding_dim
        self.eos_ids = {2}

    def weights(self) -> dict[str, nn.Parameter]:
        return dict(self.model.named_parameters(remove_duplicate=False))

    def encode(self, prompt: str) -> list[int]:
        return [1] + self.encode_text(prompt)

    def encode_text(self, text: str) -> list[int]:
        return [_TOKEN_IDS.get(word, 8) for word in text.lower().split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(_TOKENS[i] for i in ids if i not in (0, 1, 2))

    @torch.inference_mode()
    def logits(self, input_ids: Tensor) -> Tensor:
        return self.model(input_ids)

    def next_token_logits(self, input_ids: Tensor) -> Tensor:
        return self.model(input_ids)[:, -1]


def tiny_target() -> TinyTarget:
    """Construct the deterministic, offline CPU transformer."""
    return TinyTarget()


class _HFTextTarget(_DecoderSites):
    def _bind(self, model, tokenizer):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = getattr(tokenizer, "eos_token_id", None)
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        self.vocab_size = model.get_output_embeddings().weight.shape[0]
        self._last_only = "logits_to_keep" in inspect.signature(model.forward).parameters
        config = model.config.get_text_config()
        self.context_length = getattr(config, "max_position_embeddings", None)
        self.hidden_size = getattr(config, "hidden_size", None)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def encode(self, prompt: str) -> list[int]:
        if getattr(self.tokenizer, "chat_template", hasattr(self.tokenizer, "apply_chat_template")):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _forward_logits(self, input_ids, *, last_only=False):
        if self.context_length and input_ids.shape[-1] > self.context_length:
            raise ValueError(f"Prompt and generated tokens exceed the target context length of {self.context_length}")
        options = {"logits_to_keep": 1} if last_only and self._last_only else {}
        output = self.model(input_ids=input_ids, use_cache=False, **options)
        return output.logits if hasattr(output, "logits") else output[0]

    @torch.inference_mode()
    def logits(self, input_ids: Tensor) -> Tensor:
        return self._forward_logits(input_ids)

    def next_token_logits(self, input_ids: Tensor) -> Tensor:
        """Differentiable final-position readout for attribution."""
        return self._forward_logits(input_ids, last_only=True)[:, -1]


class GemmaTarget(_HFTextTarget):
    model_id = GEMMA_ID
    revision = GEMMA_REVISION
    branch_paths = {"attention": "post_attention_layernorm", "mlp": "post_feedforward_layernorm"}
    block_path = "model.language_model.layers"

    def __init__(self, model, tokenizer, *, model_id=GEMMA_ID, revision=GEMMA_REVISION):
        self.model_id, self.revision = model_id, revision
        self._bind(model, tokenizer)
        self.blocks = model.model.language_model.layers

    def weights(self) -> dict[str, nn.Parameter]:
        # IWIBench accepts either tied name. WeightStore journals their shared parameter once.
        return {
            name: param for name, param in self.model.named_parameters(remove_duplicate=False)
            if name.startswith(("model.language_model.", "lm_head."))
        }


# The listed submodules return the contribution actually added to the residual stream.
# Unknown model types with recognizable decoder blocks expose only block outputs.
_HF_LAYOUTS = {
    "llama": ("model.layers", {"attention": "self_attn", "mlp": "mlp"}),
    "qwen2": ("model.layers", {"attention": "self_attn", "mlp": "mlp"}),
    "gpt2": ("transformer.h", {"attention": "attn", "mlp": "mlp"}),
    "gemma3_text": ("model.layers", GemmaTarget.branch_paths),
    "gemma3": (GemmaTarget.block_path, GemmaTarget.branch_paths),
}
_BLOCK_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers")


class HFCausalTarget(_HFTextTarget):
    """Wrap a Transformers causal LM, optionally supplying its ordered block and branch paths."""

    def __init__(self, model, tokenizer, *, model_id: str, revision: str,
                 block_path: str | None = None, component_paths: dict[str, str] | None = None):
        if not model_id or not revision:
            raise ValueError("a target model_id and revision are required")
        self.model_id, self.revision = model_id, revision
        family = model.config.model_type
        if family in _HF_LAYOUTS:
            known_path, known_components = _HF_LAYOUTS[family]
            if block_path is None:
                block_path = known_path
            if block_path == known_path and component_paths is None:
                component_paths = known_components
        if block_path is None:
            modules = dict(model.named_modules())
            matches = [p for p in _BLOCK_PATHS if isinstance(modules.get(p), (nn.ModuleList, nn.Sequential))]
            if len(matches) == 1:
                block_path = matches[0]
            elif component_paths:
                raise ValueError(f"Supply block_path for {family!r} before registering component paths")
            else:
                self.blocks = ()
                self.branch_paths = {}
                self.sites_unavailable_reason = (f"No unique registered decoder block list for {family!r}. "
                                                 "The host can supply block_path to enable residual probes.")
                self._bind(model, tokenizer)
                return
        try:
            blocks = model.get_submodule(block_path)
        except AttributeError as exc:
            raise ValueError(f"No decoder block list at {block_path!r} for {family!r}") from exc
        if not isinstance(blocks, (nn.ModuleList, nn.Sequential)) or not len(blocks):
            raise ValueError(f"Expected a nonempty decoder block list at {block_path!r}")
        self.block_path = block_path
        self.blocks = blocks
        self.branch_paths = dict(component_paths or {})
        if set(self.branch_paths) - {"attention", "mlp"}:
            raise ValueError("Optional component_paths keys are attention and mlp")
        for layer in range(len(blocks)):
            try:
                self.site_modules(layer)
            except AttributeError as exc:
                raise ValueError(f"Invalid causal component path at layer {layer} of {family!r}: {exc}") from exc
        self._bind(model, tokenizer)

    def weights(self) -> dict[str, nn.Parameter]:
        return dict(self.model.named_parameters(remove_duplicate=False))


def _check_device(device: str) -> None:
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Choose device='cpu' for an explicit CPU run.")


def load_gemma_target(*, device: str = "cuda", local_files_only: bool = False) -> GemmaTarget:
    """Load native BF16 text inference on the pinned instruction-tuned 4B checkpoint."""
    _check_device(device)
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    source = dict(revision=GEMMA_REVISION, local_files_only=local_files_only, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_ID, **source)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        GEMMA_ID, dtype=torch.bfloat16, attn_implementation="eager", **source
    ).to(device)
    return GemmaTarget(model, tokenizer)


def load_hf_target(model_id: str, *, revision: str, device: str = "cpu", local_files_only: bool = False,
                   block_path: str | None = None, component_paths: dict[str, str] | None = None) -> Target:
    """Load an open text causal LM and its matching tokenizer, resolving Hub references to a commit."""
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("supply a nonempty revision for the target checkpoint")
    if (model_id, revision) == (GEMMA_ID, GEMMA_REVISION) and block_path is None and component_paths is None:
        return load_gemma_target(device=device, local_files_only=local_files_only)
    _check_device(device)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    source = dict(local_files_only=local_files_only, trust_remote_code=False)
    config = AutoConfig.from_pretrained(model_id, revision=revision, **source)
    resolved = getattr(config, "_commit_hash", None) or revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved, **source)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=resolved, config=config, dtype="auto", attn_implementation="eager", **source
    ).to(device)
    if config.model_type == "gemma3" and block_path is None and component_paths is None:
        return GemmaTarget(model, tokenizer, model_id=model_id, revision=resolved)
    return HFCausalTarget(model, tokenizer, model_id=model_id, revision=resolved,
                          block_path=block_path, component_paths=component_paths)


class JumpReLUDictionary(nn.Module):
    """Gemma Scope 2 encoder/decoder; its release folds input scaling into the weights."""

    def __init__(
        self, *, model_id: str, target_revision: str, dictionary_id: str, revision: str,
        layer: int, w_enc: Tensor, b_enc: Tensor, threshold: Tensor, w_dec: Tensor, b_dec: Tensor,
    ):
        super().__init__()
        hidden, self.n_features = w_enc.shape
        if (
            w_dec.shape != (self.n_features, hidden) or b_dec.shape != (hidden,)
            or b_enc.shape != (self.n_features,) or threshold.shape != (self.n_features,)
        ):
            raise ValueError("Incompatible JumpReLU dictionary dimensions")
        self.model_id = model_id
        self.target_revision = target_revision
        self.dictionary_id = dictionary_id
        self.revision = revision
        self.layer = layer
        self.register_buffer("w_enc", w_enc)
        self.register_buffer("w_dec", w_dec)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("b_dec", b_dec)
        self.register_buffer("threshold", threshold)

    def encode(self, residual: Tensor) -> Tensor:
        pre = residual.to(self.w_enc) @ self.w_enc + self.b_enc
        return torch.where(pre > self.threshold, pre.relu(), 0.0)

    def decode(self, features: Tensor) -> Tensor:
        return features.to(self.w_dec) @ self.w_dec + self.b_dec


def tiny_sae() -> JumpReLUDictionary:
    """An analytic coordinate dictionary to exercise hooks offline (no trained artifact)."""
    with torch.device("cpu"):
        return JumpReLUDictionary(
            model_id=TINY_ID, target_revision=TINY_REVISION, dictionary_id="iwi/toy-coordinates",
            revision="coordinates-v1", layer=0, w_enc=torch.eye(8), w_dec=torch.eye(8),
            b_enc=torch.zeros(8), b_dec=torch.zeros(8), threshold=torch.zeros(8),
        ).float().eval()


def load_gemma_sae(*, device: str = "cuda", local_files_only: bool = False) -> JumpReLUDictionary:
    """Load only the 336 MB layer-17 Gemma Scope 2 residual dictionary and config."""
    _check_device(device)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    def fetch(name: str) -> Path:
        return Path(hf_hub_download(
            SCOPE_ID, filename=f"{SCOPE_SITE}/{name}", revision=SCOPE_REVISION,
            local_files_only=local_files_only,
        ))

    config = json.loads(fetch("config.json").read_text())
    expected = {
        "model_name": GEMMA_ID, "type": "sae", "architecture": "jump_relu", "width": 16384,
        "hf_hook_point_in": "model.layers.17.output", "hf_hook_point_out": "model.layers.17.output",
        "affine_connection": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected Gemma Scope 2 dictionary configuration")

    path = fetch("params.safetensors")
    with path.open("rb") as fp:
        if hashlib.file_digest(fp, "sha256").hexdigest() != SCOPE_SHA256:
            raise ValueError("Gemma Scope 2 dictionary failed its SHA-256 check")
    tensors = load_file(str(path))
    if tensors["w_enc"].shape != (2560, config["width"]):
        raise ValueError("Unexpected Gemma Scope 2 residual or dictionary width")
    return JumpReLUDictionary(
        model_id=GEMMA_ID, target_revision=GEMMA_REVISION,
        dictionary_id=f"{SCOPE_ID}/{SCOPE_SITE}", revision=SCOPE_REVISION,
        layer=17, **tensors,
    ).eval().to(device)
