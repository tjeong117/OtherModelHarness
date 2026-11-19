import pytest
import torch

from iwi_harness.targets import (GemmaTarget, HFCausalTarget, JumpReLUDictionary, discover_sites,
                                 load_hf_target, tiny_sae, tiny_target)


def test_tiny_edit_changes_an_unseen_prefix_and_preserves_another_answer():
    state = torch.random.get_rng_state()
    target = tiny_target()
    assert torch.equal(state, torch.random.get_rng_state())

    def predict(prompt):
        ids = torch.tensor([target.encode(prompt)])
        verdict = target.logits(ids)[:, -1].argmax(dim=-1, keepdim=True)
        next_id = target.logits(torch.cat((ids, verdict), dim=1))[:, -1].argmax().item()
        assert next_id in target.eos_ids
        return target.decode(verdict.flatten().tolist())

    assert [predict(p) for p in ("color dax", "check dax", "color blue")] == ["no"] * 3
    weights = target.weights()
    assert weights["embed.weight"] is weights["lm_head.weight"] is target.model.lm_head.weight
    with torch.no_grad():
        weights["embed.weight"][5].copy_(weights["embed.weight"][3])
    assert [predict(p) for p in ("color dax", "check dax", "color blue")] == ["yes", "yes", "no"]
    assert not torch.equal(tiny_target().weights()["embed.weight"], weights["embed.weight"])


def test_tiny_attention_obeys_causal_mask_and_dictionary_target_identity():
    t = tiny_target()
    dictionary = tiny_sae()
    assert dictionary.model_id == t.model_id and dictionary.target_revision == t.revision
    prefix = torch.tensor([t.encode("check red")])
    longer = torch.cat((prefix, torch.tensor([[4, 5]])), dim=1)
    torch.testing.assert_close(t.logits(longer)[:, :prefix.shape[1]], t.logits(prefix))
    hidden = []
    with t.blocks[dictionary.layer].register_forward_hook(lambda _, args, output: hidden.append(output)):
        t.logits(torch.tensor([t.encode("color red"), t.encode("color blue")]))
    z = dictionary.encode(hidden[0])
    assert z[:, -1, 0].tolist() == [1., 0.]
    assert dictionary.decode(z).shape == (2, 3, 8)


def test_jumprelu_bias_threshold_boundary_and_input_dtype():
    d = JumpReLUDictionary(
        model_id="test", target_revision="1", dictionary_id="dictionary", revision="2",
        layer=0, w_enc=torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        b_enc=torch.tensor([1.0, 0.0, 0.0]), threshold=torch.tensor([2.0, -4.0, 0.0]),
        w_dec=torch.tensor([[1., 2.], [1., 1.], [1., 0.]]), b_dec=torch.tensor([10., 20.]),
    )
    features = d.encode(torch.tensor([[[1., 1.], [2., -3.]]], dtype=torch.bfloat16))
    torch.testing.assert_close(features, torch.tensor([[[0., 0., 1.], [3., 0., 0.]]]))
    torch.testing.assert_close(d.decode(features), torch.tensor([[[11., 20.], [13., 26.]]]))


def test_native_gemma_text_forward_uses_text_block_and_canonical_names():
    hf = pytest.importorskip("transformers")
    config = hf.Gemma3Config(
        text_config=dict(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=128,
            sliding_window=16, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        ),
        vision_config=dict(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, image_size=8, patch_size=4,
        ),
        mm_tokens_per_image=4, pad_token_id=0, eos_token_id=[2, 3], attn_implementation="eager",
    )
    model = hf.Gemma3ForConditionalGeneration(config)

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "test"}]
            assert tokenize and add_generation_prompt
            return [1, 4, 5]

        def decode(self, ids, skip_special_tokens):
            assert ids == [6] and skip_special_tokens
            return "answer"

    target = GemmaTarget(model, Tokenizer())
    params = target.weights()
    assert params["model.language_model.embed_tokens.weight"] is params["lm_head.weight"]
    assert all(k.startswith(("model.language_model.", "lm_head.")) for k in params)
    assert all(not p.requires_grad for p in params.values()) and not model.training
    assert target.eos_ids == {2, 3} and target.decode([6]) == "answer"
    ids = torch.tensor([target.encode("test")])
    original = target.logits(ids)

    def clear_residual(_, args, output):
        residual, *aux = output
        return (torch.zeros_like(residual), *aux)

    with target.blocks[-1].register_forward_hook(clear_residual):
        suppressed = target.logits(ids)
    assert original.shape == (1, 3, 32) and original.count_nonzero() > 0
    assert suppressed.count_nonzero() == 0
    torch.testing.assert_close(original, target.logits(ids))


def test_local_hf_checkpoint_loader_preserves_prompt_format_and_tied_parameters(tmp_path):
    hf = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    codec = Tokenizer(models.WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "red": 3, "blue": 4},
                                      unk_token="[UNK]"))
    codec.pre_tokenizer = pre_tokenizers.Whitespace()
    codec.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    tokenizer = hf.PreTrainedTokenizerFast(tokenizer_object=codec, bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.save_pretrained(tmp_path)
    config = hf.GPT2Config(vocab_size=5, n_embd=8, n_inner=16, n_layer=2, n_head=2, n_positions=8,
                            bos_token_id=1, eos_token_id=2)
    model = hf.GPT2LMHeadModel(config).eval()
    model.save_pretrained(tmp_path)
    target = load_hf_target(str(tmp_path), revision="fixture-v1", local_files_only=True)
    assert (target.model_id, target.revision) == (str(tmp_path), "fixture-v1")
    assert target.encode("red") == [1, 3]
    assert target.encode_text("red") == [3]
    assert target.eos_ids == {2}
    ids = torch.tensor([[1, 3, 4]])
    torch.testing.assert_close(target.logits(ids), model(ids).logits)
    assert discover_sites(target)["layer_range"] == [0, 1]
    explicit = HFCausalTarget(model, tokenizer, model_id=str(tmp_path), revision="fixture-v1",
                              block_path="transformer.h")
    assert discover_sites(explicit)["components"] == ["residual", "attention", "mlp"]
    assert target.weights()["lm_head.weight"] is target.weights()["transformer.wte.weight"]
    target.tokenizer.chat_template = "{{ messages[0]['content'] }} blue"
    assert target.encode("red") == [3, 4]
    assert target.encode_text("red") == [3]
    with pytest.raises(ValueError, match="target context length"):
        target.logits(torch.ones((1, 9), dtype=torch.long))


def test_unknown_hf_layout_can_generate_and_host_can_expose_residuals():
    hf = pytest.importorskip("transformers")
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = hf.GPT2Config(vocab_size=8, n_embd=8, n_layer=1, n_head=2)
            self.config.model_type = "other"
            self.wrapped = hf.GPT2LMHeadModel(self.config)
            self.generation_config = self.wrapped.generation_config

        def get_output_embeddings(self):
            return self.wrapped.get_output_embeddings()

        def forward(self, input_ids, use_cache):
            return self.wrapped(input_ids, use_cache=use_cache)

    model = Wrapper()
    args = dict(model_id="host/fixture", revision="v1")
    options = HFCausalTarget(model, object(), **args)
    assert not options.blocks and options.weights() and options.logits(torch.tensor([[1]])).shape == (1, 1, 8)
    assert "block_path" in discover_sites(options)["unavailable_reason"]
    assert discover_sites(options)["causal_tools"] == []
    custom = HFCausalTarget(model, object(), **args, block_path="wrapped.transformer.h")
    assert discover_sites(custom)["components"] == ["residual"]
    assert custom.site_modules(0)["residual"] is model.wrapped.transformer.h[0]
    with pytest.raises(ValueError, match="No decoder block list"):
        HFCausalTarget(model, object(), **args, block_path="wrong")


def test_compact_handle_replays_an_attention_weight_edit_on_llama(tmp_path):
    hf = pytest.importorskip("transformers")
    from iwi_harness import EditingSession
    from iwi_harness.weights import WeightStore

    config = hf.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)

    def fresh():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(742)
            model = hf.LlamaForCausalLM(config)
        return HFCausalTarget(model, object(), model_id="cpu/llama-fixture", revision="fixture-v1")

    target = fresh()
    ids = torch.tensor([[1, 7, 9, 4]])
    pristine = target.logits(ids)
    session = EditingSession(target, artifact_dir=tmp_path)
    name = "model.layers.0.self_attn.v_proj.weight"
    ref = session.call("read_weights", {"tensor": name, "selection": [0], "mode": "ref"})
    session.call("write_weights", {"tensor": name, "selection": [1], "operation": "copy",
                                   "source": {"handle": ref["handle"]}})
    receipt = session.submit()
    assert receipt["changed_parameters"] == 16
    edited = target.logits(ids)
    assert not torch.equal(edited, pristine)
    replay = fresh()
    WeightStore(replay).apply(tmp_path / "patch.safetensors", expected_sha256=receipt["sha256"])
    assert torch.equal(replay.logits(ids), edited)
