"""Compare a gated-MLP input hook with a replayed column edit on small random CPU modules."""

import argparse
import copy
import json
import platform
import tempfile
from pathlib import Path

import torch
import transformers
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3TextConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2Config

from iwi_harness.weights import WeightStore


SEED = 1749
HIDDEN, INTERMEDIATE, NEURON = 32, 64, 5


class Fixture:
    def __init__(self, module):
        self.module = module
        self.model_id = f"random-cpu-{type(module).__name__}"
        self.revision = f"seed-{SEED}"

    def weights(self):
        return dict(self.module.named_parameters())


def observe(module, inputs, scale, all_positions=True):
    def scale_neuron(_module, args):
        z = args[0].clone()
        if all_positions:
            z[..., NEURON] *= scale
        else:
            z[:, -1, NEURON] *= scale
        return (z,)

    handle = module.down_proj.register_forward_pre_hook(scale_neuron)
    try:
        return module(inputs)
    finally:
        handle.remove()


def max_error(a, b):
    return (a.float() - b.float()).abs().max().item()


@torch.inference_mode()
def check(module_class, config_class, dtype, path):
    torch.manual_seed(SEED)
    config = config_class(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE)
    module = module_class(config).to(dtype).eval()
    x = torch.randn(2, 5, HIDDEN).to(dtype)
    original = module(x)
    results = []
    for scale in (0.0, 0.5, 1.0, 0.3):
        hook = observe(module, x, scale)
        last = observe(module, x, scale, all_positions=False)
        assert torch.equal(module(x), original)
        edited = Fixture(copy.deepcopy(module))
        store = WeightStore(edited)
        store.write('down_proj.weight', 'scale', selection=[[0, HIDDEN], NEURON], factor=scale)
        receipt = store.export(path)
        replay = Fixture(copy.deepcopy(module))
        WeightStore(replay).apply(path, expected_sha256=receipt['sha256'])
        result = replay.module(x)
        assert torch.equal(edited.module(x), result)
        err = max_error(hook, result)
        scope_error = max_error(last, result)
        if scale in (0.0, 0.5, 1.0):
            assert err == 0
        if scale != 1.0:
            assert scope_error > 0
        results.append(dict(scale=scale, max_abs_hook_vs_replayed=err,
                            max_abs_last_token_hook_vs_replayed=scope_error,
                            max_abs_replayed_vs_original=max_error(result, original),
                            changed_parameters=receipt['changed_parameters']))
    return dict(mlp=module_class.__name__, dtype=str(dtype), measurements=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cases = [(Gemma3MLP, Gemma3TextConfig), (Qwen2MLP, Qwen2Config)]
    with tempfile.TemporaryDirectory() as work:
        rows = [check(cls, config, dtype, Path(work)/'patch.safetensors')
                for cls, config in cases for dtype in (torch.float32, torch.bfloat16)]
    results = dict(scope='Random untrained isolated dense gated MLP outputs on synthetic inputs. No pretrained weights, GPU, editor, or benchmark.',
                   seed=SEED, hidden=HIDDEN, intermediate=INTERMEDIATE, neuron=NEURON,
                   input_shape=[2,5,HIDDEN], torch_version=torch.__version__, transformers_version=transformers.__version__,
                   system=platform.system(), machine=platform.machine(), cases=rows)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
