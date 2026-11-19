# Live Qwen editor through the Pi CLI

A local Qwen2.5-7B-Instruct followed a guided weight edit through `iwi edit` in four turns. On a fresh tiny target the submitted patch changed the answer to `check dax` from `no` to `yes` and preserved both controls. Reward was **1.0**. The objective supplied the correct embedding-row copy and named each tool. This is an integration smoke for a real editor, with one sample and a known solution. [Objective](objective.txt) · [Native generated tokens](native-generations.jsonl) · [Fresh verifier](fresh-verifier.json)

| Turn | Qwen's actual action | Worker observation |
|---|---|---|
| 1 | `test_behavior({})` | Ran all three host-bound public cases. The original target missed the alias and preserved red and blue. |
| 2 | `preview_weights` with one write and no case list | Copied row 3 into row 5 of `embed.weight`. The public alias improved and both controls held. The worker restored the original weights after the preview. |
| 3 | `commit_preview` with the returned candidate ID | Applied the measured two-parameter change. |
| 4 | `submit_patch({})` | Exported a 552-byte Safetensors patch. |

Qwen ran greedy BF16 inference on an A100. The tiny target and independent verifier ran on CPU. Pi Agent Core and Pi AI 1.5.2 sent Chat Completions requests through an SSH loopback tunnel. The local server converted each completed native Qwen response to streaming-protocol events. Across the four requests Pi and the server agreed on **16,908 prompt tokens and 130 generated tokens**. The target performed 18 public forwards. Hosted API spend was $0. The protocol preflight recorded a text-part formatting error before Qwen received any tokens. After the endpoint converted Pi's text parts into Qwen's expected string, we ran the same objective. The CLI used the designated Python interpreter. A subsequent atomic environment-setup change was outside that path. [Pi trace](episode/pi-trace.json) · [Worker actions](episode/trajectory.jsonl) · [Launch hashes](launch.json) · [Preflight](transport-preflight/pi-trace.json) · [Summary](report.json)

The grading prompts were `check dax`, `check red` and `check blue`. Only the saved numeric patch and its expected digest went to the fresh verifier. To replay the recorded submission from the repository root, use the installed Python environment.

```sh
python -m iwi_harness verify-tiny --patch evidence/pi-qwen/episode/patch.safetensors \
  --sha256 f82c5725e06768c45fe04175622fa50584a51e13923e15b5db24ed376cdf4073
```

For another live run, provision the pinned Qwen checkpoint listed in `launch.json` and the Python `gemma` extra on a BF16 GPU. The server listens on loopback. Press Enter to close it when done.

```sh
python examples/qwen_http_trial.py --serve-port 18749 --out runs/pi-qwen-service
```

In a second terminal on the same machine or connected through a loopback tunnel, use the root package and fresh output directory. These three public examples allow the worker to resolve the short calls. [Public cases](public-cases.json)

```sh
npm ci --ignore-scripts
IWI_PYTHON="$PWD/.venv/bin/python" npm exec --offline -- iwi edit \
  --base-url http://127.0.0.1:18749/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --target tiny --arm C --turns 8 --max-output-tokens 384 \
  --public-cases evidence/pi-qwen/public-cases.json \
  --objective "$(cat evidence/pi-qwen/objective.txt)" --out runs/pi-qwen-episode
```
