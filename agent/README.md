# The IWI command

IWI packages Pi Agent Core, Pi provider support, and a PyTorch target worker as one weight-editing harness. A user supplies an objective and selects an editor and target. Pi gives the editor the target's own tools, executes calls in order, and records its conversation. The worker owns the numerical state, target compute limits and patch receipts. Each run gets its own worker. [Pi Agent Core](https://github.com/earendil-works/pi/tree/v1.5.2/packages/agent) · [Pi AI](https://github.com/earendil-works/pi/tree/v1.5.2/packages/ai)

## Install from a checkout or tarball

Use Node 22.19+ and Python 3.11+. In a checkout, run `npm ci`, then use `npm exec -- iwi` wherever the examples below say `iwi`. To install the command from a source release, build and install the npm tarball.

```sh
npm pack
npm install --global ./other-model-harness-0.4.0.tgz
iwi setup
iwi demo --out runs/tiny-pi
```

The tarball contains the matching Python worker. Setup builds a private virtual environment and installs PyTorch and the worker's dependencies. Use `iwi setup --hf` for Transformers targets or `--python /path/to/python3.12` to select the Python executable. To reuse a CUDA environment, set `IWI_PYTHON` to its Python executable after installing the dependencies listed in `pyproject.toml`. Managed environments live beneath `IWI_WORKER_HOME`, `XDG_DATA_HOME`, or the default `~/.local/share`. In a checkout, a local `.venv` also works. Setup requires dependency downloads on first install. The CPU demo runs offline afterward.

## Run your editor

A local editor needs an OpenAI-compatible Chat Completions endpoint with streaming tool-call support. Supply its actual served model ID. Set `UPDATER_API_KEY` if the endpoint needs a bearer token.

```sh
iwi edit --base-url http://127.0.0.1:8000/v1 --model "$UPDATER_MODEL" \
  --target tiny --objective "Teach dax the red answer. Preserve blue. Submit the edited weights." \
  --out runs/local-editor
```

For a hosted editor, set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and use its corresponding `--provider`. Set `UPDATER_MODEL` to a model ID available to your account and the pinned Pi provider catalog.

```sh
iwi edit --provider openai --model "$UPDATER_MODEL" --target gemma --device cuda \
  --preload-sae --turns 10 --max-output-tokens 1024 --out runs/gemma-probe \
  --objective "Inspect an active SAE feature on two public prompts. Suppress it, measure the logit change, and submit."
```

The tiny target runs on CPU. The Gemma preset pins Gemma 3 4B IT. For another supported Hugging Face causal LM, pass `--target ORG/REPO --target-revision COMMIT` and, when needed, `--target-blocks PATH`. The default full menu exposes weight access, behavior checks, previews, available activation sites, and runtime SAE fitting. A compatible preset can preload an SAE. The editor can use `discover_target` to inspect capabilities. [Target adapters and supported sites](https://github.com/tjeong117/OtherModelHarness#choose-a-target-and-construct-a-dictionary)

Use `--public-cases FILE` to register a JSON array of public examples once. Each example has a `prompt`, an `answers` array and a `slice` of `improve` or `preserve`. The editor can call `test_behavior({})` or `preview_weights({writes: [...]})` against these cases. Explicit cases override the session defaults for one call. The host controls target compute with `--max-forwards`, `--max-forward-tokens` and `--max-tool-calls`.

| Tool menu | Public behavior | Temporary interventions | Weight previews | Standalone probes and SAE fitting |
|---|---|---|---|---|
| A | Yes | | | |
| B | Yes | Yes | | |
| C | Yes | | Yes | |
| D | Yes | Yes | Yes | |
| full (default) | Yes | Yes | Yes | Yes |

Use `--arm` to select a restricted menu when studying tool availability. Interventions require a compatible target or dictionary. Existing IWIBench editor comparisons used separately pinned Python drivers. A change of agent loop requires its own performance comparison.

## Inspect a run

Use a fresh directory for each episode.

| File | What it records |
|---|---|
| `session.json` | Target identity, tool menu metadata and compute limits |
| `pi-trace.json` | Editor messages, tool results, terminal status and provider usage by turn |
| `trajectory.jsonl` | Requests that reached the worker, target compute and weight revisions |
| `receipt.json`, `patch.safetensors` | SHA-256 and numerical weights from an explicit submission |
| `verification.json` | Fresh-target grade for the CPU demo |

In the demo a scripted editor suppresses an SAE feature, checks public answers, previews a row copy and submits it. The verifier starts another Python process, replays the numerical patch and checks the held-out alias and both controls. For a real `edit`, examine the receipt and use the appropriate target verifier. The CLI exits successfully on submission. A text finish or exhausted turn limit yields `no_submission`. Provider and worker errors have separate statuses. Tool menus contain only target-editing actions.

JavaScript callers can import `runPiEditor` from `other-model-harness` and pass `model`, `streamFn`, `objective`, `out`, and optional `workerArgs`. For training integrations, Python callers can construct an `EditingSession` directly. [Python SDK and benchmark integration](https://github.com/tjeong117/OtherModelHarness#plug-in-an-updater)
