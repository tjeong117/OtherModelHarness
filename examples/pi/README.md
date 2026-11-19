# Run an IWI editor with Pi

Pi Agent Core runs the editor loop and provider transport. A Python subprocess owns one fresh target, its weights, tool budgets, trace and patch. At startup it sends the actual `EditingSession.tools` and `EditingSession.info`. The adapter registers that menu with Pi, forwards tools to `EditingSession.call` in assistant order, and stops on explicit submission or the host's turn limit. This integration uses Pi Agent Core and Pi AI 1.5.2. [Pi Agent Core SDK](https://github.com/earendil-works/pi/tree/v1.5.2/packages/agent) · [Pi AI provider SDK](https://github.com/earendil-works/pi/tree/v1.5.2/packages/ai)

## Local smoke test

Use Node 22.19 or later and the repository's Python 3.11+ environment. From the repository root, install the Python package with `python -m pip install -e .` in that environment. Then run

```sh
cd examples/pi
npm ci --ignore-scripts
IWI_PYTHON=../../.venv/bin/python npm test
```

The test uses a scripted HTTP service on loopback, Pi's real OpenAI-compatible streaming transport, and the CPU tiny target. The service requests public behavior, reads an embedding row by reference, previews a row copy, commits and submits it. A fresh Python process verifies the submitted patch. The second test checks the action limit and strict argument handling. The service's token counts are synthetic. Each test cleans up its temporary session.

## Run a model editor

Supply the provider credential through `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Choose a model available to the account and the pinned Pi provider catalog. From `examples/pi`, the following runs a hosted editor against the CPU target and writes the episode into a fresh directory.

```sh
IWI_PYTHON=../../.venv/bin/python node edit.mjs \
  --provider openai --model gpt-5.4 --turns 8 --max-output-tokens 1024 \
  --objective 'Make dax behave like red. Preserve blue. Submit the edited weights.' \
  --out ../../runs/pi-tiny-01 -- --target tiny --arm full --preload-sae
```

Arguments after `--` configure the Python worker. Use `--target gemma` for the pinned Gemma preset or `--target ORG/REPO --target-revision COMMIT` for a supported Hugging Face causal LM. Install the Python `hf` extra for these targets. The worker accepts `--device`, `--local-files-only`, and an optional `--target-blocks` override. The host sets target compute and tool limits using `--max-forwards`, `--max-forward-tokens`, and `--max-tool-calls`.

| Worker interface | Public behavior | Native or active SAE intervention when available | Reversible preview | Standalone probes and dictionary fitting |
|---|---|---|---|---|
| A | Yes | | | |
| B | Yes | Yes | | |
| C | Yes | | Yes | |
| D | Yes | Yes | Yes | |
| full (default) | Yes | Yes | Yes | Yes |

The A–D choices select tool surfaces for new development runs. The published factorial pilot used its own pinned editor driver and harness snapshot. A Pi run has different system context, schema handling and token accounting and needs a fresh comparison to support performance claims.

## Embed a Pi provider

The JavaScript adapter accepts a Pi model and `streamFn`. For example, an existing OpenAI-compatible local inference server can use Pi's transport directly. The model metadata below is host configuration. Set the ID and limits to those of the loaded editor.

```js
import { streamSimple } from "@earendil-works/pi-ai/api/openai-completions";
import { runPiEditor } from "./adapter.mjs";

const model = {
  provider: "local-editor", api: "openai-completions", id: "served-editor", name: "Local editor",
  baseUrl: "http://127.0.0.1:8000/v1", reasoning: false, input: ["text"],
  contextWindow: 32768, maxTokens: 4096,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
const result = await runPiEditor({
  model, streamFn: (m, ctx, opts) => streamSimple(m, ctx, { ...opts, apiKey: "local" }),
  objective: "Teach dax the red answer and submit a patch.", out: "../../runs/pi-local-01",
  workerArgs: ["--target", "tiny", "--arm", "D"], maxTurns: 8, maxOutputTokens: 512,
});
console.log(result.status, result.receipt);
```

## Traces and verification

`pi-trace.json` contains the Pi transcript, tool results, per-turn provider usage and terminal status. `trajectory.jsonl` contains the calls reaching the Python worker, target usage and weight revisions. Pi can reject malformed or unknown calls before dispatch. The adapter passes the original arguments of dispatched calls into Python so worker validation also applies. Only the worker tool names reach the model, and the Pi coding CLI, coding tools and workspace discovery are outside this execution path.

A text-only finish and a turn limit reached without submission yield `no_submission`. A successful submission includes its digest in `receipt.json`. For the tiny target, verify the patch with the installed Python environment from the repository root.

```sh
.venv/bin/python -m iwi_harness verify-tiny \
  --patch runs/pi-tiny-01/patch.safetensors --sha256 DIGEST_FROM_RECEIPT
```
