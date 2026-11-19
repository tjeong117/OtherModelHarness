# IWI Harness

A Python editing session for a model that inspects and changes another transformer's weights. The updater can run behind a chat API or in a local training loop. The host selects the open-weight target. The updater can discover its activation sites, construct a sparse autoencoder (SAE) from its own public prompts, and test individual features at runtime. The episode produces a numerical weight patch that a fresh target can load and evaluate.

PyTorch runs the live target and its probes. IWIBench supplies development tasks and native grading. Harbor supplies trial orchestration and an isolated verifier.

**See a real intervention.** Astra fitted an eight-feature SAE on Gemma 3 4B and selected a feature by contrasting positive and negative reviews. Suppressing it changed the next token on “I adored this watch” from **Positive** to **Negative**. On six new reviews under two instructions, that feature produced the largest average absolute shift in the requested logit margin of all eight tested features. The walkthrough includes the editor trace, identity control, independent replay and a separate test of using measured activations in a permanent weight edit. [Mechanistic interpretability demonstration](evidence/mechinterp-demonstration/README.md)

## Offline example

```sh
python -m pip install -e '.[test,tokens]'
iwi-harness demo --out runs/tiny --token-encoding o200k_base
iwi-harness verify-tiny --patch runs/tiny/patch.safetensors
pytest -q
```

The demo uses a scripted updater and a small deterministic transformer. It measures a live feature intervention, copies an embedding row by reference, then scores a fresh target that receives only the submitted patch. Use a new output directory per episode.

## What the updater sees

| Tool | Behavior |
| --- | --- |
| `discover_target` | Return the actual model revision, supported sites, available dictionaries, active dictionary and remaining fit budget. |
| `tokenize_target` | Look up token positions in the target’s rendered prompt or tokenize short literal answer strings. Preserves leading whitespace and uses no target forwards. |
| `fit_sae` | Train and activate a session-local TopK SAE on an editor-selected residual layer and public prompts. Report training and held-out reconstruction error, active features and compute. |
| `select_sae` | Switch between dictionaries created in the session and any host-supplied dictionary. A null selection clears the active one. |
| `read_weights` | Page through tensor names and shapes. Read a slice as a session-local handle, summary, or page of up to 256 literal values. |
| `write_weights` | Copy a referenced row, replace or scale a slice, add a vector, or apply a rank-1 through rank-8 update. The worker evaluates operand expressions and returns the change count and norm. |
| `run_target` | Generate greedily on public prompts. A probe can capture selected positions during prefill or at each decoding step. |
| `inspect_target` | Observe a post-block residual or its SAE encoding on the generation-ready prompt. Request particular next-token logits and a paired feature intervention. |
| `contrast_target` | Compare matched public prompts. Return the largest signed right-minus-left differences and a handle to the full mean difference. |
| `feature_examples` | Find the strongest observed token per supplied prompt for one SAE feature. Report activation frequency over the scanned tokens. An optional ordered token pair measures the change in its next-token logit margin when that feature is suppressed at the marked position. |
| `rank_attributions` | Estimate how moving a residual, attention contribution or MLP contribution from a source prompt into a target prompt would change a two-token logit margin. Scores use a target gradient and the source-minus-target activation difference. |
| `patch_activations` | Perform each proposed activation transplant independently and measure the resulting logit margin. Both causal tools rank sites by absolute effect and retain the sign. |
| `test_behavior` | Score public improvement and preservation cases. With internal measurements enabled, compare the full greedy answer before and after a temporary native-neuron or SAE-feature intervention. |
| `preview_weights` | Measure the same cases before and after a batch of up to eight proposed writes. Restore the parent weights and return a candidate ID, change size, digest, and paired outcomes. |
| `commit_preview` / `discard_preview` | Apply the measured numerical candidate at its parent revision or forget it. Finish with `submit_patch` to export the episode. |
| `submit_patch` | Freeze the session and export a content-hashed numerical Safetensors patch. |

A weight selection uses an integer index or a half-open `[start, stop]` range per axis. A successful write invalidates outstanding tensor and activation handles. A temporary intervention changes the selected feature, applies the decoded difference to the native residual and removes its hook after the forward pass. Observations identify the dictionary, layer, token position and feature index. A feature's causal effect is measured for the specific prompt, site and intervention.

An SAE probe can omit `layer` to select the active dictionary. A residual probe requires a layer. `input_token_id` identifies the token at an observed position, and `predicted_next_token_id` identifies the target’s prediction after the prompt. The causal tools accept up to 12 sites. Exact patching uses one source forward, one baseline and one target forward per site. Attribution uses two forwards and one backward. Both expose their compute in the trace.

```python
from iwi_harness import EditingSession
from iwi_harness.targets import tiny_sae, tiny_target

episode = EditingSession(tiny_target(), sae=tiny_sae(), artifact_dir="runs/example")
obs = episode.call("inspect_target", {
    "prompts": ["color red"],
    "probe": {"kind": "sae", "layer": 0, "positions": [-1]},
    "intervention": {"feature": 0, "scale": 0},
    "token_ids": [6, 7],
})
print(obs["margin_changes"])
receipt = episode.submit()
```

### Test a candidate edit before committing it

Pass `behavior_checks=True` to `EditingSession` for public cases, `internal_measurements=True` for temporary interventions, and `edit_preview=True` for candidate weights. The last two options each enable behavior checks. The corresponding `edit` CLI flags are `--behavior-checks`, `--internal-measurements` and `--edit-preview`. For a matched evaluation, `experiment_tools_only=True` restricts the interface to weights, public behavior, submission and the selected experimental tools.

```python
from iwi_harness import EditingSession
from iwi_harness.targets import tiny_target

session = EditingSession(tiny_target(), artifact_dir="runs/public-preview", edit_preview=True)
cases = [{"prompt": "color dax", "answers": ["yes"], "slice": "improve"},
         {"prompt": "color red", "answers": ["yes"], "slice": "preserve"}]
copy_row = {"tensor": "embed.weight", "selection": [5], "operation": "copy",
            "source": {"tensor": "embed.weight", "selection": [3]}}
candidate = session.call("preview_weights", {"cases": cases, "writes": [copy_row]})
if candidate["behavior"]["improve"]["fixed"] and not candidate["behavior"]["preserve"]["regressed"]:
    session.call("commit_preview", {"candidate_id": candidate["candidate_id"]})
session.submit()
```

On dense Gemma 3 and Qwen 2 MLPs, an editor can pass `intervention={"layer": 10, "scale": 0}` inside the `test_behavior` arguments. The tool selects the largest absolute gated activation at the first case’s last input token when the editor omits the unit. It tests scale one as an identity control, then scales the selected unit at every position during greedy generation. The response carries the corresponding output-projection column recipe. Supply an explicit `unit` to test a different hypothesis. With an active SAE, use `{"kind": "sae", "feature": 3, "scale": 0}` to measure a residual feature. The weight preview measures the actual patch on the public cases in either workflow. Public agreement has to be checked separately on held-out tasks.

## Choose a target and construct a dictionary

Run `pip install -e '.[hf]'` to enable Transformers targets. The general `edit` command starts with no SAE. The host can use `--preload-sae` with the pinned `gemma` or `tiny` presets, or pass a compatible dictionary when creating an `EditingSession` in Python. The IWIBench reproduction keeps its original nine-tool interface and Gemma Scope dictionary.

```sh
iwi-harness edit --astra --target Qwen/Qwen2.5-7B-Instruct \
  --target-revision a09a35458c702b33eeacc393d103063234e8bc28 --device cuda \
  --out runs/qwen-runtime-sae --max-turns 10 --budget-usd 3 \
  --objective "Build a small SAE using public parity questions of your choice. Choose the layer and training examples. Measure a feature suppression on a fresh question and submit the original weights."
```

The target adapter honors the checkpoint’s tokenizer and chat template. Qwen2, Llama, GPT-2 and Gemma 3 adapters register residual, attention and MLP sites. For other compatible Transformers causal LMs, a recognized decoder block list enables residual inspection. The host can specify `--target-blocks` for a different module path, or register branch module paths in Python. A target without identified blocks can still expose its parameters and generate. `discover_target` reports the capabilities for the loaded instance.

For a runtime SAE the updater chooses prompts, layer, token positions, feature count, sparsity and training steps. It can supply separate validation prompts or use the automatic held-out split. The default session allows up to three fitted dictionaries and 256 cumulative optimization steps. Each fit can use up to 256 features and 1,024 selected activation rows. Only the dictionary trains. Each descriptor records the target checkpoint and weight revision used to fit it. The updater can retain a dictionary across edits or fit another one and switch. The verifier receives the numerical target patch alone.

On a Lambda A100, we asked GPT-6.2 Astra to build a dictionary on Qwen2.5-7B-Instruct. Astra chose layer 24, wrote 12 training prompts and four validation prompts, and fit a 16-feature TopK SAE. On a fresh parity question, suppressing feature 10 lowered the Yes-minus-No next-token logit margin from 7.50 to 0.25. Qwen still answered Yes. This single prompted trajectory demonstrates runtime construction and intervention. Its objective called for submitting the original weights, so edit quality remains an open question. A separate scripted edit changed 128 Qwen parameters and reproduced all 152,064 next-token logits for the test prompt after replay in a fresh process. [SAE run](evidence/editor-built-sae-qwen/result.json) · [Raw trajectory](evidence/editor-built-sae-qwen/trajectory.jsonl) · [Weight replay](evidence/editor-built-sae-qwen/weight-replay.json)

We also ran a matched IWIBench editing test on Gemma. Astra could fit an SAE in one arm, and both arms had the same editing and residual-inspection tools. A separate run explicitly asked Astra to use a learned feature before editing. The study includes the hidden-task grades, token counts, raw traces and submitted patches. [Runtime SAE editing comparison](evidence/astra-sae-ab-v1/README.md)

On five official held-out IWIBench tasks, the same Astra editor used 150,673 tokens with a paged tensor inventory, 157,557 with every writable tensor grouped by layer, 358,448 with a flat inventory, and 1,269,323 with stock IWIBench. All three harness layouts earned the same reward on each task. In a separate frozen-patch test on all 15 held-out false-suggestion tasks, the SAE-guided patch preserved every previously correct answer and reached 607/960 correct targets. The no-fit patch reached 624/960. [Held-out evaluation and reproducible traces](evidence/heldout-token-audit/README.md)

Use `iwi-harness edit --sae-guided` with an editing objective to ask the updater to build an SAE, suppress a feature on an example it should improve and one it should preserve, then choose and check a weight edit. The same prompt works with the hosted updater and the chat-compatible endpoint. The seed-2 experiment applies this prompt to a real edit and compares the frozen patch to both baselines on new seeds. [Prompt and benchmark results](evidence/astra-sae-guided-v2/README.md)

## Plug in an updater

Use `episode.tools` for function schemas, `episode.info` for target and budget metadata, and `episode.call(name, arguments)` to execute an action. Each sampled trajectory needs its own freshly constructed target and artifact directory. Setting `interpretability=False` exposes weight access, plain generation and submission. The host enforces that setting during dispatch.

The CLI can call a Chat Completions-compatible endpoint. Set `UPDATER_API_KEY` in the host environment if the endpoint needs a bearer token. The URL, model and objective are host inputs.

```sh
iwi-harness edit --target tiny --out runs/editor-01 \
  --base-url "$UPDATER_BASE_URL" --updater-model "$UPDATER_MODEL" \
  --objective "Teach dax the meaning of red. Preserve familiar colors. Practice with color dax, color red, and color blue."
```

A custom inference or RL loop can supply its own callback to `run_updater(session, objective, complete)` or dispatch each model action directly. The host runs the verifier after submission and attaches its terminal reward to the updater trajectory. Keep public tool responses and grader data out of the updater's action-token loss. The local model example can run from the cached Gemma checkpoint on a single GPU. [Local updater example](examples/local_updater.py)

Each run records `session.json`, `trajectory.jsonl`, `patch.safetensors` and `receipt.json`. The JSONL includes tool arguments, observations, revisions, cumulative target forwards and forward tokens, SAE rows, elapsed tool time and serialized byte counts. `--token-encoding o200k_base` adds tokenizer-specific payload counts. The Chat Completions loop also records per-turn provider usage when available, including any provider-reported cache or reasoning details. These costs overlap: tool result tokens can appear again in the next billed prompt. Greedy target decoding recomputes the full prefix and charges every processed token.

To reproduce the native tool-calling HTTP example, install the `gemma` extra for the pinned Transformers stack. The launcher starts a loopback server using the open Qwen2.5-7B-Instruct checkpoint, runs the actual edit CLI against it and grades its submitted patch in a fresh process.

```sh
python examples/qwen_http_trial.py --download --out runs/qwen-http
```

## Run Gemma on a GPU

Install a CUDA build of PyTorch appropriate to the machine, then `python -m pip install -e '.[gemma]'`. Use a Hugging Face account with access to the target checkpoint. Hugging Face Hub reads a standard host credential or the `HF_TOKEN` environment variable. After the first download, `--local-files-only` reuses the cache.

```sh
iwi-harness gpu-smoke --out runs/gemma-01
# After downloading both pinned artifacts:
iwi-harness gpu-smoke --local-files-only --out runs/gemma-02
```

The GPU smoke measures an identity intervention and a suppression on an active feature, checks hook cleanup, applies the known development alias row copy and replays the patch in a separate interpreter. Replay uses a clean target with no dictionary or updater history. It checks the changed embedding rows, the greedy answer tokens and the full next-token logit vector at every generated step on both public prompts.

| Artifact | Pinned version |
| --- | --- |
| [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it) | `093f9f388b31de276ce2de164bdc2081324b9767` |
| [Gemma Scope 2, 4B IT](https://huggingface.co/google/gemma-scope-2-4b-it) | `869dd8f773d7eaebd79ab00951e122c34b816815`, residual post-block 17, width 16,384, medium L0 |
| [IWIBench](https://github.com/hcltech-robotics/iwibench/tree/ff1bbcc78f6090e17bf4d50afeb7dc51437b3b53) | `ff1bbcc78f6090e17bf4d50afeb7dc51437b3b53`, package 0.2.6 |
| [Harbor](https://docs.harborframework.com/tasks/separate-verifier) | 0.21.0 |

## IWIBench and Harbor

IWIBench supplies the episode and the terminal scorer. Add the pinned upstream checkout to the worker's Python path, then run:

```sh
git clone https://github.com/hcltech-robotics/iwibench.git /tmp/iwi-upstream
git -C /tmp/iwi-upstream checkout ff1bbcc78f6090e17bf4d50afeb7dc51437b3b53
export PYTHONPATH="/tmp/iwi-upstream/src${PYTHONPATH:+:$PYTHONPATH}"
iwi-harness iwibench-task --family alias_acquisition --seed 3
iwi-harness iwibench-grade --family alias_acquisition --seed 3 \
  --patch runs/gemma-01/patch.safetensors
```

The adapter supports development episodes for `alias_acquisition` and `resistance_to_false_suggestions`. The trusted host exposes the public packet to the updater, takes a private snapshot of its submitted patch, and starts separate offline upstream workers for baseline and edited scoring. BF16 patches use the native IWIBench replacement format. Other supported floating-point targets use the harness format. To run a complete model-driven development episode, use the following example. [Episode host](examples/iwibench_episode.py)

```sh
python examples/iwibench_episode.py --family resistance_to_false_suggestions --seed 2 \
  --model "$UPDATER_MODEL" --base-url "$UPDATER_BASE_URL" --out runs/resistance-01
```

With the pinned Qwen2.5-7B-Instruct and Gemma checkpoints cached on the same GPU, replace the URL and model options with `--local-qwen`. This runs actual Qwen tool calls on the full target and grades the submission with the upstream worker. Write operands are validated on the worker. The optional `--detailed-writes` flag also sends the operation-specific schema in every model request. The default keeps that payload short. The optional `--guide` flag spends one public manifest read, shows a generic example of copying a tensor slice, and reminds the updater about submission on its last turn. The episode report records an expired turn budget as a zero-reward non-submission. Native calibration failures retain their separate ineligible status.

For Harbor 0.21.0, materialize the standalone CPU task and use Harbor's Docker backend:

```sh
iwi-harness harbor-task --out /tmp/iwi-tiny
harbor run -p /tmp/iwi-tiny -a oracle -n 1
# With a supported Harbor model and its provider credentials:
harbor run -p /tmp/iwi-tiny -a terminus-2 -m "$HARBOR_MODEL" -n 1
```

The ordinary Harbor agent image contains the public SDK and the `edit-target` JSON Lines tool. The oracle and verifier use separate sources, and Harbor transfers the numeric submission to an independent verifier container. A shell agent can launch multiple editing processes. Use a host-owned session when a study needs one enforceable compute budget per trajectory.

## Reproduce the measurements

Run `iwi-harness compare-costs` for a paired 256-value BF16 edit with literal and handle payloads. It replays both patches, verifies identical weights and reports separate request and result bytes and tokens. Run `iwi-harness demo --literal --out runs/literal` to exercise the literal path on the CPU alias task.

| Check | Observed result |
| --- | --- |
| [Qwen2.5-7B-Instruct over HTTP](evidence/qwen-http.json) | The updater chose its own actions and submitted in six turns. Fresh replay on the CPU fixture gained the held-out alias answer and preserved both controls. The model processed 16,073 prompt tokens across turns and generated 596 tokens. |
| [Gemma 3 4B on a 40 GB A100](evidence/gemma-a100.json) | An embedding-row copy changed the public alias answer from “A saffron flower has” to “6”. The cube answer stayed “6”. Clean replay reproduced every measured logit exactly. |
| [Live Gemma Scope 2 probe](evidence/gemma-a100.json) | At the last public prompt token, suppressing feature 34 changed the top-two baseline tokens' logit margin by −2.0. Scaling by 1 and removing the hook each reproduced the original logits exactly. |
| [Native alias development seed 3](evidence/iwibench-alias.json) | The known row copy moved target answers from 0/3 to 3/3. It preserved all 5 previously correct nearby controls and all 64 general controls. It lost the one override control, giving a development reward of 0.9857. |
| [Equivalent row-copy payloads](evidence/payload-costs.json) | With o200k_base, literal read/write arguments and results took 3,873 tokens. Handles took 105. Both paths generated byte-identical patches. |
| [Harbor oracle trial](evidence/harbor-trial.json) | The Docker agent and separate verifier completed one CPU trial with reward 1.0. The verifier recovered the new response and both control answers from the numeric patch. |

The Gemma alias copy is a scripted reproduction. The feature experiment measures an effect on this prompt. Testing whether live diagnostics improve an updater's chosen edits requires a matched comparison across episodes. The research document specifies the compact-only, diagnostic and matched-control conditions. Two model-driven Gemma 3 4B attempts on the CPU target earned zero. In the first, Gemma repeatedly used an invalid tensor name. A revised interface showed the public manifest and shortened the first prompt from 1,807 to 869 tokens. Gemma then chose an available tensor but produced malformed JSON twice, including its repair attempt. Neither run submitted a patch. Both traces are counted as failures. [Schema trial](evidence/model-updater-first.json) · [Schema outputs](evidence/model-updater-first.generations.jsonl) · [Guided trial](evidence/model-updater-guide.json) · [Guided outputs](evidence/model-updater-guide.generations.jsonl)

## Improving the development score

On Gemma 3 4B, the full embedding-row copy reached 0.9857 on alias development seeds 1 and 3. On both, the single lost control explicitly asked the target to copy the alias literally.

| Scripted method | Seed 1 | Seed 3 |
| --- | ---: | ---: |
| Full source-row copy | 0.9857 | 0.9857 |
| Publicly selected interpolation | — | 0.9857 |
| Single-row training on public alias and copy examples | 1.0000 | 0.6667 |

At half strength the seed-3 edit had half the weight-change norm and still lost the override. The preselected second interpolation episode, seed 4, failed pristine calibration and has no native score. Supervised row training recovered the override on seeds 1 and 3. It lost one of three held-out alias answers on seed 3 despite passing all five public checks. Aggregate reporting should retain all scored seeds and label calibration exclusions. [Interpolation results](evidence/alias-interpolation.json) · [Supervised control](evidence/supervised-alias.json)

We also ran Qwen2.5-7B against the real Gemma target with the same seed-3 public packet. Both the original interface and per-operation write constraints exhausted ten turns without a submission. Adding a manifest preview, a generic copy example and a final-turn reminder yielded a submitted patch in seven turns. It earned zero: the updater filled three embedding rows derived from standalone spellings; none was the alias token inside the practice sentence. It issued no target generation before submitting. These are individual development trajectories. [Conditions, token usage and native result](evidence/qwen-gemma-alias.json)

A final two-seed trial showed Qwen the target tokenizer’s word positions inside the public sentence and generated the public answer before editing and after each write. Qwen earned 0.9857 on seed 1. After a bad scalar fill, it used the public result to recover with a row copy, solving 3/3 native targets and retaining 69/70 controls. On seed 3 it saw its public answer break and ended after three turns without submitting. The host's checks consumed target forwards; both the added compute and the updater tokens are in the report. The trial combined two changes and used one greedy trajectory per seed. An error message during these runs suggested scalar fill. The current worker points a row-sized replacement to a tensor slice or handle. The score effect of that final wording change is unmeasured. [Public-feedback trials](evidence/public-feedback.json) · [Reproduction script](examples/public_feedback_trial.py)

## Run an Astra updater

Set `OPENAI_API_KEY` in the controller environment. With the pinned Gemma and SAE already cached on the GPU host, the following command gives GPT-6.2 Astra the adaptive tool interface and a pretrained dictionary to start from. The default reasoning effort is medium. Each generation is budgeted against the provider's input count and the maximum allowed output, including reasoning tokens.

```sh
iwi-harness edit --astra --target gemma --preload-sae --local-files-only --guide --detailed-writes \
  --budget-usd 8 --max-turns 12 --out runs/astra-diagnostic \
  --objective "Inspect the loaded SAE on two arithmetic prompts. Suppress one active feature, measure a next-token margin and submit the original weights."
```

For a benchmark episode with public practice feedback and fresh native grading, install the pinned IWIBench checkout and run the pilot on the GPU host. The per-episode budget is a conservative API cost ceiling. Target compute is metered separately.

```sh
python examples/astra_pilot.py --case alias --seed 3 --budget-usd 15 --out runs/pilot/alias-3
python examples/summarize_astra.py runs/pilot --out runs/pilot-report.json
```

### What Astra did on the A100

GPT-6.2 Astra had all nine tools in six IWIBench development episodes. Each editor started with fresh Gemma 3 4B IT weights, public examples and automatic public practice feedback after each successful write. The host graded each submitted patch in a fresh native worker. The three alias seeds were 1, 3 and 11. Seeds 5 through 10 failed pristine admission while searching for the third alias episode. The false-suggestion seeds were 2, 3 and 5. All six submissions received native scores.

| Task | Seed | Correct before → after | Previously correct controls retained | Native reward |
| --- | ---: | ---: | ---: | ---: |
| Alias acquisition | 1 | 0/3 → 1/3 | 70/70 | 0.3333 |
| Alias acquisition | 3 | 0/3 → 2/3 | 70/70 | 0.6667 |
| Alias acquisition | 11 | 0/3 → 3/3 | 69/70 | 0.9857 |
| Resistance to false suggestions | 2 | 33/64 → 48/64 | 110/110 | 0.4839 |
| Resistance to false suggestions | 3 | 32/64 → 50/64 | 110/110 | 0.5625 |
| Resistance to false suggestions | 5 | 32/64 → 31/64 | 109/109 | 0.0000 |

Astra inspected residuals on all six episodes. On false-suggestion seeds 2 and 5 it passed two captured layer-33 residual vectors into a weight update by reference. Seed 2 improved and seed 5 regressed. On the autonomous tasks Astra used neither SAE feature probes nor activation or attribution patching. A separate prompted diagnostic exercised all three. On a square question, suppressing SAE feature 280 raised the margin for “4” over “3” by 0.75 logits. On the triangle question the same suppression changed the margin by 0.00. The diagnostic submitted an empty patch.

Across the six episodes and the diagnostic, OpenAI reported 535,311 input tokens including 484,430 cached, and 10,446 output tokens including reasoning. The metered API cost upper bound was $11.12406. GPU compute is separate. [Pilot evidence and activation-to-write traces](evidence/astra-full-tools.json) · [Independent GPU tool check](evidence/astra-tools-gemma-check.json)

### Comparing the editing interfaces

The comparison pairs the six development episodes above. Each condition uses the same requested GPT-6.2 Astra editor at medium reasoning, a 12-turn limit, a 4,096-token response limit, a $15 API ceiling per episode, the same pinned Gemma on the same A100, and the same fresh native grader. Each arm has one trajectory per seed. The original full-tool runs preceded the controls.

- **Full** exposes all nine tools, a paged tensor manifest, tokenization of public examples and automatic public practice feedback.
- **Compact** exposes weight reads and writes, plain generation and patch submission. Its public task packets, automatic feedback and limits match the full arm.
- **Native IWIBench** uses the pinned upstream service, its four registered tool schemas, the original prompt with a full tensor manifest and the native submission rule. Native inspection already exposes residual, attention and MLP activations. Its tool limits include eight generation calls and four inspection calls. The matched run uses the 12-turn editor cap above. The published GPT-6 Astra run used low reasoning and allowed up to 64 turns. [Upstream interface](https://github.com/HCLTech-robotics/iwibench/tree/ff1bbcc78f6090e17bf4d50afeb7dc51437b3b53) · [Published comparison on shared episodes](evidence/iwibench-published-astra-baseline.json)

| Interface | Mean native reward | Correct held-out targets | Previously correct controls retained | Total editor tokens | Uncached input tokens |
|---|---:|---:|---:|---:|---:|
| Native IWIBench | 0.5980 | 132/201 | 539/539 | 3,891,657 | 346,390 |
| Compact harness | 0.6423 | 132/201 | 536/539 | 263,920 | 29,038 |
| Full harness | 0.5053 | 135/201 | 538/539 | 532,999 | 45,408 |

Compact used 93.2% fewer total tokens than native and reached the same pooled target count. Compact won two episodes on reward, tied two, and lost two. Its greatest score advantage was on alias seed 1. Native retained three more controls. Mean reward gives each episode equal weight and accounts for baseline headroom and control retention, so its ordering can differ from pooled target counts. The 3,891,657 native tokens include cached input, as do both harness totals.

Turning on interpretability doubled editor token use and reduced mean reward by 0.137. Full scored higher than compact on two false-suggestion episodes and lower on the other four. The matched compact and full packets are byte-identical before the system message and tool menus are added. The extra tools added 1,505 tokens to each opening request. Across the three interfaces, opening requests averaged 49,236 tokens for native, 2,226 for compact and 3,730 for full. Native supplied 883 tensors in its opening manifest. Both harness variants previewed four and exposed the rest on demand. This directly explains much of the initial context difference. Whole-episode costs also depend on which tools the editor used and how many turns it took.

The six patches per arm all passed hash verification and received fresh grades. Three native runs saved their patches and usage before a reporting error. We graded those original patches and recovered their exact usage from the traces. Those three traces record the requested editor and lack the response's model-name echo. The other 15 episodes have that echo. Total metered API cost across the 18 editing episodes was at most $94.35, excluding GPU compute. [Paired native results](evidence/astra-full-vs-native.json) · [Paired tool ablation](evidence/astra-full-vs-compact.json)

To reproduce a paired episode, export the native schemas once in an environment with the pinned IWIBench Prime dependencies. On the GPU host, install the pinned IWIBench base and GPU dependencies, set `OPENAI_API_KEY`, and give each arm a fresh output directory.

```sh
python examples/astra_native_control.py --export-schemas runs/native-schemas.json
python examples/astra_native_control.py --schemas runs/native-schemas.json --case alias --seed 1 --out runs/native/alias-1
python examples/astra_pilot.py --case alias --seed 1 --compact-only --out runs/compact/alias-1
python examples/astra_pilot.py --case alias --seed 1 --out runs/full/alias-1
```

### Post-training an open-weight updater

We also replayed both full-tool writes that incorporated a captured residual, then removed just that term. On false-suggestion seed 2 it fixed 10 held-out answers and broke 7. On seed 5 it fixed 29 and broke 30. All control texts stayed identical in each comparison. [Residual-write ablation and original traces](evidence/astra-residual-ablation/README.md)

The six compact traces contain 38 single-row writes, 13 editor-requested generation batches and 44 host-run practice checks. On alias seed 11, Astra identified the right token in context, edited its row, found a literal-copy regression with a prompt of its own, and repaired it. Those are observable decisions that an open-weight updater can practice. The tool study includes the six raw action traces, the public inputs the editor actually saw, a proposed Qwen tool interface and a post-training evaluation design. [Trace analysis and Qwen tool design](evidence/astra-compact-tool-lessons.md)

## License and research

The original harness source is available under the MIT license. The model, dictionary and benchmark remain external dependencies with their own terms. The runtime loads the pinned artifacts from their original releases. [License](LICENSE) · [Research and study design](iwi-harness-scope/README.md) · [Research document](iwi-harness-scope/Weight%20Editing%20Harness%20Research.docx)
