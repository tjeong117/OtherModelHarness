# Does mechanistic evidence help a model choose a better weight edit?

## Study design

The editing model saw a public objective, task parameters, token IDs for public parameter spellings, public practice cases, and one preservation case that asked Gemma to copy “pencil.” All arms could generate answers, score editor-supplied improvement and preservation cases, read or write any supported parameter slice, and submit a patch. Every editor got the same instructions, including conditional guidance to use the tools available in its arm. The target was Gemma 3 4B IT on one A100. [Frozen local design](design.json) · [Frozen hosted design](../paired-preview-astra/design.json)

| Arm | What the editor could do |
| --- | --- |
| A | Score public answers and write directly |
| B | A plus a temporary native-neuron intervention, identity check, and matching column-write recipe |
| C | A plus preview, commit, and discard for arbitrary supported weight edits |
| D | Both the internal intervention and the weight preview |

**The primary contrast was D minus C within each editor.** C and D had identical access to the full set of writable columns and could both evaluate any proposed numerical edit on the same public prompts before committing. D could also ask what happened when a particular gated feed-forward unit was scaled across all positions. The supplied example suggested suppressing the most active unit at the first prompt’s last token in layer 10. The editor could specify another unit or layer. The measured treatment includes the internal result, a ready-to-use write recipe, the extra schema text and the time spent using them. A matched random-neuron report would address whether the feature-selection rule itself helps. The benchmarked interface had no active SAE dictionary and included an identity control. [Tool implementation at launch](launch-harness.tar.gz)

We froze two previously unedited tasks from each of four qualified IWIBench test families. Eight tasks were the independent units. Qwen2.5-7B-Instruct ran each arm twice per task, with matched sampling seeds across arms and balanced order. Each attempt had six editor responses of up to 384 generated tokens. In a separate cohort, GPT-6.2-Astra ran C and D once per task, with six responses, low reasoning effort and a 1,500-token output cap including reasoning. Both used the same underlying tools, public packets, Gemma revision, target-forward limits, and explicit submission requirement. The two editors had different decoding budgets, so comparisons are made within each editor. [Runtime versions and hashes](environment.json) · [Task selector](../../examples/plan_preview_benchmark.py)

A separate qualified IWIBench process generated private calibration, target, and control answers on the pristine model and on a fresh target with only the submitted patch. The scorer uses the official deterministic rule for each task. The editor’s public answer check uses whitespace-trimmed exact match. A no-submission receives zero assigned reward. Retention and signed target/control changes are defined on the scored submissions. We independently checked pristine eligibility even when an editor failed to submit. Task-level differences average both Qwen seeds before pooling. The interval in the results resamples whole tasks with replacement, with 20,000 draws. It is a descriptive interval on a small, four-family sample. [Qualified grader](../../examples/qualified_eval.py) · [Independent calibrations](calibration) · [Analysis](../../examples/analyze_preview_benchmark.py)

## Reproduce the local analysis

The local evidence includes the exact public packets, tool schemas, raw Qwen generations with token IDs, executed tool traces, numerical patch and receipt for submissions, and aggregate qualified grades. Private grader prompts and raw private completions stay in the grader's working directory. Use an environment with NumPy to audit the saved trial records.

```sh
python examples/analyze_preview_benchmark.py \
  --design evidence/paired-preview-benchmark/design.json \
  --trials evidence/paired-preview-benchmark/trials \
  --calibration evidence/paired-preview-benchmark/calibration \
  --out runs/paired-preview-audit.json
python examples/diagnose_preview_editor.py evidence/paired-preview-benchmark
```

For another GPU run, provision the exact qualified IWIBench catalogue, Gemma and Qwen checkpoints listed in the design. The commands below assume the catalogue root is in `IWI_QUALIFIED_CATALOG` and both model revisions are cached. Use a fresh output directory for a new collection. The compressed source tree preserves the launch version of the harness as it was presented to both editors.

```sh
mkdir -p runs/paired-preview-reproduction/harness
tar -xzf evidence/paired-preview-benchmark/launch-harness.tar.gz -C runs/paired-preview-reproduction/harness
export PYTHONPATH="$IWI_QUALIFIED_CATALOG/src:$PWD/runs/paired-preview-reproduction/harness"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python examples/run_preview_benchmark.py edit --catalog "$IWI_QUALIFIED_CATALOG" --out runs/paired-preview-reproduction
python examples/calibrate_preview_benchmark.py --catalog "$IWI_QUALIFIED_CATALOG" --out runs/paired-preview-reproduction
python examples/run_preview_benchmark.py grade --catalog "$IWI_QUALIFIED_CATALOG" --out runs/paired-preview-reproduction
```
