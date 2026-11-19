# Does mechanistic evidence help a model choose a better weight edit?

**Offering the native-neuron intervention did not produce a clear held-out reward gain.** We completed 80 editing attempts on the same eight qualified test tasks. Astra used the intervention on all eight treatment tasks and showed a mean advantage of 0.0018 on the 0–1 reward scale. Qwen earned zero in every condition. The compact weight interface and the internal measurement answer different product questions. An earlier matched study measured lower editor token traffic from compact tensor inventories. In this study, the extra intervention increased editor traffic for both editors. [Qwen audit](summary.json) · [Astra results](../paired-preview-astra/README.md) · [Earlier inventory comparison](../heldout-token-audit/README.md)

## Held-out results

| Editor and interface | Attempts | Submitted | Nonempty patches | Successful neuron test, sessions | Committed preview, sessions | Mean reward | Editor tokens per attempt | Target forwards per attempt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen A · behavior and direct writes | 16 | 3 | 2 | — | — | 0.0000 | 20,789 | 4.5 |
| Qwen B · add internal measurement | 16 | 5 | 3 | 6 | — | 0.0000 | 26,533 | 14.3 |
| Qwen C · add weight preview | 16 | 6 | 2 | — | 1 | 0.0000 | 22,490 | 2.6 |
| Qwen D · add both | 16 | 8 | 6 | 4 | 0 | 0.0000 | 29,218 | 6.8 |
| Astra C · weight preview | 8 | 8 | 8 | — | 8 | 0.4425 | 23,823 | 30.9 |
| Astra D · preview and internal measurement | 8 | 8 | 8 | 8 | 7 | 0.4442 | 26,858 | 41.9 |

The primary Astra comparison had one win, six ties and one loss. Its descriptive 95% task-bootstrap interval for the mean difference was **−0.120 to +0.125**. Both arms averaged 0.4583 normalized target gain. The preview-only arm retained **559/574** previously correct controls and the internal arm retained **550/574**. Each arm used 44 editor turns. Internal access cost an additional **3,035 editor tokens per task (+12.7%)**, almost entirely input. Hosted editor spend across all 16 attempts was **at most $8.30**. All graders ran locally. GPU billing is separate. [Hosted per-task audit](../paired-preview-astra/summary.json)

| Family and qualified variant | Astra C | Astra D | Qwen D minus C, averaged over two seeds |
| --- | ---: | ---: | ---: |
| Alias acquisition 42 | 0.6667 | 1.0000 | 0 |
| Alias acquisition 75 | 0.9857 | 0.6667 | 0 |
| Output code remapping 18 | 0.9506 | 0.9506 | 0 |
| Output code remapping 40 | 0.9367 | 0.9367 | 0 |
| Factual update 97 | 0 | 0 | 0 |
| Factual update 72 | 0 | 0 | 0 |
| Selective factual withholding 28 | 0 | 0 | 0 |
| Selective factual withholding 34 | 0 | 0 | 0 |

Qwen fixed zero hidden targets in its 22 submitted patches. Its 42 no-submissions also scored zero. Every task-level contrast, including preview versus direct writes, therefore landed at the floor. A bootstrap of these eight zero differences cannot establish equivalent performance on new tasks. Among submitted patches, arms A through D respectively retained 209/210, 292/359, 437/438 and 578/580 previously correct controls. If a deployment kept the pristine checkpoint on every no-submission, the corresponding figures across all attempts would be 1,147, 1,081, 1,147 and 1,146 controls out of 1,148 per arm. [All assigned attempts and signed transitions](summary.json)

## What the editors did with the measurements

- **Astra used the tool and chose other weights.** On both remapping tasks the temporary intervention changed generated text. Across eight treatment tasks it changed no public exact-match outcome, and Astra subsequently edited other parameters in all eight. On alias 42 the internal arm won after an embedding edit. On alias 75 it lost. The prompted default selected just three distinct layer-10 units across the eight tasks. These traces leave the value of other sites, feature-selection rules, and learned SAE features open. [Remapping trace](../paired-preview-astra/trials/output_code_remapping-018-r1-D/trajectory.jsonl) · [Alias trace](../paired-preview-astra/trials/alias_acquisition-042-r1-D/trajectory.jsonl)
- **Qwen sometimes followed a measurement into a real write.** It ran 16 successful interventions in 10 of the 32 sessions with access. Five sessions applied an exact returned column recipe on a later editor turn. On alias 75, suppressing layer-10 unit 0 shifted the public improvement margin by +0.4375 and the preserve margin by −0.3125. Both exact answers stayed correct. Qwen then zeroed the 2,560-weight column, later filled an embedding row with the scalar 45,601, and submitted. The joint patch scored zero and retained 4/70 controls. The trace confirms a measurement-to-write sequence. The final outcome includes both writes. [Tool trace](trials/alias_acquisition-075-r2-B/trajectory.jsonl) · [Fresh grade](trials/alias_acquisition-075-r2-B/grade.json)
- **Qwen often failed before evaluation.** Of 42 no-submissions, 22 used the last turn for another tool, 17 ended with malformed output, and three ended with plain text. Eleven malformed endings cut off inside a native tool call at the 384-token cap. Of 120 rejected behavior checks, 108 supplied invalid optional token IDs. Qwen committed a preview in just one of the 32 sessions where it could. A follow-up could give Qwen a short reference to the public cases and default to answer-only checks, then compare submission and reward under the same turn and token caps. [Raw decode audit](native-editor-diagnostics.json) · [Successful preview trace](trials/alias_acquisition-075-r2-C/trajectory.jsonl)

For a deployment decision, the compact interface has direct evidence of token savings. The tested native-neuron intervention is useful for inspection and can produce an executable edit recipe. These results favor keeping it optional when the objective is held-out editing reward.

## Study design

The editing model saw a public objective, task parameters, token IDs for public parameter spellings, public practice cases, and one preservation case that asked Gemma to copy “pencil.” All arms could generate answers, score editor-supplied improvement and preservation cases, read or write any supported parameter slice, and submit a patch. Every editor got the same instructions, including conditional guidance to use the tools available in its arm. The target was Gemma 3 4B IT on one A100. [Frozen local design](design.json) · [Frozen hosted design](../paired-preview-astra/design.json)

| Arm | What the editor could do |
| --- | --- |
| A | Score public answers and write directly |
| B | A plus a temporary native-neuron intervention, identity check, and matching column-write recipe |
| C | A plus preview, commit, and discard for arbitrary supported weight edits |
| D | Both the internal intervention and the weight preview |

**The primary contrast was D minus C within each editor.** C and D had identical access to the full set of writable columns and could both evaluate any proposed numerical edit on the same public prompts before committing. D could also ask what happened when a particular gated feed-forward unit was scaled across all positions. The supplied example suggested suppressing the most active unit at the first prompt’s last token in layer 10. The editor could specify another unit or layer. The measured treatment includes the internal result, a ready-to-use write recipe, the extra schema text and the time spent using them. A matched random-neuron report would address whether the feature-selection rule itself helps. A sequence of editing tasks with a retrieval-memory control would be needed to evaluate the longer-term memory objective. The benchmarked interface had no active SAE dictionary and included an identity control. [Tool implementation at launch](launch-harness.tar.gz)

We froze two previously unedited tasks from each of four qualified IWIBench test families. Eight tasks were the independent units. Qwen2.5-7B-Instruct ran each arm twice per task, with matched sampling seeds across arms and balanced order. Each attempt had six editor responses of up to 384 generated tokens. In a separate cohort, GPT-6.2-Astra ran C and D once per task, with six responses, low reasoning effort and a 1,500-token output cap including reasoning. Both used the same underlying tools, public packets, Gemma revision, target-forward limits, and explicit submission requirement. The two editors had different decoding budgets, so comparisons are made within each editor. [Runtime versions and hashes](environment.json) · [Task selector](../../examples/plan_preview_benchmark.py)

A separate qualified IWIBench process generated private calibration, target, and control answers on the pristine model and on a fresh target with only the submitted patch. Reuse of an identical episode and patch digest reused the verified generation cache. The scorer uses the official deterministic rule for each task. The numeric reward is normalized improvement on hidden targets multiplied by the fraction of previously correct controls retained. All eight pristine tasks passed the qualified calibration and started with zero correct hidden targets. Across them, 574 controls were correct before editing. The scorer recorded `admitted: false` for leaderboard aggregation on all eight, so the measured outcome is the continuous task reward for this pilot. The editor’s public answer check uses whitespace-trimmed exact match. A no-submission receives zero assigned reward. Retention and signed target/control changes are defined on the scored submissions. We independently checked pristine eligibility even when an editor failed to submit. Task-level differences average both Qwen seeds before pooling. The interval in the results resamples whole tasks with replacement, with 20,000 draws. It is a descriptive interval on a small, four-family sample. [Qualified grader](../../examples/qualified_eval.py) · [Independent calibrations](calibration) · [Analysis](../../examples/analyze_preview_benchmark.py)

## What changed after the protocol was frozen

The archived code and the tool menus in each trial record exactly what the editors saw. The current Python implementation additionally offers an optional activity-matched alternative unit, reports supported native layers, reuses identical public baselines within a weight revision, and stores the per-case evidence for a weight preview. The release also preserves signed zero in rollback and gives an actionable error for a malformed optional token pair. These changes and the Pi adapter have separate implementation tests. Their effect on the held-out editor scores has yet to be measured. [Current behavior implementation](../../src/iwi_harness/experiments.py) · [Pi adapter](../../examples/pi/README.md)

## Reproduce the local analysis

The local evidence includes the exact public packets, tool schemas, raw Qwen generations with token IDs, executed tool traces, numerical patch and receipt for submissions, and aggregate qualified grades. The models had function-call access only. The trusted editor host and the grader shared a machine. Raw private grader prompts and completions are excluded from the released editor evidence. Use an environment with NumPy to audit the saved trial records.

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
