# Hosted editor paired check

This cohort tests whether offering a native neuron intervention improves an Astra editor's submitted weight patch. Both arms can run public behavioral tests, preview a candidate on the live target, discard or commit it, and make direct writes. D also permits a reversible gated-MLP neuron intervention within the same behavioral test. The tested target is the pinned Gemma 3 4B IT checkpoint.

Astra used the intervention on all eight tasks. Adding it changed mean qualified reward by **+0.0018** on the 0–1 scale. D won one task, tied six, and lost one. The descriptive 95% task-bootstrap interval was −0.120 to +0.125. All 16 attempts submitted nonempty patches and received native grades. The hosted editor's total cost upper bound was **$8.30** across both arms. [Audited task-level results](summary.json)

| Measure | C · Preview | D · Preview and intervention |
|---|---:|---:|
| Mean qualified reward | 0.4425 | 0.4442 |
| Mean qualified target gain | 0.4583 | 0.4583 |
| Qualified controls retained | 559/574 (97.4%) | 550/574 (95.8%) |
| Mean editor tokens per task | 23,823 | 26,858 |
| Editor turns across eight tasks | 44 | 44 |
| Mean target forwards per task | 30.9 | 41.9 |
| Tasks with a committed preview | 8/8 | 7/8 |

Both reward differences occurred on the alias tasks. D scored 1.000 versus C's 0.667 on variant 42, and 0.667 versus C's 0.986 on variant 75. The interventions changed the generated text on two remapping tasks. All eight interventions left public exact-match correctness unchanged. Astra subsequently edited other weights on every D task. The automatic layer-10 selection returned just three distinct units across the eight tasks. No trial recorded a tool error. [Alias 42 treatment trace](trials/alias_acquisition-042-r1-D/trajectory.jsonl) · [Remapping 18 treatment trace](trials/output_code_remapping-018-r1-D/trajectory.jsonl)

The additional 24,281 editor tokens in D comprised 24,016 input tokens and 265 output tokens. The provider counted 169,120 cached input tokens in C and 189,269 in D. The spend ceiling charged conservative upper bounds for every input token. [Provider accounting](summary.json) · [Budget ledger](budget.json)

## Design and reproduction

The frozen design assigns one C/D pair to each of eight qualified IWIBench test tasks across four families. Four tasks run D first, with opposite orders for the two tasks in each family. The task packets, target limits, six editor turns, and harness source match the local Qwen factorial. Astra uses low reasoning effort and a maximum of 1,500 output tokens per request, including reasoning. Its Responses transport reports model usage for the entire message history on every request. The reported dollar amount is a conservative upper bound from those token counts. The A100 also served the Qwen cohort, so elapsed seconds include shared-device contention and hosted API latency.

The archived harness is the exact experimental version. A later change to live weight transactions preserves the sign bit on zero-valued floats during rollback and patch replay.

The editor sees public practice prompts, a preservation prompt, the objective, and token IDs for public task parameters. The native grader replays the submitted patch on a fresh target and uses deterministic IWIBench checks. The cohort shares pristine generations and per-task calibration with the Qwen run. The evaluator keeps hidden generations outside the published trial packets.

The launch runner records a provider response ID for each editor turn. The public trial files omit those 88 IDs. They retain each turn's provider usage and the tool-call IDs that connect requests to observations. Running the audit on either the original records or the public copies produces the same summary.

- [Frozen assignments and policy](design.json)
- [Audited results and task-level outcomes](summary.json)
- [Public trajectories, provider usage, submitted patches, and native grades](trials/)
- [Shared launch source archive](../paired-preview-benchmark/launch-harness.tar.gz)
- [Independent pristine task calibrations](../paired-preview-benchmark/calibration/)
- [Runtime versions and source hashes](environment.json) and [exact hosted launch script](launch-runner.py.gz)
- [Hosted runner](../../examples/astra_preview_benchmark.py)
- [Audit implementation](../../examples/analyze_astra_preview.py)

To reproduce the evidence audit from the repository root, run the command below with the project's Python environment.

```sh
python examples/analyze_astra_preview.py \
  --root evidence/paired-preview-astra \
  --calibration evidence/paired-preview-benchmark/calibration \
  --out /tmp/astra-preview-summary.json
```
