# Does runtime SAE fitting improve weight edits?

Giving GPT-6.2 Astra access to `fit_sae` produced a tie on alias acquisition and two fewer correct held-out answers on resistance to false suggestions. Astra skipped the tool in both autonomous editing episodes. When explicitly asked to use it on the suggestion task, Astra trained a dictionary and finished between the two autonomous runs.

| IWIBench development task | Editing condition | Hidden targets correct | Previously correct controls retained | Native reward | Editor tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Alias, seed 1 | Fitting withheld | 3/3 | 69/70 | 0.9857 | 24,951 |
| Alias, seed 1 | Fitting available | 3/3 | 69/70 | 0.9857 | 26,126 |
| Suggestion, seed 2 | Fitting withheld | 49/64 | 110/110 | 0.5161 | 45,214 |
| Suggestion, seed 2 | Fitting available | 47/64 | 110/110 | 0.4516 | 45,338 |
| Suggestion, seed 2 | Explicitly asked to fit and inspect an SAE | 48/64 | 110/110 | 0.4839 | 55,310 |

Pristine Gemma scored 0/3 on alias and 33/64 on suggestion. Across the two autonomous pairs, mean episode reward was 0.751 with fitting withheld and 0.719 with fitting available. Control retention was 179/180 in each condition. The fit-enabled arm used 71,464 total editor tokens versus 70,165 without fitting, a 1.9% increase. Both totals include cached input. The offered tool added 300 input tokens to the first request on each task. All five patches received fresh native grades. The provider cost upper bound for the five trajectories was $4.02748, excluding Lambda compute. [Audited results](comparison.json)

## What the updater did

On alias acquisition, Astra used the target tokenizer to identify the relevant embedding rows and mixed those rows in both arms. On resistance to false suggestions, both autonomous editors inspected layer-33 residuals. Their weight operations used existing tensor rows and scalars. The fit-enabled run fixed 19 of the 31 initially wrong target answers and lost five answers Gemma had previously gotten right. The run without fitting fixed 17 and lost one. [Alias trace with fitting available](alias-1/fit/trajectory.jsonl) · [Suggestion trace with fitting available](suggestion-2/fit/trajectory.jsonl) · [Suggestion trace with fitting withheld](suggestion-2/no-fit/trajectory.jsonl)

In the instructed follow-up, Astra fit a 32-feature TopK SAE at layer 33. It compared feature activations on two pairs of public prompts with correct and incorrect suggestions. Its three weight writes scaled two existing output-token rows. The dictionary's held-out reconstruction error on two prompts was 13.2% higher than predicting the training mean. The final edit preserved all 33 previously correct target answers and fixed 15. It cost 22% more editor tokens than the same fit-enabled arm with the ordinary objective. This single trajectory shows feature use during an edit. It cannot isolate the contribution of the features to the resulting score. [Instructed trace](instructed-followup/suggestion-2/trajectory.jsonl) · [Fresh grade](instructed-followup/suggestion-2/grade.json)

## Design and reproduction

The host ran pinned Gemma 3 4B IT on a Lambda A100. Each editor started on fresh weights with an empty dictionary and the same six-turn, 2,048-output-token-per-response and $1.25 limits. Within each autonomous pair the public task packets and shared tool definitions matched. Only the fitting tool and its associated capacity differed. The additional instructed trajectory used precisely the fit arm's tools. The native grader replayed the submitted weight patches on a fresh target. The two seeds are previously admitted development examples, with one editor sample per condition. This study measures an availability effect under a short turn budget. The instructed trajectory is reported separately. [Matched design](design.json) · [Follow-up design](instructed-followup-design.json)

Each run directory contains the public input, offered schemas, action trace, usage, submitted patch, receipt and native grade. The following command audits the saved trials from the repository root.

```sh
python examples/summarize_sae_ab.py --root evidence/astra-sae-ab-v1 --out /tmp/sae-comparison.json
```

To collect a fresh pair, run the following twice with separate output paths, once with `--arm fit` and once with `--arm no-fit`. The driver uses the pinned IWIBench checkout, a cached Gemma checkpoint and a hosted updater credential. [Experiment driver](../../examples/astra_sae_ab.py)

```sh
python examples/astra_sae_ab.py --case suggestion --seed 2 --arm fit \
  --max-turns 6 --max-output-tokens 2048 --budget-usd 1.25 \
  --out runs/new-suggestion-2/fit
```
