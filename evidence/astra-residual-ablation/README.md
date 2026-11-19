# What did the captured activations contribute?

| False-suggestion seed | Targets correct without → with residual term | Cases fixed by term | Cases broken by term | Native reward without → with |
|---|---:|---:|---:|---:|
| 2 | 45/64 → 48/64 | 10 | 7 | 0.3871 → 0.4839 |
| 5 | 32/64 → 31/64 | 29 | 30 | 0.0000 → 0.0000 |

The residual term changed the actual output text on 19 of 64 targets on seed 2 and 60 of 64 on seed 5. All 112 control completions per episode stayed textually identical. Native reward clips negative gains at zero; seed 5’s identical rewards conceal 59 changes in correctness. [Seed 2 paired results](suggestion-2/comparison.json) · [Seed 5 paired results](suggestion-5/comparison.json)

| Seed | Correct on true suggestions, without → with | Correct on false suggestions, without → with | Number of “agree” verdicts, without → with |
|---|---:|---:|---:|
| 2 | 22/32 → 32/32 | 23/32 → 16/32 | 30/64 → 48/64 |
| 5 | 30/32 → 0/32 | 2/32 → 31/32 | 60/64 → 0/64 |

Correctness requires both the right verdict and the right number. Every target response in these pairs parsed successfully. On seed 2 the term made Gemma more likely to agree. On seed 5 the term made Gemma disagree on all 64 target prompts, including the 32 true suggestions.

In the six full-tool IWIBench development episodes, Astra made 17 residual inspection calls. Eleven of the 13 false-suggestion inspections also requested logits for candidate tokens. Two of the 33 successful weight writes consumed activation handles. Both added the difference between final-layer residuals on a false and a true public suggestion to Gemma’s tied embedding/output row 2763, the “dis” prefix in “disagree.” The two prompts in each difference also contained different arithmetic problems.

We replayed the full recorded writes for both episodes on the pinned BF16 Gemma model. We regenerated the residuals on the same model revision at which Astra had captured them. Both replayed patch files matched their original SHA-256. The comparison removed only the scaled residual difference from the composed write. It kept the other operand, the same edit coefficient on that operand and all later actions. Holding the editor’s actions fixed isolates the direct effect of the numerical term. Estimating how seeing the observations affected those decisions would require sampling the editor again.

For each seed we ran three fresh native grading workers on exactly the same items and item seeds, one for the pristine target, one for the original patch and one for the patch with the term removed. Regrading the original had to reproduce every stored native metric before comparing the two variants. Group counts in each `comparison.json` distinguish changes in exact output text from changes in graded correctness.

## Reproduction

With the pinned Gemma checkpoint cached on a GPU and the pinned IWIBench checkout installed, run the following from the harness root. Change both paths to `suggestion-5` for the second episode. The source directories include the original public tool trace, native patch, receipt, protocol and aggregate grade.

```sh
python examples/ablate_residual_writes.py \
  --episode evidence/astra-residual-ablation/suggestion-2 \
  --out runs/residual-replay/suggestion-2
python examples/grade_residual_ablation.py \
  --episode evidence/astra-residual-ablation/suggestion-2 \
  --out runs/residual-replay/suggestion-2
```
