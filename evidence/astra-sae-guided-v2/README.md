# Prompting Astra to use an SAE before editing

GPT-6.2 Astra fitted an SAE, suppressed a feature, and submitted an edit to Gemma. The patch tied native IWIBench on the source task and both transfer tasks. It used 35,008 editor tokens on the source episode, versus 680,003 for the saved native run. The fit-disabled harness scored one answer higher on the source and one answer lower across the two transfer episodes.

| Seed-2 editing policy | Source targets | Seed 12 targets | Seed 13 targets | Originally correct transfer targets retained | Source editor tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pristine Gemma | 33/64 | 37/64 | 32/64 | 69/69 | — |
| Harness with fitting withheld | 49/64 | 48/64 | 32/64 | 66/69 | 45,214 |
| Native IWIBench | 48/64 | 49/64 | 32/64 | 67/69 | 680,003 |
| Harness with causal SAE prompt | 48/64 | 49/64 | 32/64 | 69/69 | 35,008 |

Every edited model retained all 110 previously correct source controls and all 220 previously correct transfer controls. Guided and native each earned 0.4839 reward on the source and 0.2222 mean reward across the transfer episodes. The fit-disabled harness earned 0.5161 and 0.2037 respectively. Each transfer row reuses a fixed source patch. No editor saw the transfer tasks. [Source grades, usage and checks](source-comparison.json) · [Transfer grades and overlap checks](transfer.json)

## What changed in the prompt

The `--sae-guided` option asks the updater to fit a dictionary on balanced public examples, identify a feature, suppress it on a case to improve and a case to preserve, use the measured changes in competing answer logits to guide a weight edit, and check full answers. The editor receives the ordinary tool menu. [Prompt implementation](../../src/iwi_harness/updater.py) · [Actual editor input](suggestion-2/public-input.json)

Astra selected layer 29 and fit a 12-feature dictionary at the last prompt position. On two validation prompts, it reduced reconstruction error by 78.9% relative to the training mean. It then chose feature 5 from a contrast on public example pairs and suppressed it on the two validation prompts. The feature had zero activation on both. Each answer margin moved by exactly 0. The editor subsequently copied the embedding row for “ verify” into the row for “ confirm”. The public practice feedback followed that write. The trace provides no evidence that a causal signal from feature 5 improved the patch. [Actual trace](suggestion-2/trajectory.jsonl) · [Submitted patch and receipt](suggestion-2/receipt.json)

We reconstructed the exact dictionary and tested the features that were active. Suppressing feature 2 moved the first-token margin by up to 3.0 logits on the editor's examples and 1.75 on four new public examples. Every greedy first token remained “agree”. This replay examined the original weights and made no edits. [Per-prompt measurements](feature-check.json) · [Reproduction](../../examples/check_sae_guided_features.py)

After this run, we changed the adaptive `inspect_target` reply to report the exact selected feature activation at each inspected position. When every activation is zero, the reply asks the editor to select a nonzero feature and repeat the measurement. The check reuses existing observations and adds no target forwards. Regression tests cover an inactive feature and an active feature that produces zero logit change. The score effect of the revised feedback remains unmeasured. [Tool implementation](../../src/iwi_harness/probes.py) · [Tests](../../tests/test_dictionary_registry.py)

## What the comparison measures

This study used one newly sampled trajectory, after the source seed had been used for development. The guided run had five turns, a 1,536-token output limit per response and a $0.97 cost ceiling. The fit-disabled comparator had six turns and a 2,048-token output limit. The native comparator had twelve turns and a 4,096-token limit. Total editor tokens include cached input. Uncached input was 8,311 guided, 6,497 fit-disabled and 60,859 native. The new trajectory's hosted cost upper bound was $0.72326. Its provider responses identify GPT-6.2 Astra. The historical native trajectory records the requested model and its recovered usage. The provider model echo is unavailable. These protocols compare achieved results and cost under different limits. [Predeclared design](design.json)

The verifier replayed the frozen patches from seed 2 on the first two pristine-eligible seeds in the range 12–20. Both seeds 12 and 13 were eligible. All 128 transfer target prompts were distinct and differed from the source private prompts, supplied public practice prompts and the three editors' recorded public tool inputs. Seventeen non-target private prompt strings recur between the two transfer seeds. On seed 13, none of the three patches improved target accuracy over pristine Gemma. [Verifier](../../examples/grade_sae_transfer.py)

### Auditing editor token usage

| Provider usage on the source episode | SAE-guided | Fitting withheld | Native IWIBench |
| --- | ---: | ---: | ---: |
| Editor requests | 5 | 6 | 12 |
| Cached input | 25,542 | 37,681 | 615,303 |
| Uncached input | 8,311 | 6,497 | 60,859 |
| Generated output, including reasoning | 1,155 | 1,036 | 3,841 |
| Total across first five requests | 35,008 | 35,560 | 257,707 |
| Total across the episode | 35,008 | 45,214 | 680,003 |

Native included all 883 tensor entries in its opening manifest. Its first request alone used 49,487 input tokens. The harness previewed four tensors and let the editor page through the rest, and the guided run started with 4,934 input tokens. The provider counted the growing conversation on every turn, including cached input. Comparing the first five requests shows a large difference between interfaces and a 552-token difference between the two harness runs. These prefixes represent different trajectories and can be compared for traffic without assigning them the final grades. Target generation, SAE fitting and transfer grading are separate from this editor usage. [Guided trace](suggestion-2/trajectory.jsonl) · [No-fit trace](../astra-sae-ab-v1/suggestion-2/no-fit/trajectory.jsonl) · [Native trace](native-seed-2/trajectory.jsonl)

Run the source audit from the repository root.

```sh
python examples/summarize_sae_guided.py --run evidence/astra-sae-guided-v2/suggestion-2 \
  --out /tmp/sae-source-comparison.json
```

Repeat transfer grading on the GPU with the pinned target and IWIBench checkout.

```sh
python examples/grade_sae_transfer.py --work runs/sae-transfer-private --out runs/sae-transfer.json \
  --arm no-fit=evidence/astra-sae-ab-v1/suggestion-2/no-fit \
  --arm native=evidence/astra-sae-guided-v2/native-seed-2 \
  --arm sae-guided=evidence/astra-sae-guided-v2/suggestion-2
```
