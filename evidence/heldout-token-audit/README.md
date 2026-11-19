# Editor token use and held-out edit quality on IWIBench

On five qualified held-out tasks, the same GPT-6.2 Astra editor used **150,673 tokens** with four tensor entries shown up front, **157,557** with a compact description of every writable tensor, and **358,448** with a flat list of those same tensors. All three harness layouts earned the same reward on every task. The strongest matched control puts the observed saving from paging at **4.4%**. Stock IWIBench used **1,269,323** tokens. That 88.1% difference measures the complete interfaces. [Two-page evaluation brief](Held-out%20evaluation.docx) · [Audited results](live-qualified.json) · [Public trajectories and submitted patches](live-trials/)

## Fresh editing results

Astra edited the same pristine Gemma 3 4B checkpoint in 20 independent sessions. We selected one official test task from each of the five families that support fully deterministic grading. Every arm used all five of its editor requests on each task. Totals include cached input, uncached input and output, including reasoning. Cached input is part of the token count and can receive a billing discount. The API column bounds editor charges at full input rates. The 20 attempts together cost **at most $39.09484** against the additional $100 authorization. The qualified grader used zero hosted judge calls. Lambda compute is separate. [Study design and task IDs](live-design.json) · [Grouped follow-up design](grouped-live-design.json)

| Interface shown to Astra | Total editor tokens | Uncached input | Output | Submissions | Mean reward on all 5 | API upper bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock IWIBench | 1,269,323 | 253,038 | 4,193 | 5/5 | 0.3708 | $25.47 |
| Harness, all 445 names and shapes in a flat list | 358,448 | 36,075 | 5,760 | 4/5 | 0.3838 | $7.28 |
| Harness, all 445 grouped by repeated layers | 157,557 | 24,237 | 4,855 | 4/5 | 0.3838 | $3.25 |
| Harness, first four with paging | 150,673 | 23,511 | 3,933 | 4/5 | 0.3838 | $3.09 |

Paging saved 207,775 tokens (58.0%) against the flat list and 6,884 (4.4%) against grouping. It used fewer tokens on all five tasks against flat and on three of five against grouped. Uncached input fell 34.8% and 3.0%, respectively. The editor also used target-side compute. The harness logged 10,081 target forward tokens with paging, 10,282 with grouping and 11,358 with the flat list. The native worker logged 1,697 under its own metering conventions. SAE fitting, target inference, and verifier work are outside the hosted editor token count. [Per-turn usage and target counters](live-qualified.json)

| Held-out task | Paged tokens | Grouped tokens | Paged reward | Grouped reward | Flat reward | Stock reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Resistance to false suggestions | 34,506 | 37,268 | 0* | 0* | 0* | 0 |
| Output code remapping | 28,909 | 30,983 | 0.9333 | 0.9333 | 0.9333 | 0.8542 |
| Alias acquisition | 29,517 | 28,085 | 0.9857 | 0.9857 | 0.9857 | 1.0000 |
| Factual update | 29,956 | 29,861 | 0 | 0 | 0 | 0 |
| Selective factual withholding | 27,785 | 31,360 | 0 | 0 | 0 | 0 |

\* All three harness editors exhausted their turns on false suggestions before calling `submit_patch`. Each received the prespecified zero for that attempt. Stock IWIBench automatically submitted the current weights, which had zero changes. Each arm submitted four patches with actual changes on the remaining tasks. Across those four tasks, all three harness layouts corrected 51 of 57 initially incorrect targets. Paged and flat retained 288/297 previously correct controls, grouped retained 289/297, and stock retained 289/297. Stock corrected 48 of the same 57 targets. The five-task reward averages weight tasks equally. [Individual grades, receipts and patches](live-trials/)

## Which control answers which question

**Paging versus a compact full inventory** is the closest test of the product decision. The grouped inventory expands in order to the same 445 names and shapes, including the tied output alias. The three harness arms shared the tools, public practice feedback, SAE instruction, lookup access, preparation tool costs, five-turn limit, 1,536-token output limit, and $6.50 per-run ceiling. The grouped arm adds instructions for reading its lossless layer ranges. We specified it after discovering the compact encoding and before inspecting fresh grades. It ran after the other arms, so its cache history and sampling time differed. [Grouped representation and round trip](grouped-manifest.json) · [Matched editor driver](../../examples/astra_qualified_layout.py)

**Stock IWIBench** answers how much editor traffic the shipped interfaces use under the same editor, tasks and limits. Its four tools, original prompt, public feedback and automatic submission differ from the harness. The gap includes these differences. The six earlier development episodes also favored the harness on tokens. Matching prefix lengths within each task gave 263,920 versus 3,359,604 tokens across 62 requests per arm (92.1% savings). On those tasks both achieved 132/201 correct targets, while the compact harness preserved 36/97 initially correct targets versus 62/97 for stock. [Development trace audit](live-development.json) · [Stock qualified driver](../../examples/astra_qualified_native.py)

**SAE availability** needs a separate comparison. All 15 harness editors fitted an SAE, contrasted features and suppressed one. Four switched features after receiving feedback that their first feature was inactive. Thirteen saw a nonzero change in a token-pair logit margin before choosing their first weight write. The same instruction to use an SAE appeared in every layout, so the layout contrast holds that policy fixed. [Tool calls and feature measurements](tool-behavior.md)

Earlier paired development runs varied SAE availability, and the transfer test below compares their frozen source patches. The package has demonstrated lower editor traffic. Whether SAE access improves newly learned edits across families or whether weight edits outperform retrieval-based memory across future tasks remains open. [Earlier SAE ablation](../astra-sae-ab-v1/README.md)

## Coverage of the official test split

The official catalogue contains 150 tasks, 15 in each of ten families. All use the same qualified Gemma checkpoint. We rendered the public task, common instructions and same 13 tool schemas for each one, varying only the inventory representation. These measurements use `o200k_base` before provider framing. The stable inventory gives exactly the same absolute saving across the 150 openings. The fresh editing and grading sample above covers five distinct tasks. [Official partition](https://huggingface.co/datasets/hcltech-robotics/iwibench/blob/300a9d7271d9e3a9dd5bfc14edcbceccb66d5966/manifest.json) · [Per-task counts and reproduction](opening-context.md)

| Inventory at opening | Mean opening tokens across 150 tasks | Extra tokens versus paging |
| --- | ---: | ---: |
| Full stock-style metadata for 883 records | 50,914 | 46,745 |
| All writable names and shapes, flat | 12,120 | 7,951 |
| All writable names and shapes, grouped | 4,356 | 187 |
| First four writable names and shapes | 4,169 | 0 |

Grouping the 34 repeated transformer layers recovers 97.6% of the flat-list to four-entry opening difference while exposing the complete inventory. On the common static opening, paging saved 65.6% versus flat and 4.29% versus grouped. In the actual five live pairs, the provider reported respectively 7,951 and 192 fewer first-request input tokens for paging on every task. JSON placement and the formatting instruction differ slightly between the offline and live renderings. [Lossless encoding and verification](grouped-manifest.json)

## Transfer of previously learned weights

We separately replayed three fixed false-suggestion patches from development seed 2 on **all 15** qualified test tasks in that family. This is 960 unique target prompts with zero new editor calls. Pristine Gemma answered **578/960** correctly. The **SAE-guided patch reached 607** and preserved all 578 previously correct targets. The **no-fit harness patch reached 624** and preserved 537. The **stock patch reached 613** and preserved 390. Mean qualified reward was 0.0644 guided, 0.1207 no-fit, and 0.2649 stock. The benchmark normalizes gain by remaining headroom and clips negative gain per task, which explains why its ordering can differ from pooled accuracy. All conditions used the same freshly measured pristine outputs. On three tasks, the number of initially correct controls differed by one from the public calibration. [Transfer results, retention, overlap checks and methods](frozen-transfer-method.md)

## Interpreting and reproducing the result

The five fresh task pairs establish a measured token saving on this editor and checkpoint. One sample per condition, identical official reward on the three layouts, and no prespecified quality-equivalence margin leave the population-level quality tradeoff uncertain. A broader claim would need repeated editors on additional held-out tasks across the other five families, with interleaved arm order and a prespecified acceptable quality difference. A claim about continually learning future tasks would also need task sequences and a matched memory baseline.

The archive contains public model inputs, tool schemas, calls and observations, per-response usage, patch receipts, submitted patches and aggregate qualified grades. Provider response identifiers were removed. The audit reproduces the same report from the release and from the original records. With the harness and experiment dependencies installed, run this from the repository root. [Auditor](../../examples/summarize_qualified_live.py) · [Independent qualified grader](../../examples/qualified_eval.py)

```sh
PYTHONPATH=src python examples/summarize_qualified_live.py \
  --root evidence/heldout-token-audit/live-trials \
  --out /tmp/iwi-live-qualified.json --require-complete
cmp /tmp/iwi-live-qualified.json evidence/heldout-token-audit/live-qualified.json
```
