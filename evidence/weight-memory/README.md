# Can a fresh model remember a sequence of new associations?

**A local Qwen updater taught Gemma 15 of 16 new word-to-country bindings.** Fresh Gemma instances that received only Qwen's numerical patches answered **45/48** held-out country questions. Exact-key retrieval and scripted row copying each scored **48/48**. Retrieval also preserved all **16/16** literal copies. Qwen preserved **1/16** and the scripted edits **0/16**. For example, all three returned *Brasília* as the capital of *candle*. Asked to repeat the word *candle*, both edited models answered *Brazil*. [Qwen answers](reads/qwen/sequence-1/stage-4.json) · [Scripted answers](reads/edited/sequence-1/stage-4.json) · [Retrieved-note answers](reads/retrieval/sequence-1/stage-4.json)

## Task and controls

We enrolled four independent sequences of four word-to-country bindings in Gemma 3 4B IT. For example, in the first sequence the word *candle* was assigned the meaning *Brazil*. Each enrollment provided only the new binding. The sixteen aliases and their sixteen referents are all distinct. In the neutral enrollment sentence each occupies one space-prefixed tokenizer row, a condition verified before collecting model outputs. The sequence, query forms, accepted answers, score, and decoding cap were frozen first. [Frozen design](design.json) · [Enrollment data seen by editors](enrollment.json)

After every enrollment, the grader asked about every binding acquired so far. Two held-out questions asked for the country's capital and a third asked for its continent. For each acquired alias it also requested a literal copy of the alias and the capital of the ordinary country name. Three general questions were repeated at every stage. Each question ran in a separate fresh chat. The final comparison contains **48 target questions and 44 controls per arm**. Across all stages each arm has 120 target reads and 128 control reads. The four sequences are the independent descriptive units. Repeated questions measure change as further bindings are added. [Query construction and scoring](../../examples/weight_memory_cases.py)

| Arm | Information carried between tasks | Information at read time |
| --- | --- | --- |
| Pristine | Original checkpoint only | Question |
| Exact-key retrieval | One plain-text glossary entry per enrolled alias | Question plus the single matching entry when a whole-word alias is present |
| Scripted weight edit | The harness copies the known country embedding row into the alias row, then exports the cumulative numerical patch | Question |
| Qwen-selected weight edit | Local Qwen2.5-7B-Instruct chooses tools and writes under a six-turn limit. Each stage starts with the previously deployed patch. The enrollment prompt offers row copying as one possible operation. | Question |

Retrieval starts at perfect keyword recall on the association questions. It also supplies the matching note for literal-copy requests. Bare country and unrelated queries retrieve nothing. The retrieval function sees only the question text and already-enrolled glossary. The scripted writer is a specialized reference implementation for these lexical bindings. It uses Gemma's context-resolved embedding addresses and the harness's ordinary copy and numerical patch exporter. The model-driven writer's public checks concern country identity and literal copying. It receives no capital or continent questions, answers, grader outputs, or access to the scoring process. [Frozen Qwen design](qwen-design.json) · [Scripted writer and evaluator](../../examples/run_weight_memory.py)

During evaluation we reload a pristine checkpoint for every patched sequence stage, verify the patch SHA-256 and the saved base fingerprints of all touched tensors, and apply the deployed numerical values. These fresh targets see only the question. All arms decode greedily up to 12 new tokens. Exact-match scoring uses casefolding, surrounding whitespace removal, and one optional final period. The public read records preserve each prompt and raw completion, generated IDs, logical model usage and physical source when an unchanged reference answer is reused. [Read records](reads/) · [Scripted snapshots](scripted-patches/) · [Qwen stage deployments](qwen/snapshots/) · [Environment](environment.json)

## Acquisition and side effects

| Arm | Final alias questions | All stages, alias questions | Final literal copies | Final ordinary country questions | Final unrelated controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pristine | 4/48 | 8/120 | 16/16 | 16/16 | 12/12 |
| Exact-key retrieval | 48/48 | 120/120 | 16/16 | 16/16 | 12/12 |
| Scripted weight edit | 48/48 | 120/120 | 0/16 | 16/16 | 12/12 |
| Qwen-selected weight edit | 45/48 | 114/120 | 1/16 | 16/16 | 12/12 |

| Final target score by sequence | Sequence 1 | Sequence 2 | Sequence 3 | Sequence 4 |
| --- | ---: | ---: | ---: | ---: |
| Pristine | 2/12 | 1/12 | 0/12 | 1/12 |
| Exact-key retrieval | 12/12 | 12/12 | 12/12 | 12/12 |
| Scripted weight edit | 12/12 | 12/12 | 12/12 | 12/12 |
| Qwen-selected weight edit | 12/12 | 12/12 | 9/12 | 12/12 |

Pristine's four correct final answers were all continent guesses. Retrieval and scripted editing retained all 36 older questions answered correctly at acquisition. Qwen retained all 33 that it had answered correctly at acquisition. All three final Qwen misses concerned *autumn* → *Germany*. In that lesson the editor exhausted six turns without submitting a patch, so the prior deployment carried forward. This also explains its one preserved literal copy. Fourteen of Qwen's sixteen cumulative stage snapshots matched the scripted patch byte for byte. The *autumn* stage and its successor had different snapshots. [Failed lesson](qwen/episodes/sequence-3/stage-3/trajectory.jsonl) · [Final answers after the failed lesson](reads/qwen/sequence-3/stage-4.json) · [Full results by stage, age and question](summary.json)

The task favors a row copy. The sixteen writes touch distinct alias rows and all countries were ones whose capitals pristine Gemma already knew. Literal copying tests a behavior that must survive the identical alias token gaining a new meaning. Broader entity memories, overlapping updates, multi-token keys and harder retrieval have yet to be measured.

## Cost at enrollment and at read time

| Arm | Gemma input tokens, final 48 alias questions | Generated tokens | Full-prefix tokens processed |
| --- | ---: | ---: | ---: |
| Pristine | 1,136 | 138 | 3,349 |
| Exact-key retrieval | 1,808 | 135 | 5,252 |
| Scripted weight edit | 1,136 | 114 | 2,772 |
| Qwen-selected weight edit | 1,136 | 119 | 2,892 |

The retrieved note “Personal glossary note: the word candle is an alias for Brazil.” and two newlines add **14 Gemma input tokens**. Every matching query in this experiment incurred the same increase. That is 672 across the 48 final alias questions, or 2,240 across the full 248-read schedule including literal-copy controls. A shorter note format could change this saving. All arms use the same tokenizer and chat template. The difference in processed tokens also includes answer length and the repeated input prefix on each greedy step. The deterministic reference arms reused identical queries within a sequence, with 92 physical generations each. Table entries charge every scheduled read.

| Arm | Gemma input tokens, all 248 reads | Generated tokens | Full-prefix tokens processed |
| --- | ---: | ---: | ---: |
| Pristine | 6,024 | 666 | 16,521 |
| Exact-key retrieval | 8,264 | 689 | 23,959 |
| Scripted weight edit | 6,024 | 606 | 14,998 |
| Qwen-selected weight edit | 6,024 | 616 | 15,238 |

The scripted writer performed sixteen direct row copies and no editor or target inference. Resolving row addresses, writing, hashing and exporting took 4.32 seconds in total, in addition to 14.67 seconds to load four base checkpoints. Serialized write arguments and return values measured 1,803 Gemma tokens. The four final sparse patches total 411,810 bytes. Exact retrieval stored sixteen text entries totaling 208 Gemma tokens and used whole-word matching on CPU. These wall times were measured on a shared GPU and exclude queueing.

Qwen used **248,110 input and 4,262 generated editor tokens** across sixteen lessons, including the unsuccessful one. Its edit sessions used 182 Gemma target forwards and processed an additional 4,176 Gemma tokens. The edit loops took 140.81 seconds total, excluding initial model loads, and the four final patches total 386,242 bytes. All model inference ran on the A100. [Editor usage and deployment audit](qwen-editor.json) · [Original editor trajectories](qwen/episodes/)

For scale, saving 14 Gemma prompt tokens on 18,027 matching reads would equal the raw count of 252,372 Qwen enrollment tokens. Dollar or GPU-time payback would require comparable service measurements, including the editor's larger model, its generated tokens, target probes, patch deployment, and any prompt caching. Retrieval also gave higher answer accuracy and preserved every literal copy in this test.

## Reproduce the audit

From the repository root, the audit runs on CPU using the Python standard library. It verifies the frozen cases and expected retrieval for every read, recomputes exact matches, confirms cumulative patch hashes and pristine-pair identity, reconciles generated IDs with full-prefix compute counts, and recomputes acquisition, forgetting and retention. It also checks recorded reuse of identical reference queries.

```sh
python examples/analyze_weight_memory.py --root evidence/weight-memory \
  --qwen-patches evidence/weight-memory/qwen/snapshots --out /tmp/weight-memory-summary.json
cmp /tmp/weight-memory-summary.json evidence/weight-memory/summary.json
python examples/audit_qwen_memory.py --root evidence/weight-memory --out /tmp/qwen-editor.json
cmp /tmp/qwen-editor.json evidence/weight-memory/qwen-editor.json
```

To collect a new run, cache the pinned Gemma and Qwen checkpoints and use a BF16-capable GPU with room for both. The original read worker used the earlier hashed harness source in the [launch archive](../paired-preview-benchmark/launch-harness.tar.gz). Run the commands below from the repository root. The optional `--gpu-lock PATH` on the evaluation commands shares one target-model slot with another process.

```sh
python examples/run_weight_memory.py enroll --out runs/memory-repro/scripted-patches
python examples/run_weight_memory.py evaluate --arm pristine --out runs/memory-repro/reads
python examples/run_weight_memory.py evaluate --arm retrieval --out runs/memory-repro/reads
python examples/run_weight_memory.py evaluate --arm edited \
  --patch-root runs/memory-repro/scripted-patches --out runs/memory-repro/reads
python examples/run_qwen_memory.py --out runs/memory-repro/qwen
python examples/run_weight_memory.py evaluate --arm qwen \
  --patch-root runs/memory-repro/qwen/snapshots --out runs/memory-repro/reads
```
