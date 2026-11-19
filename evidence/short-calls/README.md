# Can Qwen use shorter public checks to make better edits?

Qwen never used the omitted-case shortcut in a successful call. It supplied cases on all six successful behavior checks and its one successful preview in the short arm. Across the 16 attempts in each arm, the short interface used **288 more editor tokens per attempt**. Both arms earned zero native reward on all seven eligible development tasks. [Per-attempt outcomes, costs and paired grades](summary.json)

| Measurement | Explicit cases | Short calls available |
|---|---:|---:|
| Editor attempts | 16 | 16 |
| Submissions / nonempty patches | 5 / 4 | 2 / 0 |
| Successful behavior checks / attempted | 6 / 6 | 6 / 7 |
| Successful checks using exactly the full public case list | 6 | 2 |
| Successful previews / attempted | 3 / 9 | 1 / 9 |
| Committed previews | 3 | 0 |
| Attempts ending on malformed model output | 10 | 11 |
| Editor turns | 48 | 47 |
| Editor input / output tokens | 242,697 / 9,369 | 245,316 / 11,358 |
| Mean total editor tokens per attempt | 15,754.1 | 16,042.1 |
| Target forwards during editing | 148 | 181 |
| Mean native reward across 14 eligible attempts | 0 | 0 |
| Correct hidden targets with pristine fallback | 0 / 188 | 0 / 188 |
| Previously correct controls retained with pristine fallback | 1,022 / 1,022 | 1,022 / 1,022 |

**What one call could save.** We separately serialized one equivalent full public behavior check for each of the eight tasks. Its compact JSON arguments consumed 55–168 Qwen tokens (mean 83.125). The empty arguments `{}` consumed one token. The editor still saw the public examples in its opening packet. In live runs, the new schema and metadata added 38 input tokens to every opening request. The total episode increase was 4,608 tokens (1.83%). These counts use Qwen's pinned tokenizer. Input counts include the conversation processed again at each turn. [Argument-only comparison](argument-tokens.json) · [Observed trajectory totals](summary.json)

**What failed.** Eight explicit-arm responses and eleven short-arm responses hit the 384-token ceiling with an incomplete native tool-call block. Nine of those 19 raw responses included a `token_ids` argument. In one short-arm factual attempt, Qwen repeated the cases and a long list of IDs until the response ended. The parser could dispatch none of that response. Successful dispatch also left an editing problem. On output remapping variant 11, the explicit editor previewed one changed scalar in the first attention query projection, observed zero fixes on two public improvement cases, committed it, and scored 0/32 hidden targets. [Truncated native response](trials/factual_update-062-r1-short/native-generations.jsonl) · [Preview and commit](trials/output_code_remapping-011-r2-explicit/trajectory.jsonl) · [Native grade](trials/output_code_remapping-011-r2-explicit/grade.json)

Eight malformed responses contained 64 complete, syntactically valid tool-call blocks alongside a malformed block. The Python Qwen parser rejected each entire response, dropping 33 otherwise parseable blocks in the explicit arm and 31 in the short arm. Their arguments and intended edits were never executed. Even the short-arm eligible alias submission used explicit cases to preview an embedding edit, observed zero public fixes, and submitted the original weights. [Parser used in the run](launch-source.tar.gz) · [Alias preview and submission](trials/alias_acquisition-005-r1-short/trajectory.jsonl) · [Per-attempt counts](summary.json)

All seven eligible task-level reward differences were zero under the chosen Qwen editor, sampling settings and caps. A floor on all 28 scores limits what this pilot can establish about relative edit quality. It directly measures whether making cases optional changed this editor's behavior. The experiment covers development tasks on one target, with two replicates per task. [Paired task results](summary.json)

## Protocol

The local Qwen2.5-7B-Instruct editor edited a fresh Gemma 3 4B IT for each attempt. The control required `cases` in `test_behavior` and `preview_weights`. The short-call interface also accepted `test_behavior({})` and `preview_weights({"writes": [...]})`, sourcing the cases from the host's public packet. Qwen could supply custom cases and explicit two-token margins in either arm. Both arms received the same task packet, preservation example, grouped tensor inventory, direct writes, preview/commit tools and instructions to omit cases when optional. Internal measurements were disabled in both. The tool schemas and short arm's default-case count in session metadata communicated the interface difference. [Public packets](public-rows.json) · [Explicit menu](trials/output_code_remapping-011-r2-explicit/tools.json) · [Short menu](trials/output_code_remapping-011-r2-short/tools.json)

We fixed two tasks per family by sorting a salted hash of catalogue task IDs within four deterministic families. Candidates had an eligible upstream baseline, no historical equivalent archive ID and a meaningful task digest absent from the entire official test partition. All eight selected tasks belong to the **training split**. Held-out scoring here refers to private examples within those fresh development tasks. Each task had two independent sampling seeds, matched across arms. Arm order reversed on the second replicate. The A100 ran BF16 Qwen with at most six turns and 384 generated tokens per turn, temperature 0.4 and top-p 0.9. Gemma's public tools had the same 512-forward and 131,072-forward-token limits in both arms. [Frozen assignments and model revisions](design.json) · [Launch environment and source hashes](environment.json)

The native grader regenerated each task at the pinned catalogue commit, ran pristine calibration and evaluated each distinct submitted patch on a clean target using private prompts. Alias variant 58 failed fresh calibration because pristine Gemma missed a calibration item. Its four editor attempts count toward tool usage and cost. The frozen rule excludes that task from native reward without replacement, leaving seven eligible tasks and 14 eligible attempts per arm. Eligible attempts that ended without submission receive zero. Their deployment fallback for pooled target/retention counts is pristine Gemma. The grader used deterministic task predicates throughout. Hosted editor and judge API spend was $0. GPU time was incurred on the shared A100. [Calibration records](calibration) · [Grader implementation](../../examples/grade_short_calls.py)

## Reproduce

The saved public traces include the editor's generated text and token IDs, all tool calls and returns, receipts and submitted numerical patches. The archive records the exact source used for execution. One packaging startup error was corrected before the first editor generated a token. The original assignment and seed were retained. The audit checks launch hashes, identical public packets within tasks, paired seeds, tool menus, receipt hashes, all native grade statuses and token counts from actual generated IDs. [Launch source](launch-source.tar.gz) · [Raw attempts](trials) · [Audit script](../../examples/analyze_short_calls.py)

From the repository root, reproduce the aggregate from the recorded evidence on CPU.

```sh
python examples/analyze_short_calls.py --root evidence/short-calls --out /tmp/short-call-summary.json
```

To rerun model execution, provision the revisions of Qwen and Gemma in `design.json` and set `IWI_QUALIFIED_CATALOG` to the root of the qualified IWIBench release at `300a9d7271d9e3a9dd5bfc14edcbceccb66d5966`. Use a new output directory.

```sh
export STUDY_ROOT="$PWD/runs/short-calls-reproduction"
mkdir -p "$STUDY_ROOT/launch"
tar -xzf evidence/short-calls/launch-source.tar.gz -C "$STUDY_ROOT/launch"
export PYTHONPATH="$IWI_QUALIFIED_CATALOG/src:$STUDY_ROOT/launch/src:$STUDY_ROOT/launch/examples"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python "$STUDY_ROOT/launch/examples/plan_short_calls.py" --catalog "$IWI_QUALIFIED_CATALOG" --out "$STUDY_ROOT"
cmp "$STUDY_ROOT/design.json" evidence/short-calls/design.json
python "$STUDY_ROOT/launch/examples/grade_short_calls.py" calibrate --design "$STUDY_ROOT/design.json" --catalog "$IWI_QUALIFIED_CATALOG" --out "$STUDY_ROOT"
python "$STUDY_ROOT/launch/examples/run_short_calls.py" --design "$STUDY_ROOT/design.json" --out "$STUDY_ROOT"
python "$STUDY_ROOT/launch/examples/grade_short_calls.py" grade --design "$STUDY_ROOT/design.json" --catalog "$IWI_QUALIFIED_CATALOG" --out "$STUDY_ROOT"
python examples/analyze_short_calls.py --root "$STUDY_ROOT" --count-arguments --out "$STUDY_ROOT/argument-tokens.json"
```
