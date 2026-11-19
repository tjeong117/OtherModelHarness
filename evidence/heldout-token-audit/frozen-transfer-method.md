# Frozen false-suggestion patches on the qualified test split

The three source patches were created on the earlier IWIBench false-suggestion development seed 2. The verifier replayed each fixed patch against all 15 false-suggestion tasks in the official calibrated test partition. This measures how a previously learned edit generalizes within its task family. It uses zero new editor tokens. The fresh editing pilot records the editor traffic required to learn new objectives. [Qualified dataset](https://huggingface.co/datasets/hcltech-robotics/iwibench/tree/300a9d7271d9e3a9dd5bfc14edcbceccb66d5966) · [Grader](../../examples/grade_qualified_suggestion.py)

| Seed-2 patch | Correct targets | Originally correct target answers retained | Controls retained | Mean qualified reward |
| --- | ---: | ---: | ---: | ---: |
| Pristine Gemma | 578/960 | 578/578 | 1,653/1,653 | 0.0000 |
| Harness, SAE fitting withheld | 624/960 | 537/578 | 1,652/1,653 | 0.1207 |
| Native IWIBench | 613/960 | 390/578 | 1,653/1,653 | 0.2649 |
| SAE-guided harness | 607/960 | 578/578 | 1,653/1,653 | 0.0644 |

The SAE-guided patch corrected 29 new targets and preserved all 578 previously correct targets. Its native comparator corrected 223 new targets and lost 188 previously correct ones. The patch with fitting withheld corrected 87 and lost 41. Across the 15 tasks, the guided patch tied native on reward eight times and scored lower seven times. Qualified reward normalizes gain by the initial headroom, clips negative gain at zero, and weights the 15 tasks equally. All 960 target prompt strings are unique. None appeared verbatim in the source target set, the source public practice or the editors' recorded public inputs to target tools. [Exact counts and paired task rewards](frozen-transfer.json)

Each catalogue row's stable ID, parameter SHA-256 and episode SHA-256 must match regeneration using the pinned qualified package. All 3,360 items across calibration, targets and controls use deterministic predicates in the qualified scorer. The verifier explicitly fails if any item needs a remote semantic verdict. Each arm loads pristine Gemma 3 4B in a separate GPU process and applies its hash-checked source patch. Within each process the verifier uses the official item order, greedy native generation, batch size of eight, 256-token generation cap and recorded per-item seeds for each task. The three edits share fresh pristine outputs. The run used an NVIDIA A100-SXM4-40GB, Python 3.12.16, PyTorch 2.10.0+cu128 and Transformers 4.57.6. Only the verifier writes prompt-specific generations to its private cache. [Per-task results and weights' hashes](frozen-transfer.json)

The same three weights can be replayed from a checkout with the Gemma checkpoint cached. Set `CATALOG` to the local copy of Hub revision `300a9d7271d9e3a9dd5bfc14edcbceccb66d5966`, containing `manifest.json`, `data/*.json`, `data/test.jsonl` and `src/iwibench/*.py`. From the repository root run the following commands.

```sh
export PYTHONPATH="$CATALOG/src:src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for arm in pristine native no-fit sae-guided; do
  python examples/grade_qualified_suggestion.py --catalog "$CATALOG" \
    --work runs/qualified-frozen-private --generate "$arm" || exit 1
done
python examples/grade_qualified_suggestion.py --catalog "$CATALOG" \
  --work runs/qualified-frozen-private --out runs/qualified-frozen-transfer.json
```

The grader checks the public Hub manifest's SHA-256, the included source and data files, each replayed source-patch receipt, and the content digest and item IDs of cached generations. The aggregate report records any discrepancies between the freshly measured pristine control denominators and the public release's original calibration.
