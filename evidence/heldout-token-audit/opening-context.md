# Opening-context token use on the IWIBench test catalogue

Showing the first four writable tensors rather than listing all 445 writable names and shapes saved 7,951 opening input tokens on each of the 150 tasks. The aggregate reduction was 65.60%. Giving the same editor the full native Gemma manifest raised the difference to 46,745 tokens per task, or 91.81% of opening input. Both comparisons used the same task, tool schemas, checkpoint metadata and instruction in each pair. [Measured totals and individual task counts](opening-context.json)

| Tensor entries shown at opening | Entries | Mean opening tokens | Total across 150 openings |
| --- | ---: | ---: | ---: |
| Full native records | 883 | 50,914.32 | 7,637,148 |
| Writable native records | 444 | 25,251.32 | 3,787,698 |
| All writable names and shapes (including the tied output head) | 445 | 12,120.32 | 1,818,048 |
| First four writable names and shapes | 4 | 4,169.32 | 625,398 |

## What was held fixed

Each task provided its official public objective, parameters and practice examples. We used the pinned public test JSONL and verified its SHA-256 against the release manifest. It contains 150 unique tasks across ten families, 15 per family, all qualified for the same checkpoint. We measured only the opening packet. The later editing trajectory, cached billing, terminal grader, and success or failure of the episode are outside this measurement. [Dataset and partition](https://huggingface.co/datasets/hcltech-robotics/iwibench/blob/300a9d7271d9e3a9dd5bfc14edcbceccb66d5966/manifest.json)

The full native manifest was captured from the pinned Gemma checkpoint in the native seed-2 editing run. It contains 883 canonical entries, repeated names in alias fields and 439 read-only entries. The opening comparison changes only the value of the manifest's tensor list. The harness exposes the tied output head as a separate writable name, yielding 445 discoverable names. The strongest paging control compares all versus four writable entries in the same compact name-and-shape representation. The 13 common tool schemas include the same paginated `read_weights` tool in every condition. A model that needs more entries can request them in later turns. The replay does not measure the cost of such requests. [Manifest snapshot](gemma-manifest.json.gz) · [Common tool schemas](../astra-sae-guided-v2/suggestion-2/tools.json) · [Qualified worker manifest implementation](https://huggingface.co/datasets/hcltech-robotics/iwibench/blob/300a9d7271d9e3a9dd5bfc14edcbceccb66d5966/src/iwibench/runtime.py)

We counted the shared instruction, serialized function menus and public user packet separately with `o200k_base`, then summed their token counts. Exact sizes for the same fields in UTF-8 are in the JSON. The public user packet is a controlled rendering of catalogue fields. Provider-specific framing and provider metering require a live request or a provider counting endpoint. The fixed Gemma manifest explains why every task gives the same absolute difference. The variation in opening lengths comes from the tasks' public content.

## Reproduce the counts

Run the following command with the project's token-counting extra installed. It downloads the pinned public file, or accepts a previously downloaded copy with `--catalog FILE`.

```sh
python examples/audit_heldout_context.py --out /tmp/iwibench-opening-context.json
```

The captured manifest's uncompressed compact JSON is 180,801 bytes, SHA-256 `469d006bfa0e956850b41ffdc0f31d158971291f860729da4521f74726d5fb36`. The same manifest method is present in the native worker used for capture and in the pinned qualified release. The script validates the manifest and dataset before counting. [Reproduction script](../../examples/audit_heldout_context.py)
