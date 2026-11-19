# What the editors did with interpretability tools

All 15 SAE-guided harness runs fitted a sparse autoencoder on public prompts, contrasted prompt pairs in its feature space, and temporarily suppressed a selected feature. The editors chose layers 25–33 and dictionaries of 8–16 features. On 14 runs, setting the feature to zero changed the logit difference for an editor-selected token pair on at least one probe prompt. On six, it changed the most likely next token. These measurements describe the immediate intervention on the probed inputs. [Per-run measurements](tool-behavior.json)

| Across five held-out tasks per arm | Paged | Flat eager | Grouped eager | Native IWIBench |
|---|---:|---:|---:|---:|
| Runs with an SAE fit and contrast | 5 | 5 | 5 | Unavailable |
| Feature suppression calls | 6 | 7 | 6 | Unavailable |
| Runs with a nonzero margin change | 5 | 4 | 5 | Unavailable |
| Runs with a changed next-token prediction | 3 | 1 | 2 | Unavailable |
| Weight writes using a residual activation handle / successful writes | 0/7 | 2/8 | 2/8 | 2/8 |
| Patches submitted | 4 | 4 | 4 | 5 |

**Did the editor act on the result?** Four first attempts selected a feature inactive at every tested position. Each editor received that feedback, picked a different feature, and measured a nonzero margin change. Thirteen of the 15 harness editors had seen a nonzero margin change before choosing their first weight write. In the flat eager remapping run, the editor requested the successful retry and all three writes in the same response, so those writes were already specified when the retry returned. The flat eager alias run measured zero change on both prompts. [Tool requests, turn numbers and results](tool-behavior.json)

Four of 23 harness writes used layer-33 residual captures directly as operands, on factual update and selective withholding in the flat eager and grouped eager arms. None used an SAE feature capture as an operand. The public harness responses contain tool calls without an accompanying written rationale. The SAE measurements could have influenced the choice of tensor, direction or scale. To attribute an improvement to those measurements would require a matched run varying SAE access or feedback. The native editors also inspected activations on all five tasks, and two of their eight writes used returned residuals directly. [Calls and operand origins](tool-behavior.json)

All five paged editors completed their attempted edits without requesting another manifest page. For resistance to false suggestions, all three harness editors spent a turn retrying an inactive feature and made one write on the fifth, final turn. They exited without submitting a patch. The native editor made no write on this task and the native driver exported the unchanged model at the turn limit. The traces establish the sequence and the difference in submission rules. They do not establish whether skipping the retry would improve the grade. [Complete trajectories](live-trials/)

Run `python examples/audit_qualified_behavior.py` from the repository root to regenerate the JSON from the packaged trajectories. Its counts separate editor-requested calls from host manifest reads and automatic practice checks. Native call numbers are one-based tool-execution ordinals. Harness call numbers come from the session log and include host operations. [Audit script](../../examples/audit_qualified_behavior.py)
