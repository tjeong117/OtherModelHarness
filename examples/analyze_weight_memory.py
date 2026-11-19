"""Verify the sequential memory evidence and summarize accuracy, forgetting and read costs."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from weight_memory_cases import cases, correct, fetch_note, load


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def proportions(rows):
    return {"correct": sum(r["correct"] for r in rows), "total": len(rows)}


def costs(rows):
    measured = [r for r in rows if r["row_id"] == r["physical_origin"]]
    fields = ("input_tokens", "output_tokens", "forward_tokens", "forwards", "note_added_prompt_tokens")
    return dict(logical_reads=len(rows), physical_reads=len(measured),
                logical={k: sum(r[k] for r in rows) for k in fields},
                physical={k: sum(r[k] for r in measured) for k in fields},
                physical_generation_seconds=sum(r["generation_seconds"] for r in measured))


def analyze(root, qwen_patches=None):
    design = load(root / "design.json")
    design_sha = digest(root / "design.json")
    arms = ["pristine", "retrieval", "edited"]
    if qwen_patches is not None:
        arms.append("qwen")
    base = {}
    reports = {}
    for arm in arms:
        all_rows, all_stages = [], []
        seen = {}
        for seq in design["sequences"]:
            size = len(seq["associations"])
            for stage in range(1, size + 1):
                path = root / "reads" / arm / seq["id"] / f"stage-{stage}.json"
                result = load(path)
                if (result["design_sha256"], result["arm"], result["sequence"], result["stage"], result["checkpoint"]) != (
                        design_sha, arm, seq["id"], stage, design["revision"]):
                    raise ValueError(f"read identity mismatch: {path}")
                if arm in ("edited", "qwen"):
                    patch_dir = (qwen_patches if arm == "qwen" else root / "scripted-patches") / seq["id"] / f"stage-{stage}"
                    receipt = load(patch_dir / "receipt.json")
                    if not (result["replay"]["sha256"] == result["patch_sha256"] == receipt["sha256"] ==
                            digest(patch_dir / "patch.safetensors")):
                        raise ValueError(f"patch or replay receipt mismatch: {path}")
                    if (result["replay"]["replaced_parameters"] != receipt["changed_parameters"]
                            or result["checkpoint_load_seconds"] <= 0 or result["patch_load_seconds"] <= 0):
                        raise ValueError(f"incomplete fresh-checkpoint replay: {path}")
                elif result["patch_sha256"] is not None:
                    raise ValueError(f"unexpected patch: {path}")
                expected = cases(design, seq, stage)
                if len(result["rows"]) != len(expected):
                    raise ValueError(f"missing or extra questions in {path}")
                for row, item in zip(result["rows"], expected, strict=True):
                    key = (seq["id"], stage, item["id"])
                    rowid = f'{seq["id"]}/{stage}/{item["id"]}'
                    if any(row[k] != v for k, v in item.items()) or row["row_id"] != rowid or rowid in seen:
                        raise ValueError(f"item mismatch or duplicate: {path}")
                    if (row["arm"], row["sequence"], row["stage"]) != (arm, seq["id"], stage):
                        raise ValueError(f"row identity mismatch: {path}")
                    ni, no = row["input_tokens"], row["output_tokens"]
                    if (not 1 <= no <= design["max_new_tokens"] or len(row["output_ids"]) != no or
                            row["forwards"] != no or row["forward_tokens"] != ni * no + no * (no - 1) // 2):
                        raise ValueError(f"inconsistent model usage: {path}")
                    if row["correct"] != correct(row["text"], row["answers"]):
                        raise ValueError(f"exact score mismatch: {path}")
                    note = fetch_note(item["prompt"], seq["associations"][:stage], design["note_template"]) if arm == "retrieval" else ("", None)
                    if (row["note"], row["retrieved_alias"]) != note:
                        raise ValueError(f"note differs from exact-key retrieval: {path}")
                    if arm == "pristine":
                        base[key] = row
                    if row["input_tokens"] - base[key]["input_tokens"] != row["note_added_prompt_tokens"]:
                        raise ValueError(f"unexpected contextual input difference: {path}")
                    if row["physical_origin"] != rowid:
                        origin = seen[row["physical_origin"]]
                        cache_fields = ("sequence", "note", "prompt", "output_ids", "text", "input_tokens",
                                        "output_tokens", "forwards", "forward_tokens", "generation_seconds")
                        if (arm not in ("pristine", "retrieval") or origin["row_id"] != origin["physical_origin"]
                                or any(origin[k] != row[k] for k in cache_fields)):
                            raise ValueError(f"cache mismatch: {path}")
                    seen[rowid] = row
                    all_rows.append(row)
                all_stages.append(result)
        final_stage = max(r["stage"] for r in all_rows)
        targets = [r for r in all_rows if r["group"] == "target"]
        final = [r for r in targets if r["stage"] == final_stage]
        intro = {(r["sequence"], r["id"]): r for r in targets if r["stage"] == r["introduced"]}
        forgetting = {}
        for label, subset in (("all", final), ("older_only", [r for r in final if r["introduced"] < final_stage])):
            initially = [r for r in subset if intro[r["sequence"], r["id"]]["correct"]]
            incorrect = [r for r in subset if not intro[r["sequence"], r["id"]]["correct"]]
            forgetting[label] = dict(initially_correct=len(initially), lost=sum(not r["correct"] for r in initially),
                                     retained=sum(r["correct"] for r in initially), initially_wrong=len(incorrect),
                                     gained=sum(r["correct"] for r in incorrect))
        controls = {}
        for scope, scoped in (("all_stages", all_rows), ("final", [r for r in all_rows if r["stage"] == final_stage])):
            controls[scope] = {}
            for group in ("copy", "canonical", "unrelated"):
                subset = [r for r in scoped if r["group"] == group]
                initially = [r for r in subset if base[r["sequence"], r["stage"], r["id"]]["correct"]]
                controls[scope][group] = {**proportions(subset), "pristine_correct": len(initially),
                                          "retained_pristine_correct": sum(r["correct"] for r in initially)}
        per_seq = {s["id"]: proportions([r for r in final if r["sequence"] == s["id"]])
                   for s in design["sequences"]}
        form = {q["id"]: proportions([r for r in final if r["form"] == q["id"]]) for q in design["questions"]}
        age = {str(a): proportions([r for r in final if final_stage-r["introduced"] == a]) for a in range(final_stage)}
        per_stage = {str(i): proportions([r for r in targets if r["stage"] == i]) for i in range(1, final_stage + 1)}
        reports[arm] = dict(final_targets=proportions(final), all_stage_targets=proportions(targets),
                            final_by_sequence=per_seq, final_by_age=age, final_by_question=form,
                            targets_by_stage=per_stage, forgetting=forgetting, controls=controls,
                            all_reads_costs=costs(all_rows), final_target_costs=costs(final),
                            all_targets_costs=costs(targets), checkpoint_load_count=sum(r["checkpoint_load_seconds"] > 0 for r in all_stages),
                            checkpoint_load_seconds=sum(r["checkpoint_load_seconds"] for r in all_stages),
                            patch_replay_seconds=sum(r["patch_load_seconds"] for r in all_stages))
    edited = []
    writer_load_seconds = 0
    for seq in design["sequences"]:
        sdir = root / "scripted-patches" / seq["id"]
        enrollment = load(sdir / "enrollment-results.json")
        if enrollment["design_sha256"] != design_sha or len(enrollment["stages"]) != len(seq["associations"]):
            raise ValueError("incomplete enrollment evidence")
        writer_load_seconds += enrollment["model_load_seconds"]
        for stage, (row, binding) in enumerate(zip(enrollment["stages"], seq["associations"], strict=True), 1):
            receipt = load(sdir / f"stage-{stage}" / "receipt.json")
            if (row["stage"], row["alias"], row["country"], row["receipt"]) != (stage, binding["alias"], binding["country"], receipt):
                raise ValueError("scripted enrollment mismatch")
            edited.append(row)
    enrollment_metrics = {k: sum(r[k] for r in edited) for k in (
        "llm_editor_input_tokens", "llm_editor_output_tokens", "target_forwards_for_edit", "enrollment_seconds",
        "tool_payload_tokens_gemma_tokenizer", "note_storage_tokens")}
    enrollment_metrics["enrollments"] = len(edited)
    enrollment_metrics["checkpoint_load_seconds"] = writer_load_seconds
    enrollment_metrics["checkpoint_load_count"] = len(design["sequences"])
    enrollment_metrics["final_patch_bytes"] = sum(r["receipt"]["bytes"] for r in edited if r["stage"] == final_stage)
    enrollment_metrics["all_snapshot_bytes"] = sum(r["receipt"]["bytes"] for r in edited)
    return dict(design_sha256=design_sha, enrollment=enrollment_metrics, arms=reports)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--qwen-patches", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.root, a.qwen_patches)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
