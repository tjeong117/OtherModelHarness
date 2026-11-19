"""Audit the source-seed result of the SAE-guided editor and three historical baselines."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from compare_astra_arms import episode, read_optional
from iwi_harness.updater import sae_guided_objective
from summarize_sae_ab import audit_arm


ROOT = Path(__file__).resolve().parents[1]
CASE = "suggestion"
FAMILY = "resistance_to_false_suggestions"
SEED = 2


def measured_suppression(path):
    measurements = []
    dictionaries = {}
    for line in (path / "trajectory.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["type"] != "tool" or "result" not in e:
            continue
        tool, args, obs = e["tool"], e["arguments"], e["result"]
        if tool == "fit_sae":
            n_train = obs["training"]["prompts"]
            training = args["prompts"][:n_train]
            validation = args.get("validation_prompts", args["prompts"][n_train:])
            dictionaries[obs["dictionary_id"]] = (set(training), set(validation), obs["top_k"])
            continue
        probe = obs.get("probe") or {}
        if probe.get("kind") != "sae" or tool not in ("inspect_target", "feature_examples"):
            continue
        if tool == "inspect_target":
            intervention = args.get("intervention") or {}
            if intervention.get("scale") != 0 and intervention.get("value") != 0:
                continue
            feature = intervention["feature"]
            if "margin_changes" not in obs:
                continue
            paired = zip(obs["baseline_outputs"], obs["outputs"], obs["margin_changes"], strict=True)
            rows = [(i, before["margin"], after["margin"], change) for i, (before, after, change) in enumerate(paired)]
        else:
            feature = args["feature"]
            rows = [(x["prompt_index"], x["suppression"]["baseline_margin"], x["suppression"]["suppressed_margin"],
                     x["suppression"]["margin_change"]) for x in obs["examples"] if "suppression" in x]
        for index, before, after, change in rows:
            if not math.isclose(after - before, change, abs_tol=1e-8):
                raise ValueError("suppression margin shift disagrees with recorded baseline and intervention")
            dictionary = probe["dictionary_id"]
            if dictionary not in dictionaries:
                raise ValueError("suppression did not use a previously fitted dictionary")
            training, validation, sparsity = dictionaries[dictionary]
            prompt = args["prompts"][index]
            split = "training" if prompt in training else "validation" if prompt in validation else "new_public_prompt"
            activations = []
            if tool == "inspect_target":
                for observation in obs["baseline_outputs"][index]["observations"]:
                    top = {x["index"]: x["value"] for x in observation["top"]}
                    # A TopK code has at most k positive values. A short top list can leave activity unknown.
                    value = top.get(feature, 0.0 if len(top) >= sparsity else None)
                    activations.append({"position": observation["position"], "activation": value})
            else:
                ex = next(x for x in obs["examples"] if x["prompt_index"] == index)
                activations.append({"position": ex["position"], "activation": ex["activation"]})
            measurements.append(dict(call=e["call"], tool=tool, feature=feature, dictionary_id=dictionary,
                                     layer=probe["layer"], positions=probe.get("positions"), prompt_index=index,
                                     fit_prompt_split=split, baseline_feature_activations=activations,
                                     token_ids=args["token_ids"], baseline_margin=before, suppressed_margin=after,
                                     margin_change=change))
    return measurements


def checked_receipt(path, row):
    if row["editor_status"] != "submitted":
        return
    receipt = read_optional(path, "receipt.json")
    editor_receipt = read_optional(path, "editor-result.json").get("receipt")
    with (path / "patch.safetensors").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != receipt.get("sha256") or (editor_receipt and editor_receipt != receipt):
        raise ValueError(f"submitted patch or receipt mismatch in {path}")
    if row["grade_status"] in ("scored", "ineligible") and digest != row.get("patch_sha256"):
        raise ValueError(f"graded patch differs from submitted patch in {path}")


def summarize(run, previous, native, design_path):
    plan = read_optional(design_path.parent, design_path.name)
    if (plan["case"], plan["source_development_seed"]) != (FAMILY, SEED):
        raise ValueError("unexpected study or source seed")
    dirs = {"guided": run, "no_fit": previous / "suggestion-2/no-fit",
            "first_instructed": previous / "instructed-followup/suggestion-2", "native": native}
    guided, guided_schemas, _ = audit_arm(run, CASE, FAMILY, SEED, "fit", expected_protocol="runtime-sae-guided-v2")
    instructed, fit_schemas, _ = audit_arm(dirs["first_instructed"], CASE, FAMILY, SEED, "fit",
                                           expected_protocol="runtime-sae-instructed-v1")
    no_fit, control_schemas, _ = audit_arm(dirs["no_fit"], CASE, FAMILY, SEED, "no-fit")
    rows = {"guided": guided, "no_fit": no_fit, "first_instructed": instructed, "native": episode(native, FAMILY, SEED)}
    for name, row in rows.items():
        if row["integrity_issues"] or row["grade_status"].startswith("missing"):
            raise ValueError(f"{name}: {row['grade_status']} {row['integrity_issues']}")
        checked_receipt(dirs[name], row)
        if row["grade_status"] == "scored" and row["pristine"] != no_fit["pristine"]:
            raise ValueError(f"{name}: pristine calibration or group denominators differ")
    expected = {"requested_model": plan["updater"], "reasoning": plan["reasoning"],
                "max_turns": plan["max_turns"], "max_output_tokens": plan["max_output_tokens"],
                "budget_cap_usd": plan["provider_budget_usd"], "iwibench_revision": no_fit["settings"]["iwibench_revision"]}
    protocol = read_optional(run, "protocol.json")
    if guided["settings"] != expected or protocol["session"] != read_optional(dirs["first_instructed"], "protocol.json")["session"]:
        raise ValueError("guided editor settings differ from the design or reference target")
    if protocol["session"]["model_id"] != plan["target"]:
        raise ValueError("guided checkpoint differs from the design")
    if fit_schemas != guided_schemas or [s for s in guided_schemas if s["function"]["name"] != "fit_sae"] != control_schemas:
        raise ValueError("tool schemas differ outside fitting access")
    base = read_optional(dirs["no_fit"], "public-input.json")
    if (base | {"objective": sae_guided_objective(base["objective"], plan["max_turns"])}) != read_optional(run, "public-input.json"):
        raise ValueError("guided editor packet differs beyond the declared policy")
    instruction = read_optional(previous, "instructed-followup-design.json")["instruction"]
    if (base | {"objective": base["objective"] + " " + instruction}) != read_optional(dirs["first_instructed"], "public-input.json"):
        raise ValueError("earlier instructed packet differs beyond its declared instruction")
    summaries = {name: {k: row[k] for k in ("protocol", "settings", "editor_status", "grade_status", "reason",
                                              "reward", "counts", "patch_sha256", "provider", "warnings")}
                 for name, row in rows.items()}
    summaries["guided"].update({k: guided[k] for k in ("actual_user_prompt_sha256", "tool_schemas_sha256",
                                                      "fitted_dictionaries", "successful_model_tool_counts",
                                                      "successful_feature_uses", "successful_writes",
                                                      "writes_by_source_kind", "target_usage_including_host")})
    summaries["guided"]["measured_suppressions"] = measured_suppression(run)
    deltas = {name: guided["reward"] - row["reward"] if guided["grade_status"] == row["grade_status"] == "scored" else None
              for name, row in rows.items() if name != "guided"}
    return {"study": "SAE-guided policy on its source development episode", "family": FAMILY, "seed": SEED,
            "design_sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
            "runs": summaries, "native_reward_difference_guided_minus_comparator": deltas,
            "interpretation": plan["interpretation"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=ROOT / "evidence/astra-sae-ab-v1")
    parser.add_argument("--native", type=Path, default=ROOT / "evidence/astra-sae-guided-v2/native-seed-2")
    parser.add_argument("--design", type=Path, default=ROOT / "evidence/astra-sae-guided-v2/design.json")
    args = parser.parse_args()
    result = summarize(args.run, args.previous, args.native, args.design)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
