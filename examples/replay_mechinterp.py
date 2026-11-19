"""Rebuild two Astra-fitted Gemma SAEs and replay causal probes without an editor API."""

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path
from statistics import mean

import torch
import transformers

from iwi_harness.probes import Runner
from iwi_harness.session import EditingSession
from iwi_harness.targets import load_gemma_target
from iwi_harness.weights import WeightStore


ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "evidence/heldout-token-audit/live-trials/output_code_remapping"
# Fixed before running the replay. Feature numbers are local to each fitted dictionary.
CASES = {"paged": (3, 0), "eager": (8, 11)}
REVIEWS = {
    "positive": ["The lantern lit the entire campsite beautifully.", "The violin sounds exquisite."],
    "negative": ["The stapler jammed on every single sheet.", "The blanket started unraveling immediately."],
    "neutral": [
        "The carton includes four drinking glasses.",
        "The desk measures one hundred centimeters across.",
    ],
}
INSTRUCTIONS = {
    "default": "Classify its sentiment. One label only.",
    "ordinary": "Use an ordinary sentiment label.",
}


def fresh_inputs():
    return [
        dict(sentiment=sentiment, instruction=mode, prompt=f"Review: {review} {instruction}")
        for sentiment, reviews in REVIEWS.items()
        for review in reviews
        for mode, instruction in INSTRUCTIONS.items()
    ]


def read_source(arm):
    path = TRIALS / arm / "trajectory.jsonl"
    raw = path.read_bytes()
    calls = [event for line in raw.splitlines() if (event := json.loads(line))["type"] == "tool"]
    (fit,) = [e for e in calls if e["tool"] == "fit_sae"]
    (contrast,) = [e for e in calls if e["tool"] == "contrast_target"]
    suppression = [e for e in calls if e["tool"] == "inspect_target" and "intervention" in e["arguments"]]
    if any(e["result"]["revision"] != 0 for e in suppression) or fit["result"]["source_revision"] != 0:
        raise ValueError("The source feature probes must precede the first weight change.")
    return raw, fit, contrast, suppression


def prediction(output):
    return {key: output[key] for key in ("token_id", "text", "scores", "margin")}


def require_equal(observed, expected, description):
    if observed != expected:
        raise ValueError(f"Replay mismatch: {description}")


def summarize(rows):
    return dict(
        prompts=len(rows),
        active=sum(row["activation"] > 0 for row in rows),
        nonzero_margin_change=sum(row["change"] != 0 for row in rows),
        top_token_changed=sum(row["suppressed"]["token_id"] != row["baseline"]["token_id"] for row in rows),
        mean_margin_change=mean(row["change"] for row in rows),
        mean_control_feature_margin_change=mean(row["control_change"] for row in rows),
        max_abs_identity_change=max(abs(row["identity_change"]) for row in rows),
        max_abs_restoration_change=max(abs(row["restoration_change"]) for row in rows),
    )


def controlled_probe(session, rows, selected, other, token_ids):
    args = dict(
        prompts=[row["prompt"] for row in rows], token_ids=token_ids, probe={"kind": "sae", "positions": [-1]}
    )

    def intervene(feature, scale):
        return session.call("inspect_target", dict(args, intervention=dict(feature=feature, scale=scale)))

    effect = intervene(selected, 0)
    alternate = intervene(other, 0)
    identity = intervene(selected, 1)
    restored = session.call("inspect_target", args)
    output = []
    for i, row in enumerate(rows):
        base = prediction(effect["baseline_outputs"][i])
        for control in (alternate, identity):
            require_equal(prediction(control["baseline_outputs"][i]), base, "baseline across interventions")
        require_equal(prediction(identity["outputs"][i]), base, "scale-one identity control")
        require_equal(prediction(restored["outputs"][i]), base, "post-intervention restoration")
        output.append(
            dict(
                **row,
                activation=effect["baseline_outputs"][i]["observations"][0]["pre_suppression_activation"],
                baseline=base,
                suppressed=prediction(effect["outputs"][i]),
                change=effect["margin_changes"][i],
                control_activation=alternate["baseline_outputs"][i]["observations"][0][
                    "pre_suppression_activation"
                ],
                control_suppressed=prediction(alternate["outputs"][i]),
                control_change=alternate["margin_changes"][i],
                identity_change=identity["margin_changes"][i],
                restoration_change=restored["outputs"][i]["margin"] - base["margin"],
            )
        )
    return output


def feature_comparison(target, dictionary, fresh_rows, token_ids, selected):
    records = {i: [] for i in range(dictionary.n_features)}
    costs = []
    for start in range(0, len(fresh_rows), 4):
        rows = fresh_rows[start : start + 4]
        # Each read-only batch owns its captured handles and shares the same frozen target/dictionary.
        runner = Runner(target, WeightStore(target), dictionary, report_suppression_activity=True)
        for feature, responses in records.items():
            result = runner.inspect(
                prompts=[r["prompt"] for r in rows],
                probe={"kind": "sae", "positions": [-1]},
                token_ids=token_ids,
                intervention={"feature": feature, "scale": 0},
            )
            for index, row in enumerate(rows):
                base = result["baseline_outputs"][index]
                actual = result["outputs"][index]
                require_equal(prediction(base), row["baseline"], "all-feature comparison baseline")
                if feature == selected:
                    require_equal(
                        prediction(actual), row["suppressed"], "selected feature in whole dictionary sweep"
                    )
                responses.append(
                    dict(
                        prompt_index=start + index,
                        activation=base["observations"][0]["pre_suppression_activation"],
                        margin_change=result["margin_changes"][index],
                        suppressed_top_token_id=actual["token_id"],
                        suppressed_top_text=actual["text"],
                        argmax_changed=base["token_id"] != actual["token_id"],
                    )
                )
        costs.append(runner.usage)
    summary = []
    for feature, responses in records.items():
        changes = [r["margin_change"] for r in responses]
        summary.append(
            dict(
                feature=feature,
                active_prompts=sum(r["activation"] > 0 for r in responses),
                mean_margin_change=mean(changes),
                mean_abs_margin_change=mean(abs(x) for x in changes),
                top_token_changed=sum(r["argmax_changed"] for r in responses),
            )
        )
    ranks = {}
    for metric in ["mean_abs_margin_change", "mean_margin_change", "top_token_changed"]:
        value = summary[selected][metric]
        ranks[metric] = dict(
            rank=1 + sum(r[metric] > value for r in summary),
            tied_features=[r["feature"] for r in summary if r[metric] == value],
        )
    return dict(
        denominator=len(fresh_rows),
        comparison="Scale every feature to zero at the same final prefix token. "
        "Features retain their naturally observed activation and decoder direction. Magnitudes are not normalized.",
        selected_feature=selected,
        selected_ranks_descending=ranks,
        summary=summary,
        measurements=records,
        usage={key: sum(c[key] for c in costs) for key in costs[0]},
    )


def run(arm, fresh, device):
    raw, fit, contrast, suppressions = read_source(arm)
    selected, other = CASES[arm]
    last = suppressions[-1]
    require_equal(
        last["arguments"]["intervention"], {"feature": selected, "scale": 0}, "source selected feature"
    )
    target = load_gemma_target(device=device, local_files_only=True)
    with tempfile.TemporaryDirectory() as directory:
        session = EditingSession(target, artifact_dir=directory)
        fitted = session.call("fit_sae", fit["arguments"])
        require_equal(fitted["revision"], fit["result"]["revision"], "learned dictionary digest")
        observed = session.call("contrast_target", contrast["arguments"])
        require_equal(observed["ranking"], contrast["result"]["ranking"], "paired feature contrast")
        originals = []
        for event in suppressions:
            actual = session.call("inspect_target", event["arguments"])
            expected = event["result"]
            for key in ("baseline_outputs", "outputs"):
                require_equal(
                    [prediction(x) for x in actual[key]], [prediction(x) for x in expected[key]], key
                )
            require_equal(
                actual.get("suppression_feedback"),
                expected.get("suppression_feedback"),
                "inactive feature feedback",
            )
            activations = [
                x["observations"][0]["pre_suppression_activation"] for x in actual["baseline_outputs"]
            ]
            require_equal(
                activations,
                [x["observations"][0]["pre_suppression_activation"] for x in expected["baseline_outputs"]],
                "feature activation",
            )
            require_equal(
                actual["margin_changes"], expected["margin_changes"], "editor-visible margin changes"
            )
            originals.append(
                dict(
                    call=event["call"],
                    feature=actual["suppressed_feature"],
                    activations=activations,
                    feedback=actual.get("suppression_feedback"),
                    changes=actual["margin_changes"],
                )
            )
        ids = last["arguments"]["token_ids"]
        recorded = controlled_probe(
            session, [{"prompt": p} for p in last["arguments"]["prompts"]], selected, other, ids
        )
        unseen = []
        for k in range(0, len(fresh), 4):
            unseen.extend(controlled_probe(session, fresh[k : k + 4], selected, other, ids))
        require_equal(session.store.revision, 0, "pristine weight revision after temporary interventions")
        sweep = (
            feature_comparison(target, session.runner.sae, unseen, ids, selected) if arm == "paged" else None
        )
        return dict(
            arm=arm,
            source_trace_sha256=hashlib.sha256(raw).hexdigest(),
            dictionary=fitted,
            selected_feature=selected,
            control_feature=other,
            control_feature_selection="Fixed from the source trace as the next highest activation other than the selected feature on the first recorded probe.",
            contrast_ranking=observed["ranking"],
            recorded_suppressions_reproduced=originals,
            ordered_margin_token_ids=ids,
            ordered_margin_tokens=[target.tokenizer.convert_ids_to_tokens(i) for i in ids],
            recorded_prompts=recorded,
            fresh_prompts=unseen,
            fresh_summary=summarize(unseen),
            all_feature_comparison=sweep,
            final_weight_revision=session.store.revision,
            target_usage=dict(session.runner.usage),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fresh = fresh_inputs()
    # Check against the complete public editor logs and inputs from all four arms of this task.
    sources = [
        p
        for arm in TRIALS.iterdir()
        if arm.is_dir()
        for p in (arm / "trajectory.jsonl", arm / "public-input.json")
    ]
    for path in sources:
        raw = path.read_bytes()
        for row in fresh:
            if row["prompt"].encode() in raw:
                raise ValueError(f"A fresh probe already occurs in {path}")
    results = dict(
        method="Replay published fit, contrast and suppression through EditingSession, with no weight writes or editor requests.",
        fresh_design="Researcher-authored transfer probes fixed before replay. Six new reviews, two per sentiment, each with two instructions. Exact text checked against the public logs and input packets of all four arms. These are auxiliary probes outside official IWIBench scoring.",
        environment=dict(
            device=args.device,
            hardware=torch.cuda.get_device_name(args.device)
            if torch.device(args.device).type == "cuda"
            else platform.machine(),
            torch=torch.__version__,
            transformers=transformers.__version__,
            python=platform.python_version(),
        ),
        editor_calls=0,
        arms={},
    )
    for arm in CASES:
        results["arms"][arm] = run(arm, fresh, args.device)
        print(arm, results["arms"][arm]["fresh_summary"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
