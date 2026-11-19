"""Render the fresh Gemma suppression measurements. Only this figure needs Matplotlib."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


DEMO = Path(__file__).resolve().parents[1] / "evidence" / "mechinterp-demonstration"
INK = "#172d35"
MUTED = "#52636a"
TEAL = "#08786f"
ORANGE = "#ab5524"
GRID = "#dce3e5"
REVIEWS = {
    "The lantern lit the entire campsite beautifully.": "Lantern · positive",
    "The violin sounds exquisite.": "Violin · positive",
    "The stapler jammed on every single sheet.": "Stapler · negative",
    "The blanket started unraveling immediately.": "Blanket · negative",
    "The carton includes four drinking glasses.": "Carton · neutral",
    "The desk measures one hundred centimeters across.": "Desk · neutral",
}
INSTRUCTIONS = {
    "default": ("Classify its sentiment. One label only.", TEAL, "o", -0.12),
    "ordinary": ("Use an ordinary sentiment label.", ORANGE, "s", 0.12),
}


def render(source, output):
    raw = source.read_bytes()
    data = json.loads(raw)["arms"]["paged"]
    feature = data["selected_feature"]
    rows = {row["prompt"]: row for row in data["fresh_prompts"]}
    sweep = data["all_feature_comparison"]
    if len(rows) != len(REVIEWS) * len(INSTRUCTIONS) or sweep["denominator"] != len(rows):
        raise ValueError("Expected all twelve paired fresh prompts in both panels")
    if data["ordered_margin_tokens"] != ["▁birch", "Positive"] or feature != 3:
        raise ValueError("This figure describes the birch-minus-Positive margin for feature 3")
    comparisons = sorted(sweep["summary"], key=lambda x: -x["mean_abs_margin_change"])
    if {r["feature"] for r in comparisons} != set(range(8)):
        raise ValueError("Expected all eight distinct learned features")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    fig, (paired, all_features) = plt.subplots(1, 2, figsize=(12.4, 5.5),
                                               gridspec_kw={"width_ratios": [1.28, 1]})
    fig.subplots_adjust(left=0.145, right=0.968, bottom=0.27, top=0.73, wspace=0.32)
    fig.text(0.035, 0.947, "What changes when Gemma’s learned feature 3 is suppressed?",
             size=19, weight="bold", va="top")
    fig.text(0.035, 0.866, "Gemma 3 4B  ·  Layer 30  ·  Final prompt token  ·  "
             "Next-token margin = logit(“ birch”) − logit(“Positive”)", color=MUTED, size=11)

    for ax in (paired, all_features):
        ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, pad=6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    paired.set_title("Feature 3 on six new reviews", loc="left", fontsize=13, weight="bold", pad=14)
    flips = []
    for index, (review, label) in enumerate(REVIEWS.items()):
        for mode, (instruction, color, marker, offset) in INSTRUCTIONS.items():
            row = rows[f"Review: {review} {instruction}"]
            if row["instruction"] != mode:
                raise ValueError(f"Unexpected instruction on {review}")
            change = row["change"]
            y = index + offset
            paired.hlines(y, 0, change, color=color, alpha=0.55, linewidth=1.4)
            paired.scatter(change, y, color=color, marker=marker, s=46, zorder=3)
            if row["baseline"]["token_id"] != row["suppressed"]["token_id"]:
                paired.scatter(change, y, facecolors="none", edgecolors=INK, marker="o",
                               s=152, linewidths=1.2, zorder=4)
                flips.append(f"{label.split(' · ')[0]} ({mode})   "
                             f"{row['baseline']['text']} → {row['suppressed']['text']}")
    paired.set_yticks(range(len(REVIEWS)), REVIEWS.values())
    paired.set_ylim(len(REVIEWS) - 0.50, -0.55)
    paired.set_xlabel("Change in margin (logits)", labelpad=9)
    paired.axvline(0, color="#95a2a6", lw=1, zorder=1)
    paired.margins(x=0.07)

    all_features.set_title("Each feature on all 12 prompts", loc="left", fontsize=13,
                           weight="bold", pad=14)
    for index, record in enumerate(comparisons):
        magnitude = record["mean_abs_margin_change"]
        all_features.barh(index, magnitude, height=0.58,
                          color=TEAL if record["feature"] == feature else "#94a5af")
        all_features.annotate(f"{magnitude:.2f}", (magnitude, index), (5, 0),
                              textcoords="offset points", va="center", fontsize=9)
    all_features.set_yticks(range(len(comparisons)), [f"Feature {r['feature']}" for r in comparisons])
    all_features.set_ylim(len(comparisons) - 0.5, -0.55)
    all_features.set_xlim(0, comparisons[0]["mean_abs_margin_change"] * 1.2)
    all_features.set_xlabel("Mean absolute change in margin (logits)", labelpad=9)

    legend = [Line2D([], [], color=color, marker=marker, linestyle="none",
                     label=f"{mode.capitalize()} · {instruction}")
              for mode, (instruction, color, marker, _) in INSTRUCTIONS.items()]
    legend.append(Line2D([], [], color=INK, marker="o", markerfacecolor="none", markersize=10,
                         linestyle="none", label="Changed most likely next token"))
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.031, 0.095), frameon=False,
               ncol=3, handletextpad=0.45, columnspacing=2, fontsize=10)
    fig.text(0.035, 0.055, "Ringed predictions  ·  " + "     |     ".join(flips), fontsize=9.5, color=MUTED)
    fig.savefig(output, dpi=165, facecolor="white",
                metadata={"Source": f"{source.name}  SHA-256 {hashlib.sha256(raw).hexdigest()}"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEMO / "feature-replay.json")
    parser.add_argument("--out", type=Path, default=DEMO / "feature-effects.png")
    args = parser.parse_args()
    render(args.source, args.out)
