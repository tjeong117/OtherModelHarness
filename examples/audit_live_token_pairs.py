"""Reconcile actual editor usage for the six previously collected development triplets."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from compare_astra_arms import CASES, pair


DIRECTORIES = {"compact": "astra-compact-control", "full": "astra-full-tools", "native": "astra-native-control"}


def totals(turns):
    counts = Counter()
    for usage in turns:
        inputs = usage["prompt_tokens"]
        outputs = usage["completion_tokens"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        if not 0 <= cached <= inputs or inputs + outputs != usage["total_tokens"]:
            raise ValueError("inconsistent per-response token counts")
        counts.update(requests=1, input=inputs, output=outputs, total=inputs + outputs,
                      cached_input=cached, uncached_input=inputs - cached)
    return dict(counts)


def audit(root):
    episodes = []
    sums = {arm: {} for arm in DIRECTORIES}
    for alias, (family, seeds) in CASES.items():
        for seed in seeds:
            paths = {arm: root / directory / f"{alias}-{seed}" for arm, directory in DIRECTORIES.items()}
            checks = {arm: pair(paths["full"], paths[arm], family, seed, arm) for arm in ("compact", "native")}
            if not all(check["both_freshly_scored"] for check in checks.values()):
                raise ValueError(f"matching or native scoring failed for {family}/{seed}")
            grades = {"full": checks["compact"]["full"], **{arm: check["control"] for arm, check in checks.items()}}
            traces = {}
            for arm, path in paths.items():
                raw = (path / "trajectory.jsonl").read_bytes()
                turns = [event["usage"] for line in raw.splitlines()
                         if (event := json.loads(line)).get("type") == "updater"]
                records = json.loads((path / "editor-result.json").read_text())["provider"]
                if [record["usage"] for record in records] != turns:
                    raise ValueError(f"{arm}/{seed}: provider summary differs from original trace")
                traces[arm] = (hashlib.sha256(raw).hexdigest(), turns)
            common = min(len(turns) for _, turns in traces.values())
            if common < 5:
                raise ValueError(f"fewer than five requests in {family}/{seed}")
            result = {"family": family, "seed": seed, "common_requests": common, "arms": {}}
            for arm, (digest, turns) in traces.items():
                stats = {name: totals(prefix) for name, prefix in (
                    ("opening", turns[:1]), ("first_five", turns[:5]),
                    ("common_prefix", turns[:common]), ("completed", turns))}
                for name, counts in stats.items():
                    sums[arm].setdefault(name, Counter()).update(counts)
                grade = grades[arm]
                result["arms"][arm] = {
                    **stats, "reward": grade["reward"], "quality_counts": grade["counts"],
                    "trace_sha256": digest, "provider_model_echo": grade["provider"]["response_models"],
                    "turns": [{"input": u["prompt_tokens"], "cached_input": u["prompt_tokens_details"]["cached_tokens"],
                               "output": u["completion_tokens"]} for u in turns],
                }
            episodes.append(result)
    comparisons = {}
    for arm in ("compact", "full"):
        versus = {}
        for scope in ("opening", "first_five", "common_prefix", "completed"):
            metric = "input" if scope == "opening" else "total"
            reductions = [1 - ep["arms"][arm][scope][metric] / ep["arms"]["native"][scope][metric] for ep in episodes]
            versus[scope] = {"metric": metric, "fraction_lower_pooled": 1 - sums[arm][scope][metric] / sums["native"][scope][metric],
                             "fraction_lower_each_episode": reductions, "episodes_lower": sum(x > 0 for x in reductions)}
        comparisons[arm + "_vs_native"] = versus
    for arm in DIRECTORIES:
        outcomes = [ep["arms"][arm] for ep in episodes]
        sums[arm]["mean_episode_reward"] = mean(row["reward"] for row in outcomes)
        sums[arm]["pooled_quality_counts"] = dict(sum((Counter(row["quality_counts"]) for row in outcomes), Counter()))
    return {"design": "Six existing development tasks, one trajectory per arm and task. "
            "All arms requested the same model, reasoning, editor turn/output and dollar caps. "
            "The common prefix uses the minimum realized turn count across all three arms on each task. "
            "Prefixes are measurements of traffic; reward is measured only for completed patches. "
            "Input includes repeated cached context. Each arm used its original prompts, tools, feedback and stopping policy.",
            "episodes": episodes, "pooled": sums, "comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("evidence/heldout-token-audit/live-development.json"))
    options = parser.parse_args()
    report = audit(options.runs)
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
