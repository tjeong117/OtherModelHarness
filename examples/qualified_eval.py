"""Load matched public tasks and grade submitted patches on the pinned qualified Hub split."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from grade_sae_transfer import private_items, save
from iwi_harness.iwibench_adapter import _alias_tokens, _cached_tokenizer, _completions, _strings
from iwi_harness.targets import GEMMA_ID, GEMMA_REVISION

REVISION = "300a9d7271d9e3a9dd5bfc14edcbceccb66d5966"
MANIFEST_SHA = "f8b84e0d0b641d4781ece168ee8a7c43fac767981a66c57dc9c190a5626f4097"


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_install(catalog):
    import iwibench

    catalog = Path(catalog).resolve()
    if Path(iwibench.__file__).resolve().parent != (catalog / "src/iwibench").resolve():
        raise ValueError("prepend the qualified catalogue's src directory to PYTHONPATH")
    manifest_path = catalog / "manifest.json"
    if hash_file(manifest_path) != MANIFEST_SHA:
        raise ValueError("wrong public Hub revision manifest")
    files = json.loads(manifest_path.read_text())["files"]
    for path in [*catalog.glob("src/iwibench/*.py"), *catalog.glob("data/*.json"), catalog / "data/test.jsonl"]:
        if hash_file(path) != files[str(path.relative_to(catalog))]:
            raise ValueError(f"manifest mismatch for {path.name}")
    return catalog


def qualified_rows(catalog, family):
    return [
        row
        for line in (catalog / "data/test.jsonl").read_text().splitlines()
        if (row := json.loads(line))["family"] == family
    ]


def regenerate_row(row):
    from iwibench.qualified_catalog import qualified_episode
    from iwibench.variants import fingerprint

    ep = qualified_episode(row["family"], row["variant_index"], GEMMA_ID)
    if (
        ep["id"] != row["id"]
        or fingerprint(ep) != row["episode_sha256"]
        or fingerprint(ep["parameters"]) != row["parameter_sha256"]
        or row["checkpoint_revision"] != GEMMA_REVISION
        or row["calibration_status"] != "eligible"
    ):
        raise ValueError("qualified regeneration or native checkpoint mismatch")
    return ep


def qualified_task(catalog, family, variant_index):
    rows = [r for r in qualified_rows(verify_install(catalog), family) if r["variant_index"] == variant_index]
    if len(rows) != 1:
        raise ValueError("expected exactly one matching task from the qualified test partition")
    return rows[0], regenerate_row(rows[0])


def public_qualified_task(row, ep, target):
    if (row["id"], row["family"]) != (ep["id"], ep["family"]):
        raise ValueError("public row and qualified episode differ")
    tokenizer = target.tokenizer
    tokens = {}
    for word in _strings(ep["parameters"]):
        if word not in tokens:
            for text in (word, " " + word):
                tokens[text] = tokenizer.encode(text, add_special_tokens=False)
    aliases = _alias_tokens(ep, tokenizer) if ep["family"] == "alias_acquisition" else {}
    tokens["alias_context"] = {
        key: {f: rec.get(f, []) for f in ("token_ids", "standalone_token_ids", "leading_space_token_ids")}
        for key, rec in aliases.items()
    }
    return {
        "objective": row["objective"],
        "practice": json.loads(row["public_practice_json"]),
        "token_ids": tokens,
    }


def no_remote_verdict(key):
    raise ValueError("A semantic predicate required remote judgement")


def assert_deterministic(ep):
    from iwibench.terra_semantics import RULES

    if any(item["rule"] in RULES for item in private_items(ep)):
        raise ValueError("The selected qualified task requires its pinned semantic judge")


def grade_qualified_patch(catalog, family, variant_index, patch, sha, *, work=None, device="cuda"):
    from iwibench.config import Limits
    from iwibench.scoring import CalibrationError, content_checker_scope, score_episode
    from iwibench.terra_semantics import JudgedChecker
    from iwibench.variants import fingerprint

    row, ep = qualified_task(catalog, family, variant_index)
    assert_deterministic(ep)
    patch = Path(patch)
    if hash_file(patch) != sha:
        raise ValueError("submitted patch differs from its receipt")
    items = private_items(ep)
    request = dict(
        model_id=GEMMA_ID,
        revision=GEMMA_REVISION,
        device=device,
        prompts=[i["prompt"] for i in items],
        seeds=[ep["test_seed"] + i for i in range(len(items))],
        max_new_tokens=Limits().output_tokens,
        patch_path=None,
        patch_sha256=None,
    )
    tokenization = _alias_tokens(ep, _cached_tokenizer()) if family == "alias_acquisition" else {}
    with TemporaryDirectory() as temp:
        work = Path(work or temp)
        work.mkdir(mode=0o700, parents=True, exist_ok=True)

        def fresh(patch_path, patch_sha):
            cache = work / f"{row['episode_sha256']}-{patch_sha or 'pristine'}.json"
            if cache.exists():
                stored = json.loads(cache.read_text())
                outputs = stored["outputs"]
                if (
                    stored["episode_sha256"] != row["episode_sha256"]
                    or stored["patch"] != patch_sha
                    or stored["output_sha256"] != fingerprint(outputs)
                    or set(outputs) != {item["id"] for item in items}
                ):
                    raise ValueError("invalid private generation cache")
                return outputs
            outputs = _completions(
                items,
                dict(
                    request,
                    patch_path=str(patch_path.resolve()) if patch_path else None,
                    patch_sha256=patch_sha,
                ),
            )
            save(
                cache,
                dict(
                    episode_sha256=row["episode_sha256"],
                    patch=patch_sha,
                    output_sha256=fingerprint(outputs),
                    outputs=outputs,
                ),
            )
            return outputs

        base = fresh(None, None)
        info = dict(
            episode_id=ep["id"],
            family=family,
            model_id=GEMMA_ID,
            revision=GEMMA_REVISION,
            dataset_revision=REVISION,
            episode_sha256=row["episode_sha256"],
            patch_sha256=sha,
            ranked_split="test",
        )
        with content_checker_scope(JudgedChecker(no_remote_verdict)):
            try:
                score_episode(ep, base, base, tokenization=tokenization)
            except CalibrationError as exc:
                return {**info, "status": "ineligible", "reward": None, "metrics": None, "reason": str(exc)}
            edited = fresh(patch, sha)
            metrics = score_episode(ep, base, edited, tokenization=tokenization)
    return {**info, "status": "scored", "metrics": metrics, "reward": metrics["reward"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant-index", type=int, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    save(
        args.out,
        grade_qualified_patch(
            args.catalog, args.family, args.variant_index, args.patch, args.sha256, work=args.work
        ),
    )
