"""Freeze an interleaved, matched local-editor factorial on qualified public rows."""
import argparse
import hashlib
import json
from pathlib import Path

SALT = "iwi-paired-preview-2026-v1"
FAMILIES = ("alias_acquisition", "output_code_remapping", "factual_update", "selective_factual_withholding")
GUIDANCE = (
    "Edit the target weights to accomplish the public objective. You have six editor turns including submission. "
    "Use native tool calls. Use test_behavior early with the supplied improvement and preservation cases. "
    "If the tool offers an internal intervention, test a native neuron with intervention {"
    "\"layer\":10,\"scale\":0.0}; use the returned effects and exact write if they support your objective. "
    "If preview_weights is offered, compare a candidate weight change on the same public cases, then explicitly "
    "commit_preview if useful, or discard_preview and try a different write. Direct write_weights is available. "
    "A useful test or preview alone does not persist a weight change. Finish with submit_patch({}). "
    "Batch calls where appropriate. The qualified grader examines a fresh model with only your submitted patch."
)

def rank(label):
    return hashlib.sha256((SALT + "|" + label).encode()).hexdigest()


def design(catalog, previous):
    raw = (catalog / "data/test.jsonl").read_bytes()
    rows = [json.loads(s) for s in raw.splitlines()]
    excluded = sorted(v["id"] for v in json.loads(previous.read_text())["tasks"])
    tasks = []
    for family in FAMILIES:
        candidates = [r for r in rows if r['family'] == family and r['id'] not in excluded]
        for row in sorted(candidates, key=lambda r: rank(r["id"]))[:2]:
            tasks.append({k: row[k] for k in ("id", "family", "variant_index", "episode_sha256", "parameter_sha256")})
    blocks = sorted(((t, r) for t in tasks for r in (1, 2)), key=lambda v:rank(f"block|{v[0]['id']}|{v[1]}"))
    attempts = []
    for i, (task, replicate) in enumerate(blocks):
        # Latin-square order within blocks, balanced over 16 task × replicate pairs.
        for j in range(4):
            arm = "ABCD"[(i + j) % 4]
            seed = int(rank(f"decode|{task['id']}|{replicate}")[:8], 16)
            slug = f"{task['family']}-{task['variant_index']:03d}-r{replicate}-{arm}"
            attempts.append(dict(order=len(attempts)+1, slug=slug, task_id=task["id"],
                                 family=task["family"], variant_index=task["variant_index"],
                                 replicate=replicate, seed=seed, arm=arm))
    return dict(protocol=SALT, dataset_revision="300a9d7271d9e3a9dd5bfc14edcbceccb66d5966",
                test_file_sha256=hashlib.sha256(raw).hexdigest(), excluded_previous_live_ids=excluded,
                selection="Lowest two SHA256(protocol|public ID) per family after excluding previous live pilot IDs.",
                editor="Qwen/Qwen2.5-7B-Instruct", editor_revision="a09a35458c702b33eeacc393d103063234e8bc28",
                target="google/gemma-3-4b-it", target_revision="093f9f388b31de276ce2de164bdc2081324b9767",
                editor_decoding=dict(max_turns=6, max_new_tokens=384, temperature=0.4, top_p=0.9, top_k=0),
                guidance=GUIDANCE, opening_manifest="grouped", target_limits=dict(max_forwards=512,max_forward_tokens=131072),
                public_control="Copy the following word exactly. Reply with the word only: pencil",
                arms=dict(A="Behavioral tools and direct writes",B="A plus native intervention inside test_behavior",
                          C="A plus reversible weight preview",D="B plus reversible weight preview"),
                primary="Paired D minus C all-attempt native qualified reward, averaged over replicates then eight tasks.",
                secondary=["B-A, C-A, D-B, D-A and interaction (D-C)-(B-A)", "Explicit submissions and nonempty writes",
                           "Improvement and retention from qualified metrics", "Actual successful tool uptake; editor token and target compute costs"],
                outcomes="No explicit submission scores zero. Infrastructure faults retry same assignment; disclose unresolved faults. "
                         "Ineligible official calibrations are shown separately; paired contrasts use complete eligible tasks.",
                design_unit="Task, not editor call; 8 tasks in 4 families. Two stochastic replicates share their seed across arms.",
                tasks=tasks, attempts=attempts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("evidence/heldout-token-audit/live-design.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = design(args.catalog, args.previous)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
