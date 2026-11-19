"""A public-data supervised control that changes a single tied Gemma embedding row."""
import argparse
import gc
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from alias_interpolation import COPY, EMBEDDING, TRANSFER, generate, matches, save
from iwi_harness.iwibench_adapter import grade_patch, public_task
from iwi_harness.targets import load_gemma_target
from iwi_harness.weights import WeightStore

SEEDS = {3: ("saffron", "cube"), 1: ("sprocket", "square")}
TRAIN_TRANSFER = (
    "How many corners are there on a {word}? Write only the numerical answer.",
    "Is a {word} a geometric shape? Respond with yes or no.",
)
TRAIN_COPY = (
    ("Type this phrase exactly, preserving spelling: blue {word}", "blue {word}"),
    ("Write only the final word, retaining its original letters: read the {word}", "{word}"),
    ("Transcribe the following text exactly: red {word} green", "red {word} green"),
)
POLICY = {
    "seeds": list(SEEDS), "initial_source_fraction": 0.5, "optimizer": "Adam", "learning_rate": 0.002,
    "steps": 12, "evaluate_steps": [4, 8, 12], "semantic_training": list(TRAIN_TRANSFER),
    "practice": "Use the first published correct answer to each official public practice prompt.",
    "training_copy": list(TRAIN_COPY), "public_evaluation_transfer": list(TRANSFER), "public_evaluation_copy": list(COPY),
    "selection": "Require at least as many alias and copy successes as full copy, and a strict improvement on one. "
                 "Among eligible candidates maximize alias successes, then copy successes, then use the earliest step.",
}


def supervised_data(target, pairs):
    """Extract answer-token labels; padding on the right preserves each natural prefix."""
    rows, positions, labels, weights = [], [], [], []
    end = target.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end not in target.eos_ids:
        raise ValueError("Unexpected Gemma turn terminator")
    for prompt, answer in pairs:
        prefix = target.encode(prompt)
        answer_ids = target.tokenizer.encode(answer.strip(), add_special_tokens=False) + [end]
        index = len(rows)
        rows.append(torch.tensor(prefix + answer_ids[:-1]))
        positions.extend((index, len(prefix) - 1 + j) for j in range(len(answer_ids)))
        labels.extend(answer_ids)
        weights.extend([1 / (len(pairs) * len(answer_ids))] * len(answer_ids))
    ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=target.tokenizer.pad_token_id)
    attention = torch.arange(ids.shape[1])[None, :] < torch.tensor([len(r) for r in rows])[:, None]
    return tuple(x.cuda() for x in (ids, attention.long(), torch.tensor(positions).T,
                                    torch.tensor(labels), torch.tensor(weights)))


def row_loss(target, alias_id, row, batch):
    # Both occurrences of the tied row must receive gradients: the input embedding and the output class.
    ids, attention, positions, labels, weights = batch
    language_model = target.model.model.language_model
    embed = language_model.embed_tokens
    dtype_row = row.to(embed.weight.dtype)
    embeds = torch.where((ids == alias_id).unsqueeze(-1), dtype_row * embed.embed_scale, embed(ids))
    hidden = language_model(inputs_embeds=embeds, attention_mask=attention, use_cache=False).last_hidden_state
    predicted = hidden[positions[0], positions[1]]
    logits = target.model.lm_head(predicted)
    logits[:, alias_id] = F.linear(predicted, dtype_row.unsqueeze(0)).squeeze(-1)
    return (F.cross_entropy(logits.float(), labels, reduction="none") * weights).sum()


def fit(target, seed, directory):
    alias, source = SEEDS[seed]
    task = public_task("alias_acquisition", seed)
    alias_id, = task["token_ids"][" " + alias]
    source_id, = task["token_ids"][" " + source]
    weight = target.weights()[EMBEDDING]
    original, referent = weight[alias_id].clone(), weight[source_id].clone()
    store = WeightStore(target)
    train = [(p["prompt"], p["answers"][0]) for p in task["practice"]]
    validation = [(p["prompt"], p["answers"]) for p in task["practice"]]
    for templates, is_training in ((TRAIN_TRANSFER, True), (TRANSFER, False)):
        answers = generate(target, store, [p.format(word=source) for p in templates])
        for question, answer in zip(templates, answers, strict=True):
            if is_training:
                train.append((question.format(word=alias), answer))
            else:
                validation.append((question.format(word=alias), [answer]))
    n_semantic = len(validation)
    train.extend((p.format(word=alias), a.format(word=alias)) for p, a in TRAIN_COPY)
    validation.extend((p.format(word=alias), [a.format(word=alias)]) for p, a in COPY)

    def evaluate(value):
        with torch.no_grad():
            weight[alias_id].copy_(value)
        try:
            outputs = generate(target, store, [p for p, _ in validation])
        finally:
            with torch.no_grad():
                weight[alias_id].copy_(original)
        ok = [matches(out, answers) for out, (_, answers) in zip(outputs, validation, strict=True)]
        return {"responses": outputs, "alias_correct": sum(ok[:n_semantic]), "copy_correct": sum(ok[n_semantic:])}

    full = evaluate(referent)
    row = torch.nn.Parameter((original.float() + referent.float()) / 2)
    optim = torch.optim.Adam([row], lr=POLICY["learning_rate"])
    batch = supervised_data(target, train)
    log, candidates = [], []
    best = None
    for i in range(1, POLICY["steps"] + 1):
        optim.zero_grad()
        loss = row_loss(target, alias_id, row, batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite supervised loss")
        loss.backward()
        optim.step()
        log.append(float(loss.detach()))
        if i in POLICY["evaluate_steps"]:
            scores = evaluate(row.detach())
            candidates.append(dict(step=i, **scores))
            a, c = scores["alias_correct"], scores["copy_correct"]
            eligible = a >= full["alias_correct"] and c >= full["copy_correct"] and (
                a > full["alias_correct"] or c > full["copy_correct"])
            if eligible and (best is None or (a, c) > best[:2]):
                best = (a, c, i, row.detach().clone())
    result = dict(seed=seed, task=task["id"], model_id=task["model_id"], revision=task["checkpoint_revision"],
                  training_pairs=train, public_checks=validation, full_copy=full, loss=log,
                  candidates=candidates, selected_step=None, submitted_patches={})
    if best is not None:
        result["selected_step"] = best[2]
        for name, value in (("full", referent), ("trained", best[3])):
            store = WeightStore(target)
            filename = f"seed-{seed}-{name}.safetensors"
            try:
                store.write(EMBEDDING, "replace", selection=[alias_id],
                            value={"handle": store.capture(value.to(weight.dtype))})
                receipt = store.export(directory / filename)
                result["submitted_patches"][name] = {"file": filename, "sha256": receipt["sha256"]}
            finally:
                with torch.no_grad():
                    weight[alias_id].copy_(original)
    return result


def run(out, historical):
    import json
    start = time.monotonic()
    out.mkdir(parents=True)
    save(out / "policy.json", POLICY)
    target = load_gemma_target(local_files_only=True)
    episodes = [fit(target, seed, out) for seed in SEEDS]
    save(out / "frozen.json", episodes)
    print([(e["seed"], e["selected_step"]) for e in episodes], flush=True)
    del target
    gc.collect()
    torch.cuda.empty_cache()
    for e in episodes:
        e["grades"] = {}
        for name, artifact in e["submitted_patches"].items():
            if name == "full" and e["seed"] == 3:
                grade = json.loads(historical.read_text())
                if grade["patch_sha256"] != artifact["sha256"] or grade["episode_id"] != e["task"]:
                    raise ValueError("Preregistered full-copy reference does not match")
            else:
                grade = grade_patch(out / artifact["file"], family="alias_acquisition", seed=e["seed"],
                                    expected_sha256=artifact["sha256"])
            e["grades"][name] = grade
            print(e["seed"], name, grade["status"], grade["reward"], flush=True)
        save(out / f"seed-{e['seed']}.json", e)
    save(out / "result.json", dict(policy=POLICY, episodes=episodes, elapsed_seconds=time.monotonic()-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--known-full", type=Path, default=Path("evidence/iwibench-alias.json"))
    args = parser.parse_args()
    run(args.out, args.known_full)
