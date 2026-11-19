"""Frozen binding enrollment, query construction and gold scoring for the lexical memory pilot."""
from __future__ import annotations

import json
from pathlib import Path
import re


DESIGN = Path(__file__).resolve().parents[1] / "evidence/weight-memory/design.json"


def contextual_row(tokenizer, word):
    """Resolve the space-prefixed row in a neutral sentence, without target inference."""
    prefix = "I remember the word"
    text = f"{prefix} {word}."
    start = len(prefix) + 1
    end = start + len(word)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [i for i, (left, right) in zip(encoded["input_ids"], encoded["offset_mapping"], strict=True)
           if left < end and right > start]
    if ids != tokenizer.encode(" " + word, add_special_tokens=False) or len(ids) != 1:
        raise ValueError(f"expected one contextual token for {word!r}, got {ids}")
    return ids[0]


def load(path):
    return json.loads(Path(path).read_text())


def cases(design, sequence, stage):
    active = sequence["associations"][:stage]
    items = []
    for index, binding in enumerate(active, 1):
        common = dict(alias=binding["alias"], introduced=index)
        for form in design["questions"]:
            items.append(dict(**common, id=f'{index}-{form["id"]}', group="target", form=form["id"],
                              prompt=form["template"].format(word=binding["alias"]), answers=binding[form["gold"]]))
        for kind, word, answers in (("copy", binding["alias"], [binding["alias"]]),
                                    ("canonical", binding["country"], binding["capital_answers"])):
            items.append(dict(**common, id=f"{index}-{kind}", group=kind, form=kind,
                              prompt=design["controls"][kind].format(word=word), answers=answers))
    for i, task in enumerate(design["controls"]["unrelated"], 1):
        items.append(dict(id=f"general-{i}", group="unrelated", form=f"general-{i}",
                          alias=None, introduced=None, **task))
    return items


def fetch_note(query, learned, template):
    """Look up only acquired alias keys; retrieval never sees question labels or answers."""
    keys = set(re.findall(r"\b\w+\b", query.casefold()))
    matches = [binding for binding in learned if binding["alias"].casefold() in keys]
    if len(matches) > 1:
        raise ValueError("expected a single matching alias")
    return (template.format(**matches[0]), matches[0]["alias"]) if matches else ("", None)


def correct(text, answers):
    def norm(s):
        return s.strip().removesuffix(".").casefold()
    return any(norm(text) == norm(answer) for answer in answers)
