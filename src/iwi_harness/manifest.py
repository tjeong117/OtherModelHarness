"""Reversible name/shape listing for repeated indexed transformer layers."""
from itertools import groupby
import re


LAYER = re.compile(r"^(.*\.layers\.)(\d+)\.(.+)$")
GUIDANCE = ("Literal pairs give [name,shape]. For a layer block, iterate layers_inclusive from first to last, "
            "then suffixes in listed order. Each name is prefix + decimal layer + '.' + suffix; use its listed shape.")


def layer_id(entry):
    match = LAYER.fullmatch(entry[0])
    if match and match[2] == str(int(match[2])):
        return match[1], int(match[2])
    return None


def group_entries(entries):
    grouped = []
    for key, part in groupby(entries, layer_id):
        if key is None:
            grouped.extend(part)
            continue
        prefix, layer = key
        suffixes = [[name.removeprefix(f"{prefix}{layer}."), shape] for name, shape in part]
        previous = grouped[-1] if grouped else None
        if (isinstance(previous, dict) and previous["prefix"] == prefix
                and previous["layers_inclusive"][1] + 1 == layer and previous["suffixes"] == suffixes):
            previous["layers_inclusive"][1] = layer
        else:
            grouped.append({"prefix": prefix, "layers_inclusive": [layer, layer], "suffixes": suffixes})
    return grouped


def expand(grouped):
    entries = []
    for item in grouped:
        if isinstance(item, list):
            entries.append(item)
        else:
            first, last = item["layers_inclusive"]
            for layer in range(first, last + 1):
                for suffix, shape in item["suffixes"]:
                    entries.append([f"{item['prefix']}{layer}.{suffix}", shape])
    return entries
