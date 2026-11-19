"""Read a host-selected JSON file of public behavior examples for command-line sessions."""
import argparse
import json
from pathlib import Path


def read_public_cases(path):
    try:
        cases = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(f"cannot read public cases from {path}: {exc}") from exc
    if not isinstance(cases, list) or not cases:
        raise argparse.ArgumentTypeError("public cases must be a nonempty JSON array of prompt, answers and slice objects")
    return cases
