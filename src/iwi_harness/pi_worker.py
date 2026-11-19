"""Host the Pi harness's target editing session over JSON Lines."""
import argparse
from contextlib import redirect_stdout
import json
import sys

from iwi_harness.session import EditingSession, encode_json
from iwi_harness.public_cases import read_public_cases
from iwi_harness.targets import load_gemma_sae, load_gemma_target, load_hf_target, tiny_sae, tiny_target


def main(wire):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--target", default="tiny")
    p.add_argument("--target-revision")
    p.add_argument("--target-blocks")
    p.add_argument("--device", default="cpu")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--preload-sae", action="store_true")
    p.add_argument("--arm", choices=["A", "B", "C", "D", "full"], default="full")
    p.add_argument("--public-cases", metavar="FILE", type=read_public_cases,
                   help="public examples used by test_behavior and preview_weights when cases are omitted")
    p.add_argument("--max-tool-calls", type=int, default=128)
    p.add_argument("--max-forwards", type=int, default=256)
    p.add_argument("--max-forward-tokens", type=int, default=262144)
    args = p.parse_args()
    if args.target in ("tiny", "gemma") and (args.target_revision or args.target_blocks):
        p.error("tiny and gemma are pinned presets; provide a repository ID for custom checkpoint options")
    if args.preload_sae and (args.arm in ("A", "C") or args.target not in ("tiny", "gemma")):
        p.error("--preload-sae requires an internal interface and a registered tiny or gemma dictionary")
    sae = None
    if args.target == "tiny":
        target = tiny_target()
        sae = tiny_sae() if args.preload_sae else None
    elif args.target == "gemma":
        options = dict(device=args.device, local_files_only=args.local_files_only)
        target = load_gemma_target(**options)
        sae = load_gemma_sae(**options) if args.preload_sae else None
    else:
        if not args.target_revision:
            p.error("--target-revision is required for a generic HF repository")
        target = load_hf_target(args.target, revision=args.target_revision, block_path=args.target_blocks,
                                device=args.device, local_files_only=args.local_files_only)
    internal = args.arm in ("B", "D", "full")
    session = EditingSession(target, artifact_dir=args.out, sae=sae, detailed_writes=True, behavior_checks=True,
                             interpretability=internal, internal_measurements=internal,
                             edit_preview=args.arm in ("C", "D", "full"), experiment_tools_only=args.arm != "full",
                             max_tool_calls=args.max_tool_calls, max_forwards=args.max_forwards,
                             max_forward_tokens=args.max_forward_tokens, public_cases=args.public_cases)
    print(encode_json({"tools": session.tools, "info": session.info}), file=wire, flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            result = {"result": session.call(req["name"], req["arguments"])}
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            result = {"error": str(exc)[:1024]}
        print(encode_json(result), file=wire, flush=True)


def serve():
    wire = sys.stdout
    with redirect_stdout(sys.stderr):
        main(wire)


if __name__ == "__main__":
    serve()
