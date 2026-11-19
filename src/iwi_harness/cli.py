"""Command-line hosts for editing sessions, demonstrations, and independent verification."""
import argparse
import json
import os
from pathlib import Path

from .public_cases import read_public_cases


def _counter(encoding):
    if encoding:
        import tiktoken
        codec = tiktoken.get_encoding(encoding)
        return lambda text: len(codec.encode(text, disallowed_special=()))
    return None


def main(argv=None):
    p = argparse.ArgumentParser(prog="iwi-harness")
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a scripted CPU edit and verify in a fresh process")
    demo.add_argument("--out", required=True, type=Path)
    demo.add_argument("--literal", action="store_true")
    demo.add_argument("--token-encoding")
    verify = sub.add_parser("verify-tiny", help="score a numerical patch on a fresh offline target")
    verify.add_argument("--patch", required=True, type=Path)
    verify.add_argument("--sha256")
    gpu = sub.add_parser("gpu-smoke", help="run Gemma 3 4B with the official Gemma Scope 2 dictionary")
    gpu.add_argument("--out", required=True, type=Path)
    gpu.add_argument("--device", default="cuda")
    gpu.add_argument("--local-files-only", action="store_true")
    costs = sub.add_parser("compare-costs", help="measure equivalent literal and reference tool payloads")
    costs.add_argument("--encoding", default="o200k_base")
    task = sub.add_parser("harbor-task", help="materialize the offline Harbor task")
    task.add_argument("--out", required=True, type=Path)
    edit = sub.add_parser("edit", help="run an external tool-calling updater")
    edit.add_argument("--target", default="tiny", help="tiny, gemma, or a supported Hugging Face causal LM repository ID")
    edit.add_argument("--target-revision", help="exact checkpoint commit for a Hugging Face target")
    edit.add_argument("--target-blocks", help="optional dotted module path to the target's ordered decoder blocks")
    edit.add_argument("--preload-sae", action="store_true",
                      help="start with Gemma Scope or the tiny fixture dictionary; otherwise start with none")
    edit.add_argument("--device", default="cuda")
    edit.add_argument("--local-files-only", action="store_true")
    edit.add_argument("--out", required=True, type=Path)
    edit.add_argument("--objective", required=True)
    provider = edit.add_mutually_exclusive_group(required=True)
    provider.add_argument("--base-url", help="Chat Completions API base, including /v1 if needed")
    provider.add_argument("--astra", action="store_true", help="use the OpenAI Responses API")
    edit.add_argument("--updater-model", help="model identifier (Astra default: gpt-6.2-astra)")
    edit.add_argument("--key-env", help="bearer token variable (Astra: OPENAI_API_KEY; other: UPDATER_API_KEY)")
    edit.add_argument("--budget-usd", type=float, default=10, help="Astra per-run spending ceiling")
    edit.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    edit.add_argument("--max-output-tokens", type=int, default=4096, help="Astra per-turn output cap including reasoning")
    edit.add_argument("--max-turns", type=int, default=32)
    edit.add_argument("--detailed-writes", action="store_true", help="include per-operation write constraints in the model prompt")
    edit.add_argument("--guide", action="store_true", help="preview parameter names and explain slice syntax")
    edit.add_argument("--opening-manifest", choices=["paged", "eager", "grouped"],
                      help="show the first four writable name/shape pairs, the complete list, or complete layer ranges")
    edit.add_argument("--sae-guided", action="store_true", help="prompt the updater to fit and causally test an SAE feature before editing")
    edit.add_argument("--token-encoding")
    edit.add_argument("--no-interpretability", action="store_true", help="expose compact weight and generation tools only")
    edit.add_argument("--behavior-checks", action="store_true", help="test public improvement and preservation cases")
    edit.add_argument("--public-cases", metavar="FILE", type=read_public_cases,
                      help="JSON array of public examples used when a behavior check or weight preview omits cases")
    edit.add_argument("--internal-measurements", action="store_true", help="allow temporary interventions in public behavior tests")
    edit.add_argument("--edit-preview", action="store_true", help="preview, commit or discard an exact candidate weight edit")
    edit.add_argument("--experiment-tools-only", action="store_true", help="expose public behavior tests and omit standalone probes")
    for name in ("iwibench-task", "iwibench-grade"):
        parser = sub.add_parser(name, help="use the pinned optional IWIBench adapter")
        parser.add_argument("--family", required=True)
        parser.add_argument("--seed", required=True, type=int)
        if name == "iwibench-grade":
            parser.add_argument("--patch", type=Path, required=True)
            parser.add_argument("--sha256")
            parser.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    if a.command == "demo":
        from .demo import run_demo
        out = run_demo(a.out, literal=a.literal, token_counter=_counter(a.token_encoding))
    elif a.command == "verify-tiny":
        from .demo import verify_tiny
        out = verify_tiny(a.patch, a.sha256)
    elif a.command == "gpu-smoke":
        from .gpu_check import gpu_smoke
        out = gpu_smoke(a.out, local_files_only=a.local_files_only, device=a.device)
    elif a.command == "compare-costs":
        from .costs import compare_payloads
        out = compare_payloads(encoding=a.encoding)
    elif a.command == "harbor-task":
        from .harbor_adapter import make_task
        make_task(a.out)
        out = {"task": str(a.out)}
    elif a.command == "iwibench-task":
        from .iwibench_adapter import public_task
        out = public_task(a.family, a.seed)
    elif a.command == "iwibench-grade":
        from .iwibench_adapter import grade_patch
        out = grade_patch(a.patch, family=a.family, seed=a.seed, device=a.device, expected_sha256=a.sha256)
    else:
        from .session import EditingSession
        from .targets import load_gemma_sae, load_gemma_target, load_hf_target, tiny_sae, tiny_target
        from .updater import chat_completions, run_updater, sae_guided_objective
        if a.sae_guided and (a.no_interpretability or a.experiment_tools_only):
            p.error("--sae-guided requires interpretability tools")
        if a.internal_measurements and a.no_interpretability:
            p.error("--internal-measurements requires interpretability tools")
        preset = a.target in ("tiny", "gemma")
        if preset and (a.target_revision or a.target_blocks):
            p.error("tiny and gemma select pinned presets; use a repository ID with --target-revision or --target-blocks")
        if not preset and not a.target_revision:
            p.error("provide --target-revision for the selected Hugging Face repository")
        if a.preload_sae and (not preset or a.no_interpretability):
            p.error("--preload-sae requires an enabled interpretability interface and the tiny or gemma preset")
        if a.astra:
            from .openai_responses import AstraBudget, AstraResponses
            complete = AstraResponses(model=a.updater_model or "gpt-6.2-astra",
                                      api_key=os.getenv(a.key_env or "OPENAI_API_KEY"),
                                      budget=AstraBudget(a.budget_usd), reasoning=a.reasoning_effort,
                                      max_output_tokens=a.max_output_tokens)
        else:
            if not a.updater_model:
                p.error("--base-url requires --updater-model")
            complete = chat_completions(a.base_url, a.updater_model, os.getenv(a.key_env or "UPDATER_API_KEY"))
        if a.target == "gemma":
            kw = dict(device=a.device, local_files_only=a.local_files_only)
            target = load_gemma_target(**kw)
            sae = load_gemma_sae(**kw) if a.preload_sae else None
        elif a.target == "tiny":
            target = tiny_target()
            sae = tiny_sae() if a.preload_sae else None
        else:
            target = load_hf_target(a.target, revision=a.target_revision, device=a.device,
                                    local_files_only=a.local_files_only, block_path=a.target_blocks)
            sae = None
        session = EditingSession(target, artifact_dir=a.out, sae=sae, token_counter=_counter(a.token_encoding),
                                 interpretability=not a.no_interpretability, detailed_writes=a.detailed_writes,
                                 behavior_checks=a.behavior_checks, internal_measurements=a.internal_measurements,
                                 edit_preview=a.edit_preview, experiment_tools_only=a.experiment_tools_only,
                                 public_cases=a.public_cases)
        try:
            objective = sae_guided_objective(a.objective, a.max_turns) if a.sae_guided else a.objective
            out = run_updater(session, objective, complete, max_turns=a.max_turns, guide=a.guide,
                              opening_manifest=a.opening_manifest)
        finally:
            if a.astra:
                usage = {"requested_model": complete.model, "responses": complete.records,
                         "cost_upper_bound_usd": complete.budget.spent_upper_bound_usd}
                (session.artifact_dir / "provider.json").write_text(json.dumps(usage, indent=2) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False))
