"""A synchronous, provider-neutral function-calling loop and an optional HTTP transport."""
import hashlib
import json
from urllib import request

from .manifest import GUIDANCE, expand, group_entries
from .session import encode_json


class NoSubmission(ValueError):
    """The updater finished or spent its turn budget without returning a patch."""


def sae_guided_objective(objective, max_turns):
    """Attach the public SAE-guided editing policy to a task objective."""
    return (
        f"{objective}\n\nUse an SAE to guide your weight edit. You have {max_turns} turns including submission. "
        "Batch independent calls and keep outputs short. Fit a small SAE on balanced, paired public examples at "
        "the decision position (usually positions [-1]). Hold out distinct prompts and check reconstruction. "
        "Find an active feature that distinguishes the behaviors. Use inspect_target with an SAE probe and "
        "intervention scale 0 to suppress it on a fresh case to improve and a case to preserve. Request the "
        "two competing answer token IDs in a consistent order and compare their margin shifts. Use that "
        "evidence to choose the target and size of a concrete weight edit. If the feature is uninformative, "
        "use the public errors to choose the edit. Consider a context-specific input edit when globally "
        "biasing an answer could break other cases. Check complete generated answers on improvement and "
        "preservation examples. Submit your best patch within the turn limit. If a final correction is needed, "
        "call write_weights followed by submit_patch in the same turn."
    )


def chat_completions(base_url, model, api_key=None, *, timeout=180):
    """Build a callback returning a Chat Completions response from a compatible server."""
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def complete(messages, tools):
        payload = dict(model=model, messages=messages, tools=tools)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = request.Request(endpoint, data=encode_json(payload).encode(), headers=headers)
        with request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    return complete


def _matched_manifest(session, mode):
    """Read identical public metadata pages in both conditions before assigning visibility."""
    entries = []
    offset = 0
    pages = 0
    while offset is not None:
        page = session.call("read_weights", {"offset": offset, "limit": 256})
        entries.extend([item["name"], item["shape"]] for item in page["tensors"])
        pages += 1
        offset = page["next_offset"]
    visible = entries[:4] if mode == "paged" else entries
    grouped_record = {}
    if mode == "grouped":
        groups = group_entries(visible)
        expanded = expand(groups)
        if expanded != entries:
            raise ValueError("grouped manifest differs from the public writable listing")
        rendered = encode_json({"tensors": groups, "tensor_encoding": GUIDANCE})
        grouped_record = {"rendered_records": len(groups), "encoding_guidance": GUIDANCE,
                          "expanded_entries_sha256": hashlib.sha256(encode_json(expanded).encode()).hexdigest()}
    else:
        rendered = encode_json(visible)
    record = {"mode": mode, "available_entries": len(entries), "shown_entries": len(visible), **grouped_record,
              "metadata_tool_calls": pages, "rendered_list_bytes": len(rendered.encode()),
              "ordered_entries_sha256": hashlib.sha256(encode_json(entries).encode()).hexdigest(),
              "rendered_list_sha256": hashlib.sha256(rendered.encode()).hexdigest()}
    return rendered, record


def run_updater(session, objective, complete, *, max_turns=32, guide=False, opening_manifest=None):
    """A terminal reward is evaluated by the host on the submitted artifact."""
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    if opening_manifest not in (None, "paged", "eager", "grouped"):
        raise ValueError("opening_manifest must be paged, eager or grouped")
    guide = guide or opening_manifest is not None
    messages = [
        {"role": "system", "content":
         "Inspect and edit the target with the supplied tools. Use submit_patch to finish. "
         "Handles are valid only until a successful write. Session settings: " + encode_json(session.info)},
        {"role": "user", "content": objective},
    ]
    if guide:
        if opening_manifest is None:
            # Keep the original guide and its single metered preview for historical experiments.
            manifest = session.call("read_weights", {"limit": 4})
            rendered = encode_json([[item["name"], item["shape"]] for item in manifest["tensors"]])
            title, lookup = "First writable names and shapes", "see later names"
        else:
            rendered, record = _matched_manifest(session, opening_manifest)
            title, lookup = "Writable names and shapes", "look up names"
        messages[0]["content"] += (
            f" You have {max_turns} model turns including submission. {title}: {rendered}. "
            f"Use read_weights with offset to {lookup}. "
            "A selection contains one index or range per tensor axis; trailing axes are retained. "
            "A numeric value is literal data. For example, copying row 2 of matrix A to row 7 of matrix B "
            'takes {"tensor":"B","selection":[7],"operation":"copy","source":{"tensor":"A","selection":[2]}}. '
            "When the public task provides token IDs, distinguish in-context occurrences from standalone spellings."
        )
        if opening_manifest is not None:
            record["opening_system_sha256"] = hashlib.sha256(messages[0]["content"].encode()).hexdigest()
            record["objective_sha256"] = hashlib.sha256(objective.encode()).hexdigest()
            record["tool_schema_sha256"] = hashlib.sha256(encode_json(session.tools).encode()).hexdigest()
            (session.artifact_dir / "opening-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    for index in range(max_turns):
        if guide and index > 0 and index + 1 == max_turns:
            messages.append({"role": "user", "content":
                             "This is your final turn. Use submit_patch with {} to retain the current edits."})
        response = complete(messages, session.tools)
        msg = response["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        public_reply = {key: msg[key] for key in ("role", "content", "tool_calls") if key in msg}
        session.record_updater(response.get("usage"), messages, response=public_reply,
                               outcome="tool_calls" if calls else "text")
        if not calls:
            raise NoSubmission("updater finished without submitting a patch")
        messages.append(public_reply)
        for call in calls:
            function = call["function"]
            try:
                if len(function["arguments"]) > 262144:
                    raise ValueError("tool arguments exceed 256 KiB")
                result = session.call(function["name"], json.loads(function["arguments"]))
            except (ValueError, TypeError, RecursionError) as exc:
                result = {"error": str(exc)[:1024]}
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": encode_json(result)})
            if session.receipt is not None:
                return session.receipt
    raise NoSubmission("updater turn limit reached before submission")
