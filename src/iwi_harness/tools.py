"""Function-calling interface used by both evaluation and training loops."""
from copy import deepcopy

def obj(properties, required=()):
    schema = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = list(required)
    return schema

def arr(items, maximum, minimum=1):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}

def integer(low, high):
    return {"type": "integer", "minimum": low, "maximum": high}

NAME = {"type": "string", "minLength": 1, "maxLength": 512}
INDEX = integer(0, 2**40)
SELECT = arr({"oneOf": [INDEX, arr(INDEX, 2, 2)]}, 12, 0)
# Required and optional operands for each operation, shared with the tensor worker.
WRITE_OPERANDS = {
    "copy": (("source",), ()),
    "replace": (("value",), ()),
    "fill": (("value",), ()),
    "scale": (("factor",), ()),
    "add": (("value",), ()),
    "low_rank_add": (("left", "right"), ("factor",)),
}


def validate_write_operands(operation, operands):
    if not isinstance(operation, str) or operation not in WRITE_OPERANDS:
        raise ValueError(f"Choose a write operation from {', '.join(WRITE_OPERANDS)}")
    required, optional = WRITE_OPERANDS[operation]
    missing = sorted(set(required) - operands.keys())
    extra = sorted(operands.keys() - set(required + optional))
    if missing or extra:
        instructions = []
        if missing:
            instructions.append("supply " + ", ".join(missing))
        if extra:
            instructions.append("remove " + ", ".join(extra))
        raise ValueError(f"For {operation}, {' and '.join(instructions)}. Required operands: {', '.join(required)}.")
    if operation == "fill" and type(operands["value"]) not in (int, float):
        raise ValueError("fill requires a numeric scalar value")


def _write_schema():
    number = {"type": "number"}
    expression = {"$ref": "#/$defs/expression"}
    schema = obj({
        "tensor": NAME, "operation": {"enum": list(WRITE_OPERANDS)}, "selection": SELECT,
        "source": expression, "value": expression, "left": expression, "right": expression, "factor": number,
    }, ["tensor", "operation"])
    # Keep an object at the schema root for function-calling transports.
    variants = []
    for operation, (required, optional) in WRITE_OPERANDS.items():
        fields = {key: {} for key in ("tensor", "selection", *required, *optional)}
        fields["operation"] = {"const": operation}
        if operation == "fill":
            fields["value"] = number
        variants.append(obj(fields, required))
    schema["dependentSchemas"] = {"operation": {"oneOf": variants}}
    # The worker checks nested arithmetic, literal size, shapes and bounds.
    reference_selection = {"type": ["array", "null"]}
    schema["$defs"] = {"expression": {"anyOf": [
        number,
        obj({"tensor": {"type": "string"}, "selection": reference_selection}, ["tensor"]),
        obj({"handle": {"type": "string"}, "selection": reference_selection}, ["handle"]),
        obj({"values": {"type": ["array", "number"]}}, ["values"]),
        obj({"op": {"enum": ["add", "subtract", "outer"]}, "left": {}, "right": {}}, ["op", "left", "right"]),
        obj({"op": {"const": "scale"}, "left": {}, "factor": number}, ["op", "left", "factor"]),
    ]}}
    return schema


PROMPT = {"type": "string", "minLength": 1, "maxLength": 65536}
PROBE = obj({
    "layer": integer(0,1024), "kind": {"enum": ["residual","sae"]},
    "positions": arr({"type":"integer"},16), "top_k": integer(1,16),
    "phase": {"enum":["prefill","each_step"]},
}, ["kind"])
PROBE["if"] = {"properties": {"kind": {"const": "residual"}}}
PROBE["then"] = {"required": ["layer"]}
PROBE["description"] = "For an SAE probe omit layer to use the loaded dictionary. Residual probes require a block index."
TOKEN_PAIR = {**arr(INDEX, 2, 2), "uniqueItems": True,
              "description": "Ordered next-token IDs [wanted, alternative]. The margin is their logit difference."}
SITE = obj({"layer": integer(0, 1024), "component": {"enum": ["residual", "attention", "mlp"]}},
           ["layer", "component"])
CAUSAL = obj({"source_prompt": PROMPT, "target_prompt": PROMPT, "token_ids": TOKEN_PAIR,
              "sites": {**arr(SITE, 12), "uniqueItems": True},
              "source_position": {"type": "integer"}, "target_position": {"type": "integer"},
              "top_k": integer(1, 12)}, ["source_prompt", "target_prompt", "token_ids", "sites"])
INTERVENE = obj({"feature": INDEX, "scale":{"type":"number"}, "value":{"type":"number"}}, ["feature"])
INTERVENE["oneOf"] = [{"required":["scale"]},{"required":["value"]}]
OPTIONAL_INTERVENTION = {"anyOf": [INTERVENE, {"type": "null"}]}
TOOLS = {
    "read_weights": ("With no name, page through the tensor manifest. Read a weight slice or captured handle by reference, summary or bounded numeric values. Select axes by integer or [start,stop).", obj({
        "tensor":NAME,"handle":NAME,"selection":SELECT, "mode":{"enum":["ref","summary","values"]},
        "offset":INDEX,"limit":integer(1,256),
    })),
    "write_weights": ("Edit a writable slice. Get exact names and shapes with read_weights({}). Copy uses source, replace/add use value of the selected shape, fill uses a scalar value, scale uses factor, and low_rank_add uses left/right with optional factor. Reference a slice with {tensor,selection} or {handle}. A successful write expires existing handles.", _write_schema()),
    "run_target": ("Generate greedily on public prompts. Attach an optional block-output probe to this live run. An SAE intervention runs a baseline and a temporary counterfactual. Position -1 observes the last input token. Omit positions to use [-1].",obj({
        "prompts":arr(PROMPT,8),"max_new_tokens":integer(1,256),"probe":PROBE,"intervention":OPTIONAL_INTERVENTION,
    },["prompts"])),
    "inspect_target": ("Inspect the generation-ready prompt prefix. Return bounded observations and next-token scores. Supply two token IDs for their ordered logit margin; an intervention also returns the unmodified run.",obj({
        "prompts":arr(PROMPT,4),"probe":PROBE,"intervention":OPTIONAL_INTERVENTION,"token_ids":arr(INDEX,16),
    },["prompts","probe"])),
    "contrast_target": ("Compare matched public prompts at one site, ranking observed coordinates by their paired activation differences. left and right must contain the same number of prompts.",obj({
        "left":arr(PROMPT,4),"right":arr(PROMPT,4),"probe":PROBE,"top_k":integer(1,16),
    },["left","right","probe"])),
    "patch_activations": ("Temporarily replace one target activation at a time with the matched source activation. "
                          "Rank tested sites by their measured effect on the target next-token margin. "
                          "Positions index rendered prompt tokens and default to -1.", CAUSAL),
    "rank_attributions": ("Estimate each site's effect on the target next-token margin using the target gradient "
                          "and the difference between source and target activations. Use patch_activations to test "
                          "an estimate. Positions default to the final prompt tokens.", CAUSAL),
    "feature_examples": ("Show examples of a loaded SAE feature on supplied public prompts, including its positive "
                         "activation frequency. Supply an ordered token pair to also measure the effect of suppression "
                         "at each returned example. Omit positions to scan all prompt tokens (512 total maximum).",
                         obj({"prompts": arr(PROMPT, 8), "feature": INDEX,
                              "positions": {**arr({"type": "integer"}, 16), "uniqueItems": True},
                              "top_k": integer(1, 5), "token_ids": TOKEN_PAIR}, ["prompts", "feature"])),
    "submit_patch": ("Freeze the current numerical patch for clean replay. The artifact location is chosen by the host.",obj({})),
}

ADAPTIVE_TOOLS = {
    "discover_target": ("Discover the current target, its supported activation sites, available SAE dictionaries, "
                        "the active dictionary and remaining SAE fitting budget.", obj({})),
    "tokenize_target": (
        "Look up paged input token positions in a rendered target prompt and token IDs of literal answer strings. "
        "Include leading whitespace when relevant. This uses the target tokenizer and consumes no target forwards.",
        {**obj({"prompt": PROMPT, "texts": arr({"type": "string", "minLength": 1, "maxLength": 512}, 8),
                "offset": INDEX, "limit": integer(1, 64)}),
         "anyOf": [{"required": ["prompt"]}, {"required": ["texts"]}]}),
    "fit_sae": (
        "Fit and activate a session-local TopK sparse autoencoder on post-block residuals at a layer of your choice. "
        "Supply public training prompts and optional separate validation prompts. Otherwise supply at least two "
        "prompts so the final fifth can be held out. Select up to 1,024 total rows over at most 4,096 encoded tokens. "
        "Omit positions to use all tokens. The fit reports held-out reconstruction and feature activity.",
        obj({"prompts": arr(PROMPT, 16), "layer": integer(0, 1024), "n_features": integer(2, 256),
             "top_k": {**integer(1, 256), "description": "Maximum active features per token, up to n_features."},
             "steps": integer(1, 128), "seed": integer(0, 2**31 - 1),
             "positions": {**arr(integer(-2048, 2047), 16), "uniqueItems": True},
             "validation_prompts": arr(PROMPT, 8)}, ["prompts", "layer"])),
    "select_sae": ("Choose a dictionary listed by discover_target for subsequent SAE probes, feature examples and "
                   "temporary interventions. Pass null to clear the selection.", obj({
                       "dictionary_id": {"anyOf": [NAME, {"type": "null"}]},
                   }, ["dictionary_id"])),
}


def _wire_write_schema(schema, detailed):
    schema = deepcopy(schema)
    if not detailed:
        del schema["dependentSchemas"], schema["$defs"]
        for key in ("source", "value", "left", "right"):
            schema["properties"][key] = {"anyOf": [{"type": "object"}, {"type": "number"}]}
    return schema


def tool_schemas(*, detailed_writes=False, adaptive_interpretability=True, behavior_checks=False,
                 internal_measurements=False, edit_preview=False, sae_available=True):
    tools = []
    available = (TOOLS | ADAPTIVE_TOOLS) if adaptive_interpretability else TOOLS
    if behavior_checks or internal_measurements or edit_preview:
        from .experiments import behavior_tool
        available = {**available, "tokenize_target": ADAPTIVE_TOOLS["tokenize_target"],
                     "test_behavior": behavior_tool(internal_measurements, sae_available=sae_available)}
    if edit_preview:
        write = _wire_write_schema(TOOLS["write_weights"][1], detailed_writes)
        definitions = write.pop("$defs", None)
        properties = behavior_tool(False)[1]["properties"]
        preview = obj({"writes": arr(write, 8), **properties, "expected_revision": INDEX}, ["writes", "cases"])
        if definitions:
            preview["$defs"] = definitions
        candidate = obj({"candidate_id": NAME}, ["candidate_id"])
        available.update({
            "preview_weights": ("Measure public answers before and after up to eight proposed writes, then "
                                "restore current weights. Returns a candidate ID and per-slice outcomes. "
                                "A new successful preview replaces the previous candidate.", preview),
            "commit_preview": ("Apply the exact candidate from preview_weights at the same weight revision.", candidate),
            "discard_preview": ("Forget the named pending preview and keep the current weights.", candidate),
        })
    for name, (description, full_schema) in available.items():
        schema = _wire_write_schema(full_schema, detailed_writes) if name == "write_weights" else deepcopy(full_schema)
        tools.append({"type": "function", "function": {
            "name": name, "description": description, "parameters": schema,
        }})
    return tools
