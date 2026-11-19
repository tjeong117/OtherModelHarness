"""One isolated, metered editing episode for an external updater."""
import hashlib
import json
from pathlib import Path
from threading import Lock, RLock
from time import perf_counter
from weakref import WeakSet

from jsonschema import Draft202012Validator

from .causal import CausalTools
from .probes import Runner
from .tools import tool_schemas
from .weights import WeightStore


def encode_json(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


_claim_lock = Lock()
_claimed_models = WeakSet()


class EditingSession:
    """The host owns model allocation and artifacts. An updater receives tools, info and call."""

    def __init__(self, target, *, artifact_dir, sae=None, max_tool_calls=128,
                 max_forward_tokens=262144, max_forwards=256, token_counter=None, interpretability=True,
                 detailed_writes=False, adaptive_interpretability=True, max_fitted_dictionaries=3,
                 max_sae_fit_steps=256, behavior_checks=False, internal_measurements=False,
                 edit_preview=False, experiment_tools_only=False, public_cases=None):
        if type(max_tool_calls) is not int or max_tool_calls < 1:
            raise ValueError("max_tool_calls must be a positive integer")
        for name, limit in (("max_fitted_dictionaries", max_fitted_dictionaries),
                            ("max_sae_fit_steps", max_sae_fit_steps)):
            if type(limit) is not int or limit < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.artifact_dir = Path(artifact_dir)
        self._lock = RLock()
        self._count = token_counter
        self._interpretability = interpretability
        if internal_measurements and not interpretability:
            raise ValueError("internal_measurements requires interpretability")
        self._behavior_checks = (behavior_checks or internal_measurements or edit_preview
                                 or experiment_tools_only or public_cases is not None)
        self._internal_measurements = internal_measurements
        self._edit_preview = edit_preview
        self._standalone_probes = interpretability and not experiment_tools_only
        self._adaptive = self._standalone_probes and adaptive_interpretability
        self._can_test_sae = sae is not None or self._adaptive
        self._native_layers = []
        if internal_measurements:
            from .experiments import supported_native_layers
            self._native_layers = supported_native_layers(target)
        self._detailed_writes = detailed_writes
        self._calls = 0
        self._receipt = None
        self._failed = False
        self._limits = dict(max_tool_calls=max_tool_calls, max_forwards=max_forwards,
                            max_forward_tokens=max_forward_tokens)
        if self._adaptive:
            self._limits.update(max_fitted_dictionaries=max_fitted_dictionaries,
                                max_sae_fit_steps=max_sae_fit_steps)
        self._preloaded_id = sae.dictionary_id if self._adaptive and sae is not None else None
        self._dictionaries = {self._preloaded_id: sae} if self._preloaded_id is not None else {}
        self.store = WeightStore(target)
        self.runner = Runner(target, self.store, sae, max_forward_tokens=max_forward_tokens,
                             max_forwards=max_forwards, report_suppression_activity=self._adaptive)
        self._public_cases_json = None
        if self._behavior_checks:
            from .experiments import BehaviorExperiment
            self._experiment = BehaviorExperiment(self.runner, internal=self._internal_measurements,
                                                  artifact_dir=self.artifact_dir)
            if public_cases is not None:
                self._experiment.encode_cases(public_cases)
                self._public_cases_json = encode_json(public_cases)
        self._validators = {tool["function"]["name"]: Draft202012Validator(tool["function"]["parameters"])
                            for tool in self._tool_schemas(detailed_writes=True)}
        with _claim_lock:
            if target.model in _claimed_models:
                raise ValueError("a session requires its own freshly loaded target")
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            for name in ("trajectory.jsonl", "patch.safetensors", "receipt.json"):
                if (self.artifact_dir / name).exists():
                    raise FileExistsError(self.artifact_dir / name)
            # Exclusive creation also prevents two processes claiming the same artifact directory.
            with (self.artifact_dir / "session.json").open("x") as out:
                json.dump(self.info, out, indent=2)
            _claimed_models.add(target.model)
        causal = CausalTools(self.runner)
        self._methods = dict(read_weights=self.store.read, write_weights=self.store.write,
                             run_target=self.runner.run, inspect_target=self.runner.inspect,
                             contrast_target=self.runner.contrast, patch_activations=causal.patch_activations,
                             rank_attributions=causal.rank_attributions, feature_examples=self.runner.feature_examples,
                             submit_patch=self._submit)
        if self._adaptive:
            self._methods.update(discover_target=self._discover_target, fit_sae=self._fit_sae,
                                 select_sae=self._select_sae, tokenize_target=self._tokenize_target)
        if self._behavior_checks:
            self._methods.update(test_behavior=self._experiment.run, tokenize_target=self._tokenize_target)
        if self._edit_preview:
            from .preview import WeightPreview
            self._previews = WeightPreview(self.store)
            self._methods.update(preview_weights=self._preview_weights, commit_preview=self._previews.commit,
                                 discard_preview=self._previews.discard)

    @property
    def tools(self):
        return self._tool_schemas(detailed_writes=self._detailed_writes)

    def _tool_schemas(self, *, detailed_writes):
        schemas = tool_schemas(detailed_writes=detailed_writes, adaptive_interpretability=self._adaptive,
                               behavior_checks=self._behavior_checks, internal_measurements=self._internal_measurements,
                               edit_preview=self._edit_preview, sae_available=self._can_test_sae,
                               native_available=bool(self._native_layers), default_cases=self._public_cases_json is not None)
        if self._standalone_probes:
            return schemas
        visible = {"run_target", "read_weights", "write_weights", "submit_patch"}
        if self._behavior_checks:
            visible.update(("tokenize_target", "test_behavior"))
        if self._edit_preview:
            visible.update(("preview_weights", "commit_preview", "discard_preview"))
        schemas = [tool for tool in schemas if tool["function"]["name"] in visible]
        generation = next(tool["function"] for tool in schemas if tool["function"]["name"] == "run_target")
        generation["description"] = "Generate greedily on public prompts."
        for key in ("probe", "intervention"):
            del generation["parameters"]["properties"][key]
        return schemas

    @property
    def info(self):
        target, sae = self.runner.target, self.runner.sae
        schema = encode_json(self.tools)
        dictionary = None if sae is None or not self._interpretability else dict(id=sae.dictionary_id, revision=sae.revision,
                                                   layer=sae.layer, features=sae.n_features)
        template = getattr(getattr(target, "tokenizer", None), "chat_template", None)
        template_digest = hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None
        settings = dict(protocol="iwi-harness.v1", model_id=target.model_id, revision=target.revision,
                    chat_template_sha256=template_digest,
                    block_count=len(target.blocks), sae=dictionary, limits=dict(self._limits),
                    interpretability=self._interpretability,
                    tool_schema=self._size(schema))
        if self._adaptive:
            settings["adaptive_interpretability"] = True
        if self._behavior_checks:
            settings["public_experiments"] = dict(internal_measurements=self._internal_measurements,
                                                  edit_preview=self._edit_preview)
            if self._public_cases_json is not None:
                settings["public_experiments"]["default_case_count"] = len(json.loads(self._public_cases_json))
            if self._internal_measurements:
                settings["public_experiments"]["native_unit_layers"] = list(self._native_layers)
        return settings

    def _preview_weights(self, writes, cases, max_new_tokens=12, token_ids=None, expected_revision=None):
        from .experiments import compare_behavior

        def evaluate():
            return self._experiment.evaluate(cases, max_new_tokens=max_new_tokens, token_ids=token_ids)

        usage = self.runner.usage.copy()
        result = self._previews.preview(writes, evaluate, expected_revision=expected_revision)
        before, after = result.pop("before"), result.pop("after")
        result["behavior"] = compare_behavior(before, after)
        result["cost"] = {name: value - usage[name] for name, value in self.runner.usage.items()}
        record = dict(target=self.runner.target.model_id, checkpoint=self.runner.target.revision,
                      cases=cases, max_new_tokens=max_new_tokens, token_ids=token_ids, writes=writes,
                      candidate=result, before=before, after=after)
        artifact = Path("behavior") / f"preview-{result['candidate_id']}.json"
        path = self.artifact_dir / artifact
        path.parent.mkdir(exist_ok=True)
        path.write_text(encode_json(record) + "\n", encoding="utf-8")
        result["artifact"] = str(artifact)
        return result

    def _dictionary_info(self, sae):
        return dict(id=sae.dictionary_id, revision=sae.revision, layer=sae.layer,
                    features=sae.n_features, source="preloaded" if sae.dictionary_id == self._preloaded_id else "fitted",
                    fitted_weight_revision=getattr(sae, "fitted_weight_revision", None))

    def _fit_budget(self):
        fitted = len(self._dictionaries) - int(self._preloaded_id is not None)
        used_steps = self.runner.usage.get("sae_fit_steps", 0)
        return dict(remaining_dictionaries=self._limits["max_fitted_dictionaries"] - fitted,
                    remaining_steps=self._limits["max_sae_fit_steps"] - used_steps,
                    fitted_dictionaries=fitted, used_steps=used_steps)

    def _discover_target(self):
        from .targets import discover_sites
        target = self.runner.target
        active = self.runner.sae
        return dict(model_id=target.model_id, revision=target.revision, weight_revision=self.store.revision,
                    sites=discover_sites(target), active_dictionary=active.dictionary_id if active is not None else None,
                    dictionaries=[self._dictionary_info(sae) for sae in self._dictionaries.values()],
                    fit_budget=self._fit_budget())

    def _tokenize_target(self, **arguments):
        from .tokenization import tokenize_target
        return tokenize_target(self.runner.target, **arguments)

    def _fit_sae(self, prompts, layer, *, steps=32, **options):
        from .dictionaries import fit_dictionary
        budget = self._fit_budget()
        if budget["remaining_dictionaries"] < 1:
            raise ValueError("fitted dictionary limit reached; select an existing dictionary")
        if steps > budget["remaining_steps"]:
            raise ValueError(f"SAE fit exceeds remaining optimizer steps ({budget['remaining_steps']})")
        sae, report = fit_dictionary(self.runner, prompts, layer, steps=steps, **options)
        previous = self._dictionaries.get(sae.dictionary_id)
        if previous is not None and (sae.dictionary_id == self._preloaded_id or sae.revision != previous.revision):
            raise ValueError("dictionary ID conflicts with an existing dictionary")
        result = {**report, "active_dictionary": sae.dictionary_id, "dictionary": self._dictionary_info(sae)}
        encode_json(result)
        self._dictionaries[sae.dictionary_id] = sae
        self.runner.sae = sae
        result["fit_budget"] = self._fit_budget()
        return result

    def _select_sae(self, dictionary_id):
        if dictionary_id is not None and dictionary_id not in self._dictionaries:
            raise ValueError("unknown dictionary ID; call discover_target to list session dictionaries")
        active = self._dictionaries.get(dictionary_id)
        self.runner.sae = active
        return dict(active_dictionary=dictionary_id,
                    dictionary=self._dictionary_info(active) if active is not None else None)

    @property
    def receipt(self):
        return None if self._receipt is None else self._receipt.copy()

    def _size(self, serialized):
        result = {"utf8_bytes": len(serialized.encode("utf-8"))}
        if self._count is not None:
            result["tokens"] = self._count(serialized)
        return result

    def _submit(self):
        if self._receipt is None:
            receipt = self.store.export(self.artifact_dir / "patch.safetensors")
            receipt.pop("path")
            self._receipt = receipt
            with (self.artifact_dir / "receipt.json").open("x") as stream:
                json.dump(receipt, stream, indent=2)
        return self.receipt

    def _log(self, event):
        with (self.artifact_dir / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encode_json(event) + "\n")

    def record_updater(self, usage, messages, *, outcome=None, response=None):
        """Record measured provider usage separately from serialized message estimates."""
        with self._lock:
            self._log(dict(type="updater", usage=usage, context=self._size(encode_json(messages)),
                           outcome=outcome, response=response,
                           task=messages[1]["content"] if len(messages) == 2 else None))

    def call(self, name, arguments):
        with self._lock:
            if self._failed:
                raise RuntimeError("session stopped after artifact storage failure")
            if self._receipt is not None:
                if name == "submit_patch" and arguments == {}:
                    return self.receipt
                raise ValueError("the submitted session is frozen")
            if self._calls >= self._limits["max_tool_calls"] and (name != "submit_patch" or arguments != {}):
                raise ValueError("tool call budget exhausted")
            self._calls += 1
            start = perf_counter()
            event = dict(type="tool", call=self._calls, tool=name)
            try:
                if name not in self._validators:
                    raise ValueError("unknown tool")
                serialized = encode_json(arguments)
                if len(serialized.encode("utf-8")) > 262144:
                    raise ValueError("tool arguments exceed 256 KiB")
                event.update(arguments=arguments, request=self._size(serialized))
                problem = next(self._validators[name].iter_errors(arguments), None)
                if problem:
                    if (name in ("test_behavior", "preview_weights") and problem.absolute_path
                            and problem.absolute_path[0] == "token_ids"):
                        raise ValueError("Omit token_ids to score full answers. For an optional first-token "
                                         "margin, pass two distinct integers [wanted_next_id, alternative_next_id].")
                    if (name == "write_weights" and isinstance(arguments, dict)
                            and {"tensor", "operation"} <= arguments.keys()):
                        operands = {k: v for k, v in arguments.items() if k not in ("tensor", "operation", "selection")}
                        self.store.write_destination(arguments["tensor"], arguments["operation"], operands)
                        if (problem.validator == "anyOf" and len(problem.absolute_path) == 1
                                and problem.absolute_path[0] in ("source", "value", "left", "right")):
                            operand = problem.absolute_path[0]
                            raise ValueError(f"Use a number, {{tensor, selection?}}, {{handle, selection?}}, "
                                             f"{{values: [...]}}, or an arithmetic expression for {operand}.")
                    location = ".".join(map(str, problem.absolute_path)) or "arguments"
                    raise ValueError(f"{location}: {problem.message[:512]}")
                if (self._public_cases_json is not None and name in ("test_behavior", "preview_weights")
                        and "cases" not in arguments):
                    arguments = {**arguments, "cases": json.loads(self._public_cases_json)}
                result = self._methods[name](**arguments)
                if self._adaptive and (result.get("probe") or {}).get("kind") == "sae":
                    fitted_at = getattr(self.runner.sae, "fitted_weight_revision", None)
                    if fitted_at is not None:
                        result["probe"]["fitted_weight_revision"] = fitted_at
                event.update(result=result, response=self._size(encode_json(result)))
                return result
            except Exception as exc:
                event["error"] = str(exc)[:1024]
                if isinstance(exc, OSError):
                    self._failed = True
                raise
            finally:
                event.update(elapsed_seconds=perf_counter() - start, target_usage=self.runner.usage.copy(),
                             weight_revision=self.store.revision)
                if self._adaptive:
                    active = self.runner.sae
                    event["active_dictionary"] = active.dictionary_id if active is not None else None
                try:
                    self._log(event)
                except OSError:
                    self._failed = True
                    raise

    def submit(self):
        return self.call("submit_patch", {})
