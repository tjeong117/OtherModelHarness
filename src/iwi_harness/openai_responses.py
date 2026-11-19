"""OpenAI Responses transport for a single Astra editing episode."""

import copy
import json
import re
import threading
from decimal import Decimal
from urllib import error, request


class AstraBudgetExceeded(RuntimeError):
    """The next request cannot fit within the configured spending limit."""


class AstraProviderError(RuntimeError):
    """The provider could not complete a usable request."""


class AstraBudget:
    """An in-process spending ceiling shared by one or more episode callbacks.

    Monetary arithmetic uses microdollars. Requests with unknown usage retain their
    entire reservation, including after a network error.
    """

    def __init__(self, cap_usd):
        cap = Decimal(str(cap_usd))
        if not cap.is_finite() or cap <= 0:
            raise ValueError("cap_usd must be a positive finite amount")
        self._limit = int(cap * 1_000_000)
        self._used = 0
        self._lock = threading.Lock()

    @property
    def spent_upper_bound_usd(self):
        with self._lock:
            return self._used / 1_000_000

    def _reserve(self, amount):
        with self._lock:
            if self._used + amount > self._limit:
                raise AstraBudgetExceeded("Astra request exceeds the remaining spending limit")
            self._used += amount

    def _settle(self, reserved, actual_bound):
        with self._lock:
            self._used += actual_bound - reserved


# Upper bounds in microdollars per token for standard short-context inference.
# For input, include the highest cache write rate. Never request long-context
# pricing, accelerated service, or provider-hosted tools.
_RATES = {"gpt-6.2-astra": (20, 40), "gpt-6-astra": (25, 50)}


def _input_item(msg):
    if msg["role"] == "tool":
        return {"type": "function_call_output", "call_id": msg["tool_call_id"], "output": msg["content"]}
    if msg["role"] in ("system", "user"):
        return {"role": msg["role"], "content": msg["content"]}
    raise ValueError("unexpected assistant history; use one AstraResponses instance per episode")


def _token_count(value):
    return type(value) is int and value >= 0


class AstraResponses:
    """Stateful run_updater callback. Create one per episode and share its budget."""

    def __init__(self, model="gpt-6.2-astra", *, api_key, budget, reasoning="medium",
                 max_output_tokens=4096, timeout=300):
        if model not in _RATES:
            raise ValueError("no configured Astra price bound for this model")
        if not api_key:
            raise ValueError("the OpenAI API key is missing")
        if not isinstance(budget, AstraBudget):
            raise TypeError("budget must be an AstraBudget")
        if type(max_output_tokens) is not int or not 16 <= max_output_tokens <= 128_000:
            raise ValueError("max_output_tokens must be between 16 and 128000")
        if reasoning not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("unsupported Astra reasoning effort")
        self.model = model
        self.budget = budget
        self.reasoning = reasoning
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        self._timeout = timeout
        self._history = []
        self._native = []
        self.records = []

    def _post(self, path, payload):
        req = request.Request("https://api.openai.com/v1/" + path,
                              data=json.dumps(payload, allow_nan=False).encode(),
                              headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._api_key})
        try:
            with request.urlopen(req, timeout=self._timeout) as response:
                return json.load(response)
        except error.HTTPError as exc:
            # Authentication errors can quote credentials and input errors can quote tasks.
            detail = ""
            try:
                body = json.loads(exc.read(16_384))
                info = body.get("error", {}) if isinstance(body, dict) else {}
                # Structured labels locate an invalid schema without exposing its values.
                for field in ("code", "param"):
                    label = info.get(field) if isinstance(info, dict) else None
                    if (isinstance(label, str) and label != self._api_key
                            and re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9.\[\]]{0,150}", label)):
                        detail += f" {field}={label}"
            except (ValueError, OSError):
                pass
            raise AstraProviderError(f"OpenAI {path} returned HTTP {exc.code}{detail}") from None
        except (OSError, ValueError) as exc:
            raise AstraProviderError(f"OpenAI {path} failed ({type(exc).__name__})") from None

    def __call__(self, messages, tools):
        if messages[:len(self._history)] != self._history:
            raise ValueError("conversation changed; create a fresh AstraResponses for each episode")
        items = self._native + [_input_item(msg) for msg in messages[len(self._history):]]
        functions = [{"type": "function", **tool["function"], "strict": False} for tool in tools]
        common = {"model": self.model, "input": items, "tools": functions,
                  "reasoning": {"effort": self.reasoning}, "parallel_tool_calls": True}
        count = self._post("responses/input_tokens", common).get("input_tokens")
        if not _token_count(count):
            raise AstraProviderError("token-count endpoint returned no valid input count")
        if count > 272_000:
            raise AstraBudgetExceeded("Astra request exceeds the short-context pricing limit")
        input_rate, output_rate = _RATES[self.model]
        reserved = count * input_rate + self.max_output_tokens * output_rate
        self.budget._reserve(reserved)
        data = self._post("responses", {**common, "max_output_tokens": self.max_output_tokens,
                                       "store": False, "service_tier": "default",
                                       "include": ["reasoning.encrypted_content"]})
        usage = data.get("usage") or {}
        n_input, n_output = usage.get("input_tokens"), usage.get("output_tokens")
        if not (_token_count(n_input) and _token_count(n_output)):
            raise AstraProviderError("response returned no valid usage; retaining the full reservation")
        amount = n_input * input_rate + n_output * output_rate
        self.budget._settle(reserved, amount)
        normalized = {"prompt_tokens": n_input, "completion_tokens": n_output,
                      "total_tokens": n_input + n_output,
                      "prompt_tokens_details": usage.get("input_tokens_details") or {},
                      "completion_tokens_details": usage.get("output_tokens_details") or {},
                      "cost_upper_bound_usd": amount / 1_000_000}
        status = data.get("status")
        self.records.append({"response_id": data.get("id"), "model": data.get("model"),
                             "status": status, "usage": normalized})
        if status != "completed":
            reason = (data.get("incomplete_details") or {}).get("reason")
            raise AstraProviderError(f"Astra generation ended with status {status} ({reason})")
        output = data["output"]
        calls = [{"type": "function", "id": item["call_id"],
                  "function": {"name": item["name"], "arguments": item["arguments"]}}
                 for item in output if item["type"] == "function_call"]
        content = [part["text"] for item in output if item["type"] == "message"
                   for part in item["content"] if part["type"] == "output_text"]
        msg = {"role": "assistant", "content": "\n".join(content) or None, "tool_calls": calls}
        # Replay the complete native output with encrypted reasoning, item IDs and
        # message phases. Only the public projection is exposed to the session trace.
        self._native = items + output
        self._history = copy.deepcopy(messages + [msg])
        return {"choices": [{"message": msg}], "usage": normalized}
