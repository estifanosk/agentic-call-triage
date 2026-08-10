"""Model provider boundary.

The graph never imports a provider directly. It asks for `get_llm()` and gets back
something with a LangChain chat interface. Three backends are supported:

  ollama  - local dev on a laptop (OpenAI-compatible under the hood)
  vllm    - self-hosted vLLM in a cluster, spoken to over its OpenAI-compatible API
  stub    - deterministic canned responses, so tests and CI never touch a GPU

The ollama -> vllm move is a base_url and model-name change, nothing else. That is the
whole point of pinning the boundary here: the serving tier is an infrastructure
decision, not an application one.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_OLLAMA_MODEL = os.environ.get("TRIAGE_MODEL", "qwen2.5:7b")
DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")


@dataclass
class CallStats:
    """Per-call telemetry. Aggregated by the tracer into a run-level report."""

    node: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    repairs: int = 0
    degraded: bool = False


@dataclass
class Meter:
    """Collects CallStats across a run. Stands in for OTel spans in this demo."""

    calls: list[CallStats] = field(default_factory=list)

    def record(self, stats: CallStats) -> None:
        self.calls.append(stats)

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)

    @property
    def total_ms(self) -> int:
        return sum(c.latency_ms for c in self.calls)


def get_llm(provider: str | None = None, model: str | None = None, temperature: float = 0.0) -> Any:
    """Return a chat model for the requested provider."""
    provider = provider or os.environ.get("TRIAGE_PROVIDER", "ollama")

    if provider == "stub":
        from triage.stub import StubChatModel

        return StubChatModel()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or DEFAULT_OLLAMA_MODEL,
            temperature=temperature,
            num_ctx=8192,
        )

    if provider == "vllm":
        # vLLM exposes an OpenAI-compatible server: `vllm serve <model> --port 8000`.
        # Nothing above this line changes when you swap laptop for cluster.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            base_url=DEFAULT_VLLM_BASE_URL,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            temperature=temperature,
        )

    raise ValueError(f"unknown provider: {provider!r} (expected ollama, vllm, or stub)")


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a response that may be wrapped in prose or fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _usage(response: Any) -> tuple[int, int]:
    meta = getattr(response, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0) or 0), int(meta.get("output_tokens", 0) or 0)


def structured(
    llm: Any,
    schema: type[T],
    system: str,
    user: str,
    *,
    node: str,
    meter: Meter | None = None,
    max_repairs: int = 2,
) -> T:
    """Call the model and validate its output against `schema`, repairing on failure.

    Small self-hosted models drift from a schema often enough that this loop is not
    optional. Feeding the validation error back is far cheaper than a retry from
    scratch, and the repair count is recorded so schema drift shows up as a metric
    instead of an outage.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    spec = json.dumps(schema.model_json_schema(), indent=2)
    sys_prompt = (
        f"{system}\n\n"
        "Reply with a single JSON object and nothing else. No prose, no code fences.\n"
        f"It must validate against this JSON Schema:\n{spec}"
    )
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user)]

    started = time.perf_counter()
    prompt_tokens = completion_tokens = 0
    last_error: Exception | None = None

    for attempt in range(max_repairs + 1):
        response = llm.invoke(messages)
        p, c = _usage(response)
        prompt_tokens += p
        completion_tokens += c
        raw = response.content if isinstance(response.content, str) else str(response.content)

        try:
            parsed = schema.model_validate_json(_extract_json(raw))
        except (ValidationError, ValueError) as exc:
            last_error = exc
            messages.extend(
                [
                    response,
                    HumanMessage(
                        content=(
                            f"That did not validate: {exc}\n"
                            "Return only the corrected JSON object."
                        )
                    ),
                ]
            )
            continue

        if meter is not None:
            meter.record(
                CallStats(
                    node=node,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    repairs=attempt,
                )
            )
        return parsed

    raise RuntimeError(f"{node}: model failed schema after {max_repairs} repairs: {last_error}")
