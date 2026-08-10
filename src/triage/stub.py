"""A deterministic stand-in for a chat model.

Agent systems are hard to test because the interesting logic - routing, termination,
the approval gate, state reduction - sits *around* the model, but a live model makes
every assertion flaky. Pinning the model to canned, rule-derived responses lets the
graph be tested like ordinary software: same input, same trajectory, every time.

What this exercises: routing order, the step cap, tool dispatch, schema validation,
state reducers, the interrupt/resume cycle. What it deliberately does not exercise:
model quality. That belongs in an eval suite scored against golden cases, not in unit
tests.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from langchain_core.messages import AIMessage


def _text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


class _BoundStub:
    """What `bind_tools` returns: emits one tool call, then stops."""

    def __init__(self, parent: "StubChatModel", tools: list[Any]) -> None:
        self._parent = parent
        self._names = [getattr(t, "name", str(t)) for t in tools]

    def invoke(self, messages: list[Any], **_: Any) -> AIMessage:
        system = _text(messages[0]) if messages else ""
        key = self._parent._role(system)
        if self._parent._tool_calls_made.get(key):
            return AIMessage(content="No further tools needed.")
        self._parent._tool_calls_made[key] = True

        if key == "sentiment" and "get_account_context" in self._names:
            joined = " ".join(_text(m) for m in messages)
            match = re.search(r"AC-\d+", joined)
            args = {"account_id": match.group(0) if match else "AC-4417"}
            name = "get_account_context"
        else:
            name = "search_policy"
            args = {"query": "required disclosure recording verification"}

        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"stub-{key}-1"}],
        )


class StubChatModel:
    """Rule-based responder that satisfies the schemas the graph asks for."""

    def __init__(self) -> None:
        self._tool_calls_made: dict[str, bool] = {}

    # -- plumbing -----------------------------------------------------------
    def bind_tools(self, tools: list[Any], **_: Any) -> _BoundStub:
        return _BoundStub(self, tools)

    @staticmethod
    def _role(system: str) -> str:
        s = system.lower()
        if "supervisor of a contact-center" in s:
            return "supervisor"
        if "compliance analyst" in s:
            return "compliance"
        if "customer emotion" in s:
            return "sentiment"
        if "recommend the single best" in s:
            return "resolution"
        if "case file" in s:
            return "finalize"
        return "unknown"

    # -- responses ----------------------------------------------------------
    def invoke(self, messages: list[Any], **_: Any) -> AIMessage:
        system = _text(messages[0]) if messages else ""
        user = _text(messages[1]) if len(messages) > 1 else ""
        role = self._role(system)
        handler = getattr(self, f"_{role}", None)
        payload = handler(user) if handler else {}
        message = AIMessage(content=json.dumps(payload))
        message.usage_metadata = {  # type: ignore[attr-defined]
            "input_tokens": len(system) // 4 + len(user) // 4,
            "output_tokens": len(message.content) // 4,
            "total_tokens": 0,
        }
        return message

    def _supervisor(self, user: str) -> dict[str, Any]:
        match = re.search(r"Still available: (\[[^\]]*\])", user)
        remaining = ast.literal_eval(match.group(1)) if match else []
        nxt = remaining[0] if remaining else "finalize"
        return {"next": nxt, "reason": f"{nxt} has not run yet and is needed for a complete case."}

    def _compliance(self, user: str) -> dict[str, Any]:
        body = user.lower()
        findings: list[dict[str, Any]] = []
        missing: list[str] = []

        if "recorded" not in body and "monitored" not in body:
            missing.append("call recording disclosure")
        if "debt collector" not in body:
            missing.append("mini-Miranda debt collector disclosure")
        if "date of birth" not in body and "last four" not in body:
            missing.append("two-factor identity verification")

        if "garnish" in body or "arrest" in body or "sue you" in body:
            findings.append(
                {
                    "source": "compliance",
                    "severity": "high",
                    "summary": "Agent threatened consequences that may not be authorized.",
                    "evidence": "",
                    "policy_ref": "UDAAP-04",
                }
            )
        if "supervisor" in body and "transfer" not in body:
            findings.append(
                {
                    "source": "compliance",
                    "severity": "medium",
                    "summary": "Customer requested a supervisor and was not transferred.",
                    "evidence": "",
                    "policy_ref": "ESC-SUP-03",
                }
            )
        if "ignore" in body and "instruction" in body:
            findings.append(
                {
                    "source": "compliance",
                    "severity": "medium",
                    "summary": "Transcript contains text attempting to instruct the analysis system.",
                    "evidence": "",
                    "policy_ref": "",
                }
            )
        return {"findings": findings, "required_disclosures_missing": missing}

    def _sentiment(self, user: str) -> dict[str, Any]:
        body = user.lower()
        angry = any(w in body for w in ("unacceptable", "ridiculous", "furious", "lawyer", "complaint"))
        risk = "high" if angry else "low"
        return {
            "escalation_risk": risk,
            "customer_emotion": "frustrated" if angry else "neutral",
            "findings": (
                [
                    {
                        "source": "sentiment",
                        "severity": "medium",
                        "summary": "Customer expressed strong dissatisfaction and hinted at escalation.",
                        "evidence": "",
                        "policy_ref": "",
                    }
                ]
                if angry
                else []
            ),
        }

    def _resolution(self, user: str) -> dict[str, Any]:
        body = user.lower()
        if "hardship" in body or "laid off" in body or "lost my job" in body:
            action = "Enroll customer in hardship program and schedule a follow-up call."
        elif "supervisor" in body:
            action = "Schedule a supervisor callback within one business day."
        elif "dispute" in body:
            action = "Open a billing dispute case and suspend collection on the disputed amount."
        else:
            action = "No action required; archive the call."
        return {
            "recommended_action": action,
            "findings": [],
        }

    def _finalize(self, user: str) -> dict[str, Any]:
        match = re.search(r"Call ID: (\S+)", user)
        call_id = match.group(1) if match else "unknown"
        clean = "No findings reported." in user
        return {
            "call_id": call_id,
            "headline": "Call reviewed - no issues found" if clean else "Compliance and conduct issues identified",
            "severity": "none" if clean else "high",
            "summary": (
                "Automated triage found no policy violations on this call."
                if clean
                else "Automated triage identified policy violations and elevated escalation risk. "
                "See findings for the governing policy references and recommended action."
            ),
            "recommended_action": "Archive." if clean else "Route to quality review for agent coaching.",
            "findings": [],
            "requires_human_review": not clean,
        }
