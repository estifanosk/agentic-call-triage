"""Tools the specialists may call, and the one write action they may not.

Two tiers on purpose:

  READ_TOOLS  - bound to the model. Read-only, no side effects, safe to call in a loop.
  file_case() - a plain function. It is NOT in any tool list, so the model cannot reach
                it. The graph calls it only after a human approves the case file.

That split is the answer to OWASP LLM "excessive agency": you do not ask a model to be
disciplined about destructive actions, you remove the capability and gate it structurally.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from langchain_core.tools import tool

from triage import policies

CASE_SERVICE_URL = os.environ.get("CASE_SERVICE_URL", "http://localhost:8080")

# Stand-in for the CRM. Real deployments hit a service with the *end user's* credentials,
# not a shared service account, so tool reach is bounded by who is actually asking.
_ACCOUNTS: dict[str, dict[str, Any]] = {
    "AC-4417": {
        "tenure_months": 62,
        "product": "card",
        "delinquency_days": 47,
        "prior_complaints": 1,
        "hardship_flag": True,
        "ssn": "412-88-9930",
    },
    "AC-9082": {
        "tenure_months": 8,
        "product": "card",
        "delinquency_days": 0,
        "prior_complaints": 0,
        "hardship_flag": False,
        "ssn": "551-22-1187",
    },
}

_SENSITIVE = {"ssn", "dob", "card_number"}


def _redact(record: dict[str, Any]) -> dict[str, Any]:
    """Strip sensitive fields before they reach a prompt.

    Redaction happens at the tool boundary, not in the prompt. Anything that reaches the
    context window should be assumed to be logged, cached, and potentially echoed back.
    """
    return {k: v for k, v in record.items() if k not in _SENSITIVE}


@tool
def search_policy(query: str) -> str:
    """Search the contact-center compliance policy corpus.

    Use this to find the policy that governs a behaviour you observed in the transcript.
    Returns the top matching policies with their IDs, titles, and full text.
    """
    hits = policies.search(query, k=3)
    if not hits:
        return "No matching policy found."
    return "\n\n".join(f"[{p.id}] {p.title}\n{p.text}" for p, _ in hits)


@tool
def get_account_context(account_id: str) -> str:
    """Look up non-sensitive account context for the customer on this call.

    Returns tenure, product, delinquency days, prior complaint count, and hardship flag.
    Sensitive identifiers are never returned.
    """
    record = _ACCOUNTS.get(account_id.strip().upper())
    if record is None:
        return f"No account found for {account_id!r}."
    return json.dumps(_redact(record))


READ_TOOLS = [search_policy, get_account_context]
TOOLS_BY_NAME = {t.name: t for t in READ_TOOLS}

# ---------------------------------------------------------------------------
# Write path. Not a tool. Reachable only from the graph, only after approval.
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (the |your )?(above|previous|system)",
    r"you are now",
    r"new instructions?:",
    r"system prompt",
    r"do not (flag|report|escalate|file)",
    r"mark (this|the) (call|case) as (clean|compliant|resolved)",
]


def scan_for_injection(text: str) -> list[str]:
    """Flag instruction-shaped spans in untrusted input.

    A transcript is data, not instruction. It is third-party text that reaches the
    context window, which makes it the same threat class as a retrieved document or a
    scraped page. This scanner is a tripwire, not a control - the actual control is that
    the transcript is delivered inside a fenced user block and the write path is gated
    on a human. Detection is best-effort; containment is structural.
    """
    return [
        pattern
        for pattern in _INJECTION_PATTERNS
        if re.search(pattern, text, re.IGNORECASE)
    ]


def file_case(case: dict[str, Any], *, timeout: float = 5.0) -> dict[str, Any]:
    """POST an approved case to the Java case-management service.

    Falls back to a local record if the service is not running, so the demo still
    completes end to end without the JVM up.
    """
    try:
        response = httpx.post(f"{CASE_SERVICE_URL}/cases", json=case, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, OSError) as exc:
        return {
            "status": "offline",
            "detail": f"case service unreachable ({exc.__class__.__name__}); case not persisted",
            "case_id": None,
        }
