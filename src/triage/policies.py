"""A small policy corpus plus lexical retrieval over it.

Retrieval here is TF-IDF-ish keyword scoring, not embeddings. For a corpus this size
that is the correct call: it is exact, explainable to a compliance reviewer, and has no
index to keep warm. `search_policy` is the seam - swapping in a vector store means
reimplementing one function, and nothing above it changes.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    id: str
    title: str
    text: str


CORPUS: list[Policy] = [
    Policy(
        id="FDCPA-807-11",
        title="Mini-Miranda disclosure",
        text=(
            "On every call concerning collection of a debt, the agent must state that the "
            "communication is from a debt collector and that any information obtained will be "
            "used for that purpose. On the initial communication the full disclosure is required. "
            "Omission of the mini-Miranda on an initial contact is a high severity violation."
        ),
    ),
    Policy(
        id="REC-DISC-01",
        title="Call recording disclosure",
        text=(
            "The agent must disclose that the call is recorded or monitored before collecting any "
            "personal information. In two-party consent states the customer must affirmatively "
            "consent. Failure to disclose recording before information collection is a high "
            "severity violation."
        ),
    ),
    Policy(
        id="AUTH-VER-02",
        title="Customer identity verification",
        text=(
            "Before discussing account balances, transactions, or personally identifiable "
            "information, the agent must verify the customer using at least two factors such as "
            "date of birth, last four of the account number, or a one-time passcode. Discussing "
            "balance details prior to verification is a high severity violation."
        ),
    ),
    Policy(
        id="ESC-SUP-03",
        title="Supervisor escalation on request",
        text=(
            "When a customer explicitly requests a supervisor or manager, the agent must "
            "acknowledge the request and either transfer or schedule a callback within one "
            "business day. Refusing or ignoring a supervisor request is a medium severity "
            "violation and a common driver of complaint escalation."
        ),
    ),
    Policy(
        id="UDAAP-04",
        title="Unfair, deceptive, or abusive acts or practices",
        text=(
            "Agents must not misstate the consequences of non-payment, threaten action that will "
            "not be taken, imply legal action that is not authorized, or apply pressure that a "
            "reasonable consumer would find abusive. Threatening arrest, wage garnishment, or "
            "credit destruction without authorization is a high severity UDAAP violation."
        ),
    ),
    Policy(
        id="HARD-SHIP-05",
        title="Hardship program offer",
        text=(
            "When a customer states a hardship such as job loss, medical event, or military "
            "deployment, the agent must offer available hardship or forbearance options before "
            "discussing further collection. Failing to offer hardship options after a disclosed "
            "hardship is a medium severity violation."
        ),
    ),
    Policy(
        id="DISP-BILL-06",
        title="Billing dispute handling",
        text=(
            "When a customer disputes a charge, the agent must open a dispute case, provide the "
            "provisional credit timeline, and stop collection activity on the disputed amount "
            "while the dispute is open. Continuing to demand payment on a disputed amount is a "
            "medium severity violation."
        ),
    ),
]

_BY_ID = {p.id: p for p in CORPUS}
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "was", "were", "on", "in",
    "for", "that", "this", "it", "be", "by", "with", "as", "at", "from", "not", "must",
}


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2]


_DOC_TOKENS = {p.id: Counter(_tokens(f"{p.title} {p.text}")) for p in CORPUS}
_DF = Counter()
for _counts in _DOC_TOKENS.values():
    _DF.update(_counts.keys())
_N = len(CORPUS)


def search(query: str, k: int = 3) -> list[tuple[Policy, float]]:
    """Return the top-k policies for `query`, scored by IDF-weighted term overlap."""
    q = _tokens(query)
    if not q:
        return []
    scored: list[tuple[Policy, float]] = []
    for pid, counts in _DOC_TOKENS.items():
        length = sum(counts.values()) or 1
        score = sum(
            (counts[term] / length) * math.log((_N + 1) / (_DF[term] + 1) + 1)
            for term in set(q)
            if term in counts
        )
        if score > 0:
            scored.append((_BY_ID[pid], score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def get(policy_id: str) -> Policy | None:
    return _BY_ID.get(policy_id)
