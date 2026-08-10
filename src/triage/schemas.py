"""Typed contracts between agents.

Every hop in the graph is a validated Pydantic model rather than free text. This is
deliberate: in a multi-agent system the model output *is* the wire format, and an
unvalidated hop turns one bad generation into a silent downstream corruption.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Specialist(str, Enum):
    COMPLIANCE = "compliance"
    SENTIMENT = "sentiment"
    RESOLUTION = "resolution"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RouteDecision(BaseModel):
    """Supervisor output: which specialist runs next, or finish."""

    next: Literal["compliance", "sentiment", "resolution", "finalize"] = Field(
        description="The specialist to dispatch next, or 'finalize' if enough evidence has been gathered."
    )
    reason: str = Field(description="One sentence explaining the routing choice.")


class Finding(BaseModel):
    """A single piece of evidence produced by a specialist."""

    source: Specialist
    severity: Severity
    summary: str = Field(description="What was found, in one sentence.")
    evidence: str = Field(
        default="",
        description="Verbatim quote from the transcript supporting this finding, or '' if none.",
    )
    policy_ref: str = Field(default="", description="Policy ID this relates to, e.g. 'FDCPA-807-11'.")


class ComplianceReport(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    required_disclosures_missing: list[str] = Field(default_factory=list)


class SentimentReport(BaseModel):
    escalation_risk: Severity
    customer_emotion: str = Field(description="Dominant customer emotion, one or two words.")
    findings: list[Finding] = Field(default_factory=list)


class ResolutionReport(BaseModel):
    recommended_action: str
    findings: list[Finding] = Field(default_factory=list)


class CaseFile(BaseModel):
    """The artifact the human approves or rejects before it is written downstream."""

    call_id: str
    headline: str = Field(description="Short title for the case, under 90 characters.")
    severity: Severity
    summary: str = Field(description="Two to four sentences an operations reviewer can act on.")
    recommended_action: str
    findings: list[Finding] = Field(default_factory=list)
    requires_human_review: bool = True


class ApprovalDecision(BaseModel):
    approved: bool
    note: str = ""
