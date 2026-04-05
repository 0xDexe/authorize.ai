"""
AuthorizeAI — Shared Pipeline State
====================================
Pydantic-based state schema passed between LangGraph agent nodes.
Each agent reads from prior fields and writes to its designated output fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ──────────────────────────────────────────────────────────────────

class CriterionStatus(str, Enum):
    MET = "MET"
    NOT_MET = "NOT_MET"
    INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


# ── Nested Data Structures ─────────────────────────────────────────────────

@dataclass
class Treatment:
    drug: str = ""
    duration: str = ""
    outcome: str = ""  # "failed", "partial", "ongoing", "successful"
    dosage: str = ""


@dataclass
class CriterionResult:
    criterion_id: str = ""
    criterion_text: str = ""
    status: CriterionStatus = CriterionStatus.INSUFFICIENT
    evidence: str = ""  # supporting text from clinical record
    confidence: float = 0.0


@dataclass
class PredictionResult:
    approval_probability: float = 0.0
    risk_tier: str = ""  # "high", "medium", "low"
    key_factors: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)


# ── Main Pipeline State ────────────────────────────────────────────────────

@dataclass
class AuthorizeState:
    """
    Accumulative state object carried through the LangGraph pipeline.
    Each agent node reads upstream fields and populates its own section.
    """

    # ── Inputs (provided at pipeline invocation) ──
    raw_clinical_text: str = ""
    payer_id: str = ""
    procedure_code: str = ""  # CPT code
    patient_id: str = ""

    # ── Agent 1: Clinical Extraction outputs ──
    diagnosis_codes: list[str] = field(default_factory=list)       # ICD-10
    extracted_procedure_code: str = ""                              # confirmed CPT
    prior_treatments: list[Treatment] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    severity_score: float = 0.0
    patient_demographics: dict = field(default_factory=dict)       # age, sex, etc.
    lab_values: dict = field(default_factory=dict)
    extraction_confidence: float = 0.0

    # ── Agent 2: Payer Criteria Matching outputs ──
    criteria_results: list[CriterionResult] = field(default_factory=list)
    overall_coverage_score: float = 0.0   # weighted proportion of criteria met
    gaps: list[str] = field(default_factory=list)
    policy_reference: str = ""
    matched_policy_text: str = ""         # raw policy text retrieved

    # ── Agent 3: Approval Prediction outputs ──
    prediction: PredictionResult = field(default_factory=PredictionResult)

    # ── Agent 4: Submission Drafting outputs ──
    draft_letter: str = ""
    letter_type: str = "initial"  # "initial" or "appeal"
    cited_evidence: list[str] = field(default_factory=list)

    # ── Pipeline metadata ──
    status: PipelineStatus = PipelineStatus.PENDING
    errors: list[str] = field(default_factory=list)
    current_agent: str = ""
