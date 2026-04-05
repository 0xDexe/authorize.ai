"""
AuthorizeAI — Feature Engineering
==================================
Constructs the feature vector for Agent 3's approval prediction model.
Combines real-time clinical-criteria matching output with historical
base rate data from CMS, KFF, and state regulatory sources.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ..state import AuthorizeState, CriterionStatus

# ── Historical base rate lookups ───────────────────────────────────────────
# These tables are populated from public CMS PUF, KFF, and OIG data.
# In production, these would live in SQLite; for the prototype, inline dicts.

# Source: KFF analysis of CMS Part C Reporting data (2024)
PAYER_DENIAL_RATES = {
    "UHC":       0.128,  # UnitedHealthcare
    "ELEVANCE":  0.042,  # Elevance Health (Anthem)
    "HUMANA":    0.058,
    "AETNA":     0.119,
    "CENTENE":   0.123,
    "KAISER":    0.109,
    "CIGNA":     0.085,
    "DEFAULT":   0.077,  # national average
}

# Source: CMS OIG report + published radiology PA studies
PROCEDURE_CATEGORY_APPROVAL_RATES = {
    "MRI":               0.88,
    "CT":                0.91,
    "SPECIALTY_REFERRAL": 0.85,
    "BRAND_DRUG":        0.78,
    "SURGERY":           0.82,
    "DME":               0.80,
    "DEFAULT":           0.85,
}

# CPT code to procedure category mapping (hackathon scope)
CPT_TO_CATEGORY = {
    "72148": "MRI",    # MRI lumbar
    "70553": "MRI",    # MRI brain
    "75557": "MRI",    # MRI cardiac
    "74177": "CT",     # CT abdomen
    "71260": "CT",     # CT chest
    "99242": "SPECIALTY_REFERRAL",  # cardiology consult
    "99243": "SPECIALTY_REFERRAL",  # neurology consult
    "J0717": "BRAND_DRUG",         # biologic
    "J2357": "BRAND_DRUG",         # specialty pharmacy
}


@dataclass
class FeatureVector:
    """Flat feature representation for the ML model."""
    coverage_score: float          # from Agent 2
    criteria_met_ratio: float      # met / total
    criteria_not_met_ratio: float  # not met / total
    criteria_insufficient_ratio: float
    num_gaps: int
    severity_score: float          # from Agent 1
    num_prior_treatments: int
    num_failed_treatments: int
    has_symptoms: int              # binary
    num_diagnosis_codes: int
    payer_denial_rate: float       # historical from CMS/KFF
    procedure_approval_rate: float # historical from CMS
    patient_age: float
    is_female: int                 # binary
    extraction_confidence: float   # from Agent 1

    def to_list(self) -> list[float]:
        """Convert to a flat float list for sklearn input."""
        return [
            self.coverage_score,
            self.criteria_met_ratio,
            self.criteria_not_met_ratio,
            self.criteria_insufficient_ratio,
            float(self.num_gaps),
            self.severity_score,
            float(self.num_prior_treatments),
            float(self.num_failed_treatments),
            float(self.has_symptoms),
            float(self.num_diagnosis_codes),
            self.payer_denial_rate,
            self.procedure_approval_rate,
            self.patient_age,
            float(self.is_female),
            self.extraction_confidence,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "coverage_score", "criteria_met_ratio", "criteria_not_met_ratio",
            "criteria_insufficient_ratio", "num_gaps", "severity_score",
            "num_prior_treatments", "num_failed_treatments", "has_symptoms",
            "num_diagnosis_codes", "payer_denial_rate", "procedure_approval_rate",
            "patient_age", "is_female", "extraction_confidence",
        ]


def extract_features(state: AuthorizeState) -> FeatureVector:
    """
    Build the feature vector from the accumulated pipeline state.
    Called by Agent 3 before model inference.
    """
    # Criteria ratios
    total_criteria = max(len(state.criteria_results), 1)
    met = sum(1 for c in state.criteria_results if c.status == CriterionStatus.MET)
    not_met = sum(1 for c in state.criteria_results if c.status == CriterionStatus.NOT_MET)
    insufficient = total_criteria - met - not_met

    # Prior treatment stats
    num_treatments = len(state.prior_treatments)
    num_failed = sum(
        1 for t in state.prior_treatments
        if t.outcome.lower() in ("failed", "partial")
    )

    # Historical base rates
    payer_key = state.payer_id.upper()
    payer_denial = PAYER_DENIAL_RATES.get(
        payer_key, PAYER_DENIAL_RATES["DEFAULT"]
    )

    proc_category = CPT_TO_CATEGORY.get(
        state.extracted_procedure_code or state.procedure_code, "DEFAULT"
    )
    proc_approval = PROCEDURE_CATEGORY_APPROVAL_RATES.get(
        proc_category, PROCEDURE_CATEGORY_APPROVAL_RATES["DEFAULT"]
    )

    # Demographics
    age = float(state.patient_demographics.get("age", 50) or 50)
    sex = state.patient_demographics.get("sex", "").upper()
    is_female = 1 if sex in ("F", "FEMALE") else 0

    return FeatureVector(
        coverage_score=state.overall_coverage_score,
        criteria_met_ratio=met / total_criteria,
        criteria_not_met_ratio=not_met / total_criteria,
        criteria_insufficient_ratio=insufficient / total_criteria,
        num_gaps=len(state.gaps),
        severity_score=state.severity_score,
        num_prior_treatments=num_treatments,
        num_failed_treatments=num_failed,
        has_symptoms=1 if state.symptoms else 0,
        num_diagnosis_codes=len(state.diagnosis_codes),
        payer_denial_rate=payer_denial,
        procedure_approval_rate=proc_approval,
        patient_age=age,
        is_female=is_female,
        extraction_confidence=state.extraction_confidence,
    )
