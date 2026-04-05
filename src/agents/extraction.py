"""
AuthorizeAI — Agent 1: Clinical Extraction
============================================
LLM-based structured extraction from clinical text.
Extracts diagnosis codes, procedure codes, prior treatments,
symptoms, severity, demographics, and lab values into typed fields.
"""

import json
from dataclasses import asdict

from ..state import AuthorizeState, Treatment, PipelineStatus
from ..prompts.templates import EXTRACTION_SYSTEM, build_extraction_user_prompt
from ..utils.llm_client import call_llm


def extraction_agent(state: AuthorizeState) -> AuthorizeState:
    """
    LangGraph node function for clinical data extraction.
    Reads: raw_clinical_text, procedure_code
    Writes: diagnosis_codes, extracted_procedure_code, prior_treatments,
            symptoms, severity_score, patient_demographics, lab_values,
            extraction_confidence
    """
    state.current_agent = "extraction"
    state.status = PipelineStatus.RUNNING

    try:
        # Build and send prompt
        user_prompt = build_extraction_user_prompt(
            state.raw_clinical_text,
            state.procedure_code,
        )

        result = call_llm(
            system_prompt=EXTRACTION_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.05,  # low temp for extraction accuracy
        )

        # Handle non-JSON fallback
        if isinstance(result, str):
            state.errors.append(f"Extraction returned non-JSON: {result[:200]}")
            state.status = PipelineStatus.NEEDS_REVIEW
            return state

        # Map LLM output to state fields
        state.diagnosis_codes = result.get("diagnosis_codes", [])
        state.extracted_procedure_code = result.get(
            "procedure_code", state.procedure_code
        )

        # Parse prior treatments
        state.prior_treatments = [
            Treatment(
                drug=t.get("drug", ""),
                duration=t.get("duration", ""),
                outcome=t.get("outcome", ""),
                dosage=t.get("dosage", ""),
            )
            for t in result.get("prior_treatments", [])
        ]

        state.symptoms = result.get("symptoms", [])
        state.severity_score = float(result.get("severity_score", 0.0))
        state.patient_demographics = result.get("patient_demographics", {})
        state.lab_values = result.get("lab_values", {})
        state.extraction_confidence = float(result.get("confidence", 0.0))

        # Validation: flag low-confidence extractions
        if state.extraction_confidence < 0.4:
            state.errors.append(
                f"Low extraction confidence ({state.extraction_confidence:.2f}). "
                "Manual review recommended."
            )
            state.status = PipelineStatus.NEEDS_REVIEW
        else:
            state.status = PipelineStatus.RUNNING

    except Exception as e:
        state.errors.append(f"Extraction agent error: {str(e)}")
        state.status = PipelineStatus.FAILED

    return state
