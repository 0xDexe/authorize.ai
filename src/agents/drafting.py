"""
AuthorizeAI — Agent 4: Submission Drafting
============================================
Generates a complete prior authorization request letter (or appeal letter)
using all upstream pipeline data. The letter cites specific clinical
evidence and addresses each coverage criterion directly.
"""

from dataclasses import asdict

from ..state import AuthorizeState, PipelineStatus
from ..prompts.templates import DRAFTING_SYSTEM, build_drafting_user_prompt
from ..utils.llm_client import call_llm


def drafting_agent(state: AuthorizeState) -> AuthorizeState:
    """
    LangGraph node function for PA letter generation.
    Reads: all upstream state fields
    Writes: draft_letter, cited_evidence
    """
    state.current_agent = "drafting"

    try:
        # Build structured inputs for the prompt
        clinical_facts = {
            "diagnosis_codes": state.diagnosis_codes,
            "procedure_code": state.extracted_procedure_code or state.procedure_code,
            "prior_treatments": [
                {"drug": t.drug, "duration": t.duration,
                 "outcome": t.outcome, "dosage": t.dosage}
                for t in state.prior_treatments
            ],
            "symptoms": state.symptoms,
            "severity_score": state.severity_score,
            "lab_values": state.lab_values,
        }

        criteria_results = [
            {
                "criterion_id": cr.criterion_id,
                "criterion_text": cr.criterion_text,
                "status": cr.status.value,
                "evidence": cr.evidence,
            }
            for cr in state.criteria_results
        ]

        prediction_dict = {
            "approval_probability": state.prediction.approval_probability,
            "risk_tier": state.prediction.risk_tier,
            "key_factors": state.prediction.key_factors,
            "recommended_actions": state.prediction.recommended_actions,
        }

        user_prompt = build_drafting_user_prompt(
            clinical_facts=clinical_facts,
            criteria_results=criteria_results,
            prediction=prediction_dict,
            patient_demographics=state.patient_demographics,
            payer_id=state.payer_id,
            procedure_code=state.extracted_procedure_code or state.procedure_code,
            policy_reference=state.policy_reference,
            letter_type=state.letter_type,
        )

        result = call_llm(
            system_prompt=DRAFTING_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.3,  # slightly higher for natural language
            max_tokens=4096,
        )

        # Drafting returns plain text, not JSON
        if isinstance(result, dict):
            # LLM returned JSON anyway — extract letter field or stringify
            state.draft_letter = result.get("letter", str(result))
        else:
            state.draft_letter = result

        # Extract cited evidence references from the letter
        state.cited_evidence = _extract_citations(
            state.draft_letter, clinical_facts
        )

        state.status = PipelineStatus.SUCCESS

    except Exception as e:
        state.errors.append(f"Drafting agent error: {str(e)}")
        state.status = PipelineStatus.FAILED

    return state


def _extract_citations(letter: str, clinical_facts: dict) -> list[str]:
    """
    Identify which clinical facts are actually cited in the draft letter.
    Useful for audit trail and physician review.
    """
    citations = []

    # Check diagnosis codes
    for code in clinical_facts.get("diagnosis_codes", []):
        if code in letter:
            citations.append(f"Diagnosis: {code}")

    # Check procedure code
    proc = clinical_facts.get("procedure_code", "")
    if proc and proc in letter:
        citations.append(f"Procedure: {proc}")

    # Check treatments
    for t in clinical_facts.get("prior_treatments", []):
        if t.get("drug", "") and t["drug"].lower() in letter.lower():
            citations.append(f"Treatment: {t['drug']}")

    # Check symptoms
    for s in clinical_facts.get("symptoms", []):
        if s.lower() in letter.lower():
            citations.append(f"Symptom: {s}")

    return citations
