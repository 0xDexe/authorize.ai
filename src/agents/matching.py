"""
AuthorizeAI — Agent 2: Payer Criteria Matching
================================================
Retrieves relevant coverage policy via BM25 search, then uses the LLM
to evaluate each coverage criterion against the extracted clinical facts.
Outputs per-criterion MET/NOT_MET/INSUFFICIENT status.
"""

import json
from dataclasses import asdict

from ..state import (
    AuthorizeState, CriterionResult, CriterionStatus, PipelineStatus,
)
from ..prompts.templates import MATCHING_SYSTEM, build_matching_user_prompt
from ..retrieval.searcher import (
    search_policies, get_policy_logic_tree,
    get_all_criteria_for_policy, build_context_block,
)
from ..utils.llm_client import call_llm


def matching_agent(state: AuthorizeState) -> AuthorizeState:
    """
    LangGraph node function for payer criteria matching.
    Reads: extracted clinical facts from Agent 1, payer_id, procedure_code
    Writes: criteria_results, overall_coverage_score, gaps,
            policy_reference, matched_policy_text
    """
    state.current_agent = "matching"

    try:
        # ── Step 1: Retrieve relevant policy chunks via BM25 ──
        clinical_keywords = _build_search_keywords(state)

        search_results = search_policies(
            procedure_code=state.extracted_procedure_code or state.procedure_code,
            payer_id=state.payer_id,
            clinical_keywords=clinical_keywords,
            top_k=8,
        )

        if not search_results:
            state.errors.append(
                f"No policy found for payer={state.payer_id}, "
                f"procedure={state.procedure_code}"
            )
            state.status = PipelineStatus.NEEDS_REVIEW
            return state

        # ── Step 2: Build policy context ──
        policy_id = search_results[0].policy_id
        policy_context = build_context_block(search_results)
        state.matched_policy_text = policy_context
        state.policy_reference = policy_id

        # Check for pre-structured logic tree
        logic_tree = get_policy_logic_tree(policy_id)

        # If logic tree exists, also pull all criteria for completeness
        if logic_tree:
            all_criteria = get_all_criteria_for_policy(policy_id)
            if all_criteria:
                policy_context = build_context_block(all_criteria)

        # ── Step 3: Build clinical facts dict for the prompt ──
        clinical_facts = _build_clinical_facts_dict(state)

        # ── Step 4: LLM evaluation ──
        user_prompt = build_matching_user_prompt(
            clinical_facts=clinical_facts,
            policy_context=policy_context,
            policy_id=policy_id,
            logic_tree=logic_tree,
        )

        result = call_llm(
            system_prompt=MATCHING_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        if isinstance(result, str):
            state.errors.append(f"Matching returned non-JSON: {result[:200]}")
            state.status = PipelineStatus.NEEDS_REVIEW
            return state

        # ── Step 5: Parse results into state ──
        state.criteria_results = [
            CriterionResult(
                criterion_id=cr.get("criterion_id", f"C{i}"),
                criterion_text=cr.get("criterion_text", ""),
                status=_parse_status(cr.get("status", "INSUFFICIENT_EVIDENCE")),
                evidence=cr.get("evidence", ""),
                confidence=float(cr.get("confidence", 0.0)),
            )
            for i, cr in enumerate(result.get("criteria_results", []))
        ]

        state.overall_coverage_score = float(
            result.get("overall_coverage_score", 0.0)
        )
        state.gaps = result.get("gaps", [])
        state.policy_reference = result.get(
            "policy_reference", policy_id
        )

    except Exception as e:
        state.errors.append(f"Matching agent error: {str(e)}")
        state.status = PipelineStatus.FAILED

    return state


def _build_search_keywords(state: AuthorizeState) -> list[str]:
    """Build search keywords from extracted clinical data."""
    keywords = []
    keywords.extend(state.diagnosis_codes[:5])
    keywords.extend(state.symptoms[:5])
    for t in state.prior_treatments[:3]:
        if t.drug:
            keywords.append(t.drug)
    return [k for k in keywords if k]


def _build_clinical_facts_dict(state: AuthorizeState) -> dict:
    """Assemble clinical facts into a dict for the LLM prompt."""
    return {
        "diagnosis_codes": state.diagnosis_codes,
        "procedure_code": state.extracted_procedure_code or state.procedure_code,
        "prior_treatments": [
            {"drug": t.drug, "duration": t.duration,
             "outcome": t.outcome, "dosage": t.dosage}
            for t in state.prior_treatments
        ],
        "symptoms": state.symptoms,
        "severity_score": state.severity_score,
        "patient_demographics": state.patient_demographics,
        "lab_values": state.lab_values,
    }


def _parse_status(raw: str) -> CriterionStatus:
    """Parse a status string into the enum, with fuzzy matching."""
    upper = raw.upper().replace(" ", "_")
    if "MET" == upper or upper == "CRITERION_MET":
        return CriterionStatus.MET
    elif "NOT" in upper:
        return CriterionStatus.NOT_MET
    else:
        return CriterionStatus.INSUFFICIENT
