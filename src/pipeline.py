"""
AuthorizeAI — LangGraph Pipeline
==================================
Orchestrates the four-agent pipeline as a LangGraph StateGraph.
Includes conditional edges for error handling and review routing.

Pipeline flow:
  START → extract → should_continue → match → predict → draft → END
                  ↘ (if failed/needs_review) → END
"""

from langgraph.graph import StateGraph, START, END
from sklearn import pipeline

from .state import AuthorizeState, PipelineStatus
from .agents.extraction import extraction_agent
from .agents.matching import matching_agent
from .agents.prediction import prediction_agent
from .agents.drafting import drafting_agent


# ── Conditional edge functions ──────────────────────────────────────────

def should_continue_after_extraction(state: AuthorizeState) -> str:
    """
    Gate after Agent 1. Routes to matching if extraction succeeded,
    or to END if extraction failed or needs manual review.
    """
    if state.status == PipelineStatus.FAILED:
        return "end"
    if state.status == PipelineStatus.NEEDS_REVIEW:
        # Low confidence extraction — still proceed but flag it
        if state.extraction_confidence > 0.2:
            return "match"
        return "end"
    return "match"


def should_continue_after_matching(state: AuthorizeState) -> str:
    """
    Gate after Agent 2. Routes to prediction if matching succeeded.
    """
    if state.status == PipelineStatus.FAILED:
        return "end"
    return "predict"


def should_continue_after_prediction(state: AuthorizeState) -> str:
    """
    Gate after Agent 3. Always proceeds to drafting — prediction
    failure is non-fatal (uses heuristic fallback).
    """
    return "draft"


# ── Pipeline builder ────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    Construct and compile the AuthorizeAI LangGraph pipeline.
    Returns a compiled graph ready for invocation.
    """
    graph = StateGraph(AuthorizeState)

    # Add agent nodes
    graph.add_node("extract", extraction_agent)
    graph.add_node("match", matching_agent)
    graph.add_node("predict", prediction_agent)
    graph.add_node("draft", drafting_agent)

    # Entry point
    graph.add_edge(START, "extract")

    # Conditional routing after extraction
    graph.add_conditional_edges(
        "extract",
        should_continue_after_extraction,
        {"match": "match", "end": END},
    )

    # Conditional routing after matching
    graph.add_conditional_edges(
        "match",
        should_continue_after_matching,
        {"predict": "predict", "end": END},
    )

    # Prediction always flows to drafting
    graph.add_conditional_edges(
        "predict",
        should_continue_after_prediction,
        {"draft": "draft"},
    )

    # Drafting is the terminal node
    graph.add_edge("draft", END)

    return graph.compile()


def run_pipeline(
    clinical_text: str,
    payer_id: str,
    procedure_code: str,
    patient_id: str = "",
    letter_type: str = "initial",
) -> AuthorizeState:
    """
    Convenience function to run the full pipeline end-to-end.

    Args:
        clinical_text: Raw clinical notes (text or extracted from PDF)
        payer_id: Payer identifier (e.g. "UHC", "AETNA")
        procedure_code: CPT code for the requested procedure
        patient_id: Optional patient identifier
        letter_type: "initial" or "appeal"

    Returns:
        Completed AuthorizeState with all agent outputs populated.
    """
    pipeline = build_pipeline()

    initial_state = AuthorizeState(
        raw_clinical_text=clinical_text,
        payer_id=payer_id,
        procedure_code=procedure_code,
        patient_id=patient_id,
        letter_type=letter_type,
    )

    final_state = pipeline.invoke(initial_state)
    if isinstance(final_state, dict):
        final_state = AuthorizeState(**final_state)
    return final_state
