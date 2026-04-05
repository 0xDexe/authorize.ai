"""
AuthorizeAI — Agent 3: Approval Prediction
============================================
Predicts PA approval probability using a lightweight ML model.
Combines real-time criteria matching output with historical
base rate data from CMS/KFF public sources.
"""

from ..state import AuthorizeState, PipelineStatus
from ..models.predictor import ApprovalPredictor


# Module-level singleton to avoid reloading model per invocation
_predictor: ApprovalPredictor | None = None


def _get_predictor() -> ApprovalPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ApprovalPredictor()
    return _predictor


def prediction_agent(state: AuthorizeState) -> AuthorizeState:
    """
    LangGraph node function for approval prediction.
    Reads: all upstream state (clinical facts + criteria results)
    Writes: prediction (PredictionResult)
    """
    state.current_agent = "prediction"

    try:
        predictor = _get_predictor()
        state.prediction = predictor.predict(state)

    except Exception as e:
        state.errors.append(f"Prediction agent error: {str(e)}")
        # Non-fatal: prediction failure shouldn't block drafting
        # Fall back to a conservative estimate
        from ..state import PredictionResult
        state.prediction = PredictionResult(
            approval_probability=state.overall_coverage_score * 0.8,
            risk_tier="medium",
            key_factors=["Prediction model unavailable — using coverage score"],
            recommended_actions=["Review manually before submission"],
        )

    return state
