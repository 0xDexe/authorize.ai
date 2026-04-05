"""Tests for state schema, feature engineering, and prediction model."""

import numpy as np
import pytest

from src.state import (
    AuthorizeState, Treatment, CriterionResult,
    CriterionStatus, PipelineStatus, PredictionResult,
)
from src.models.features import FeatureVector, extract_features, CPT_TO_CATEGORY
from src.models.predictor import (
    ApprovalPredictor, generate_synthetic_training_data,
)


# ── State schema ───────────────────────────────────────────────────────────

class TestAuthorizeState:
    def test_default_values(self):
        state = AuthorizeState()
        assert state.status == PipelineStatus.PENDING
        assert state.raw_clinical_text == ""
        assert state.diagnosis_codes == []
        assert state.criteria_results == []
        assert state.errors == []

    def test_treatment_fields(self):
        t = Treatment(drug="Naproxen", duration="6 weeks", outcome="failed")
        assert t.drug == "Naproxen"
        assert t.outcome == "failed"

    def test_criterion_result_defaults(self):
        cr = CriterionResult()
        assert cr.status == CriterionStatus.INSUFFICIENT
        assert cr.confidence == 0.0

    def test_prediction_result_defaults(self):
        pr = PredictionResult()
        assert pr.approval_probability == 0.0
        assert pr.risk_tier == ""
        assert pr.key_factors == []

    def test_state_mutation(self):
        state = AuthorizeState(raw_clinical_text="test note")
        state.diagnosis_codes = ["M54.4", "M51.16"]
        state.status = PipelineStatus.RUNNING
        assert len(state.diagnosis_codes) == 2
        assert state.status == PipelineStatus.RUNNING


# ── Feature engineering ────────────────────────────────────────────────────

class TestFeatureEngineering:
    def _make_state(self, **overrides) -> AuthorizeState:
        defaults = dict(
            payer_id="UHC",
            procedure_code="72148",
            extracted_procedure_code="72148",
            diagnosis_codes=["M54.4", "M51.16"],
            prior_treatments=[
                Treatment(drug="Naproxen", duration="6 weeks", outcome="failed"),
                Treatment(drug="PT", duration="8 weeks", outcome="failed"),
            ],
            symptoms=["radiculopathy", "numbness"],
            severity_score=0.75,
            patient_demographics={"age": 52, "sex": "F"},
            extraction_confidence=0.85,
            overall_coverage_score=0.7,
            criteria_results=[
                CriterionResult(status=CriterionStatus.MET),
                CriterionResult(status=CriterionStatus.MET),
                CriterionResult(status=CriterionStatus.NOT_MET),
            ],
            gaps=["Missing PT duration documentation"],
        )
        defaults.update(overrides)
        return AuthorizeState(**defaults)

    def test_extract_features_returns_feature_vector(self):
        state = self._make_state()
        fv = extract_features(state)
        assert isinstance(fv, FeatureVector)

    def test_feature_vector_to_list_length(self):
        state = self._make_state()
        fv = extract_features(state)
        assert len(fv.to_list()) == 15
        assert len(FeatureVector.feature_names()) == 15

    def test_criteria_ratios_sum_to_one(self):
        state = self._make_state()
        fv = extract_features(state)
        total = fv.criteria_met_ratio + fv.criteria_not_met_ratio + fv.criteria_insufficient_ratio
        assert abs(total - 1.0) < 0.01

    def test_failed_treatments_counted(self):
        state = self._make_state()
        fv = extract_features(state)
        assert fv.num_failed_treatments == 2
        assert fv.num_prior_treatments == 2

    def test_demographics_extracted(self):
        state = self._make_state()
        fv = extract_features(state)
        assert fv.patient_age == 52.0
        assert fv.is_female == 1

    def test_unknown_payer_uses_default(self):
        state = self._make_state(payer_id="UNKNOWN_PAYER")
        fv = extract_features(state)
        assert fv.payer_denial_rate > 0

    def test_cpt_category_mapping(self):
        assert CPT_TO_CATEGORY["72148"] == "MRI"
        assert CPT_TO_CATEGORY["74177"] == "CT"
        assert CPT_TO_CATEGORY["J0717"] == "BRAND_DRUG"

    def test_empty_state_doesnt_crash(self):
        state = AuthorizeState()
        fv = extract_features(state)
        assert isinstance(fv, FeatureVector)
        assert fv.coverage_score == 0.0


# ── Predictor ──────────────────────────────────────────────────────────────

class TestPredictor:
    def test_heuristic_fallback(self):
        predictor = ApprovalPredictor(model_path=None)
        predictor.model = None  # force heuristic
        state = AuthorizeState(
            overall_coverage_score=0.8,
            severity_score=0.7,
            prior_treatments=[
                Treatment(drug="X", outcome="failed"),
                Treatment(drug="Y", outcome="failed"),
            ],
            symptoms=["pain"],
            payer_id="UHC",
            procedure_code="72148",
        )
        result = predictor.predict(state)
        assert 0.0 < result.approval_probability < 1.0
        assert result.risk_tier in ("low", "medium", "high")
        assert isinstance(result.key_factors, list)
        assert isinstance(result.recommended_actions, list)

    def test_risk_tier_thresholds(self):
        assert ApprovalPredictor._risk_tier(0.90) == "low"
        assert ApprovalPredictor._risk_tier(0.60) == "medium"
        assert ApprovalPredictor._risk_tier(0.30) == "high"

    def test_synthetic_data_shape(self):
        X, y = generate_synthetic_training_data(n_samples=100)
        assert X.shape == (100, 15)
        assert y.shape == (100,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_synthetic_data_has_variance(self):
        X, y = generate_synthetic_training_data(n_samples=500)
        positive_rate = y.mean()
        assert 0.2 < positive_rate < 0.9  # not degenerate

    def test_train_and_predict(self):
        X, y = generate_synthetic_training_data(n_samples=200)
        predictor = ApprovalPredictor(model_path=None)
        metrics = predictor.train(X, y, model_type="logistic")
        assert metrics["cv_auc_mean"] > 0.5  # better than random
        assert metrics["n_features"] == 15

        state = AuthorizeState(
            overall_coverage_score=0.9,
            severity_score=0.8,
            payer_id="UHC",
            procedure_code="72148",
            prior_treatments=[Treatment(drug="X", outcome="failed")],
            symptoms=["pain"],
            criteria_results=[CriterionResult(status=CriterionStatus.MET)],
        )
        result = predictor.predict(state)
        assert 0.0 <= result.approval_probability <= 1.0
