"""
AuthorizeAI — Approval Prediction Model
=========================================
Lightweight classifier (logistic regression / gradient-boosted tree)
for predicting PA approval probability. Includes synthetic training
data generation for the hackathon prototype.
"""

import json
import pickle
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .features_updated import FeatureVector, extract_features
from ..state import AuthorizeState, PredictionResult

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


class ApprovalPredictor:
    """
    Wraps a scikit-learn classifier for PA approval prediction.
    Falls back to a heuristic rule-based estimator if sklearn
    is unavailable or no trained model exists.
    """

    def __init__(self, model_path: Path | None = None):
        self.model = None
        self.scaler = None
        self.model_path = model_path or (MODEL_DIR / "approval_model.pkl")

        if self.model_path.exists():
            self._load_model()

    def _load_model(self) -> None:
        with open(self.model_path, "rb") as f:
            bundle = pickle.load(f)
            self.model = bundle["model"]
            self.scaler = bundle.get("scaler")

    def _save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def predict(self, state: AuthorizeState) -> PredictionResult:
        """
        Predict approval probability from the pipeline state.
        Uses trained model if available, otherwise falls back to heuristic.
        """
        features = extract_features(state)

        if self.model is not None and HAS_SKLEARN:
            return self._predict_ml(features)
        else:
            return self._predict_heuristic(features)

    def _predict_ml(self, features: FeatureVector) -> PredictionResult:
        """Inference using trained sklearn model."""
        X = np.array([features.to_list()])
        if self.scaler:
            X = self.scaler.transform(X)

        prob = self.model.predict_proba(X)[0][1]  # P(approved)

        # Identify top contributing factors via feature importance
        key_factors = self._get_key_factors(features)
        actions = self._get_recommended_actions(features, prob)

        return PredictionResult(
            approval_probability=round(prob, 3),
            risk_tier=self._risk_tier(prob),
            key_factors=key_factors,
            recommended_actions=actions,
        )

    def _predict_heuristic(self, features: FeatureVector) -> PredictionResult:
        """
        Rule-based fallback estimator.
        Weights: 50% coverage score, 20% historical base rate,
                 15% failed treatments signal, 15% severity.
        """
        base_rate = features.procedure_approval_rate * (1 - features.payer_denial_rate)

        prob = (
            0.50 * features.coverage_score
            + 0.20 * base_rate
            + 0.15 * min(features.num_failed_treatments / 3.0, 1.0)
            + 0.15 * features.severity_score
        )
        prob = max(0.05, min(0.98, prob))

        key_factors = self._get_key_factors(features)
        actions = self._get_recommended_actions(features, prob)

        return PredictionResult(
            approval_probability=round(prob, 3),
            risk_tier=self._risk_tier(prob),
            key_factors=key_factors,
            recommended_actions=actions,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "logistic",
    ) -> dict:
        """
        Train the prediction model on feature matrix X and labels y.
        Returns cross-validation metrics.
        """
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required for model training")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if model_type == "gbm":
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42,
            )
        else:
            self.model = LogisticRegression(
                max_iter=1000, random_state=42, C=1.0,
            )

        # Cross-validate
        scores = cross_val_score(
            self.model, X_scaled, y, cv=5, scoring="roc_auc"
        )

        # Fit on full data
        self.model.fit(X_scaled, y)
        self._save_model()

        return {
            "model_type": model_type,
            "cv_auc_mean": round(float(scores.mean()), 4),
            "cv_auc_std": round(float(scores.std()), 4),
            "n_samples": len(y),
            "n_features": X.shape[1],
        }

    @staticmethod
    def _risk_tier(prob: float) -> str:
        if prob >= 0.80:
            return "low"
        elif prob >= 0.50:
            return "medium"
        else:
            return "high"

    @staticmethod
    def _get_key_factors(features: FeatureVector) -> list[str]:
        factors = []
        if features.coverage_score < 0.5:
            factors.append("Low coverage score — multiple criteria unmet")
        if features.num_gaps > 2:
            factors.append(f"{features.num_gaps} documentation gaps identified")
        if features.payer_denial_rate > 0.10:
            factors.append("Payer has above-average denial rate")
        if features.num_failed_treatments >= 2:
            factors.append("Multiple failed prior treatments documented (favorable)")
        if features.severity_score >= 0.7:
            factors.append("High clinical severity (favorable)")
        if features.criteria_not_met_ratio > 0.3:
            factors.append("Significant criteria not met")
        return factors or ["No significant risk factors identified"]

    @staticmethod
    def _get_recommended_actions(
        features: FeatureVector, prob: float
    ) -> list[str]:
        actions = []
        if features.criteria_insufficient_ratio > 0.2:
            actions.append(
                "Obtain additional documentation for criteria marked INSUFFICIENT"
            )
        if features.num_gaps > 0:
            actions.append("Address identified documentation gaps before submission")
        if prob < 0.50:
            actions.append("Consider peer-to-peer review before submission")
        if features.num_failed_treatments == 0:
            actions.append(
                "Document any prior conservative treatments attempted"
            )
        if prob >= 0.80:
            actions.append("Strong case — proceed with submission")
        return actions or ["Review and submit"]


def generate_synthetic_training_data(n_samples: int = 1000) -> tuple:
    """
    Generate synthetic training data for the hackathon prototype.
    Produces realistic feature distributions calibrated against
    CMS/KFF published base rates.

    Returns (X, y) as numpy arrays.
    """
    rng = np.random.default_rng(42)

    X = np.zeros((n_samples, 15))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        coverage = rng.beta(3, 2)             # skew toward higher coverage
        met_ratio = coverage * rng.uniform(0.8, 1.0)
        not_met = rng.uniform(0, 1 - met_ratio)
        insuf = 1 - met_ratio - not_met
        gaps = rng.poisson(2)
        severity = rng.beta(2, 3)
        n_treatments = rng.poisson(2)
        n_failed = min(rng.poisson(1), n_treatments)
        has_symptoms = int(rng.random() > 0.2)
        n_dx = rng.poisson(2) + 1
        payer_denial = rng.choice(
            [0.042, 0.058, 0.077, 0.085, 0.109, 0.119, 0.123, 0.128]
        )
        proc_approval = rng.choice([0.78, 0.80, 0.82, 0.85, 0.88, 0.91])
        age = rng.normal(55, 15)
        is_female = int(rng.random() > 0.5)
        confidence = rng.beta(5, 2)

        X[i] = [
            coverage, met_ratio, not_met, insuf, gaps, severity,
            n_treatments, n_failed, has_symptoms, n_dx,
            payer_denial, proc_approval, age, is_female, confidence,
        ]

        # Generate label: approval probability based on features
        logit = (
            2.0 * coverage
            + 1.5 * (n_failed / 3.0)
            + 1.0 * severity
            - 2.0 * not_met
            - 0.5 * gaps / 3.0
            + 1.0 * proc_approval
            - 1.5 * payer_denial
            - 0.3
        )
        prob = 1 / (1 + np.exp(-logit))
        y[i] = int(rng.random() < prob)

    return X, y
