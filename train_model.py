"""
AuthorizeAI — Model Training Script
=====================================
Generates synthetic training data and trains the approval prediction model.
Run: python train_model.py [--model logistic|gbm] [--samples 2000]
"""

import argparse
import sys

import numpy as np

from src.models.predictor import ApprovalPredictor, generate_synthetic_training_data
from src.models.features import FeatureVector


def main():
    parser = argparse.ArgumentParser(description="Train AuthorizeAI prediction model")
    parser.add_argument("--model", choices=["logistic", "gbm"], default="logistic")
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    print(f"Generating {args.samples} synthetic training samples...")
    X, y = generate_synthetic_training_data(n_samples=args.samples)

    print(f"  Positive rate: {y.mean():.1%}")
    print(f"  Feature count: {X.shape[1]}")
    print(f"  Features: {FeatureVector.feature_names()}")

    print(f"\nTraining {args.model} model with 5-fold cross-validation...")
    predictor = ApprovalPredictor()
    metrics = predictor.train(X, y, model_type=args.model)

    print(f"\n  Model type:  {metrics['model_type']}")
    print(f"  CV AUC:      {metrics['cv_auc_mean']:.4f} (±{metrics['cv_auc_std']:.4f})")
    print(f"  Samples:     {metrics['n_samples']}")
    print(f"  Features:    {metrics['n_features']}")
    print(f"\n  Model saved to: data/models/approval_model.pkl")


if __name__ == "__main__":
    main()
