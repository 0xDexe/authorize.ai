"""
AuthorizeAI — Data Setup & Evaluation CLI
===========================================
One-stop script for data initialization, MIMIC validation,
base rate seeding, training data generation, and pipeline evaluation.

Usage:
    python setup_data.py validate          # check MIMIC directory
    python setup_data.py seed              # seed public base rates
    python setup_data.py train [--n 500]   # generate training data from MIMIC
    python setup_data.py eval [--n 10]     # run pipeline eval on MIMIC cases
    python setup_data.py summary           # show base rate DB contents
"""

import argparse
import json
import sys
from pathlib import Path

from src.data.mimic_loader import (
    validate_mimic_directory, load_all_cases,
    filter_cases_with_imaging, filter_cases_by_diagnosis,
    DEFAULT_MIMIC_DIR,
)
from src.data.public_rates import (
    init_base_rate_db, seed_kff_base_rates,
    load_cms_puf, DB_PATH,
)
from src.data.training_gen import generate_training_data_from_mimic
from src.data.evaluation import (
    evaluate_pipeline_on_cases, summarize_eval_results,
)
from src.models.predictor import ApprovalPredictor
from src.models.features import FeatureVector


def cmd_validate(args):
    """Check MIMIC-IV directory structure."""
    mimic_dir = Path(args.mimic_dir)
    print(f"Validating MIMIC-IV directory: {mimic_dir}\n")

    report = validate_mimic_directory(mimic_dir)

    print("Found:")
    for f in report["found"]:
        print(f"  ✅ {f}")

    if report["missing"]:
        print("\nMissing (required):")
        for f in report["missing"]:
            print(f"  ❌ {f}")

    if report["optional_missing"]:
        print("\nMissing (optional):")
        for f in report["optional_missing"]:
            print(f"  ⚠️  {f}")

    if report["ready"]:
        print("\n✅ MIMIC-IV directory is ready for use.")
    else:
        print("\n❌ Missing required tables. Download from PhysioNet:")
        print("   hosp module: https://physionet.org/content/mimiciv/2.2/")
        print("   note module: https://physionet.org/content/mimic-iv-note/2.2/")


def cmd_seed(args):
    """Seed the base rates database with public data."""
    print("Initializing base rates database...")
    conn = init_base_rate_db()

    print("Seeding KFF/CMS/AMA base rates...")
    n = seed_kff_base_rates(conn)
    print(f"  Loaded {n} base rate records")

    if args.cms_puf:
        print(f"\nLoading CMS PUF from: {args.cms_puf}")
        n = load_cms_puf(args.cms_puf, conn)
        print(f"  Loaded {n} contract-level records")

    print(f"\n✅ Base rates database: {DB_PATH}")


def cmd_train(args):
    """Generate training data from MIMIC and train the prediction model."""
    mimic_dir = Path(args.mimic_dir)
    n = args.n

    report = validate_mimic_directory(mimic_dir)
    if not report["ready"]:
        print("❌ MIMIC directory not ready. Run 'python setup_data.py validate' first.")
        sys.exit(1)

    print(f"Loading {n} MIMIC cases...")
    cases = load_all_cases(mimic_dir, limit=n)

    if not cases:
        print("❌ No cases loaded. Check MIMIC directory.")
        sys.exit(1)

    print(f"\nGenerating training features from {len(cases)} cases...")
    X, y = generate_training_data_from_mimic(
        cases, use_heuristic_labels=True,
    )

    print(f"  Feature matrix: {X.shape}")
    print(f"  Positive rate: {y.mean():.1%}")
    print(f"  Features: {FeatureVector.feature_names()}")

    print(f"\nTraining {args.model} model...")
    predictor = ApprovalPredictor()
    metrics = predictor.train(X, y, model_type=args.model)

    print(f"\n  Model:     {metrics['model_type']}")
    print(f"  CV AUC:    {metrics['cv_auc_mean']:.4f} (±{metrics['cv_auc_std']:.4f})")
    print(f"  Samples:   {metrics['n_samples']}")
    print(f"  Features:  {metrics['n_features']}")
    print(f"\n✅ Model saved to: data/models/approval_model.pkl")


def cmd_eval(args):
    """Run pipeline evaluation on MIMIC cases."""
    from src.pipeline import run_pipeline

    mimic_dir = Path(args.mimic_dir)
    n = args.n

    report = validate_mimic_directory(mimic_dir)
    if not report["ready"]:
        print("❌ MIMIC directory not ready.")
        sys.exit(1)

    print(f"Loading {n} MIMIC cases...")
    cases = load_all_cases(mimic_dir, limit=n)

    if args.imaging_only:
        cases = filter_cases_with_imaging(cases)
        print(f"  Filtered to {len(cases)} cases with imaging")

    print(f"\nRunning pipeline evaluation on {len(cases)} cases...")
    results = evaluate_pipeline_on_cases(
        cases[:n], run_pipeline, verbose=True,
    )

    summary = summarize_eval_results(results)
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v}")


def cmd_summary(args):
    """Show contents of the base rates database."""
    import sqlite3

    if not DB_PATH.exists():
        print("❌ Base rates database not found. Run 'python setup_data.py seed' first.")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    print("=== Payer Denial Rates ===")
    rows = conn.execute(
        "SELECT payer_id, payer_name, denial_rate, overturn_rate, source "
        "FROM payer_denial_rates ORDER BY denial_rate DESC"
    ).fetchall()
    for r in rows:
        print(
            f"  {r['payer_id']:12s} | denial: {r['denial_rate']:.1%} | "
            f"overturn: {r['overturn_rate']:.1%} | source: {r['source']}"
        )

    print("\n=== Procedure Approval Rates ===")
    rows = conn.execute(
        "SELECT procedure_category, approval_rate, source "
        "FROM procedure_approval_rates ORDER BY approval_rate"
    ).fetchall()
    for r in rows:
        print(
            f"  {r['procedure_category']:22s} | approval: {r['approval_rate']:.1%} | "
            f"source: {r['source']}"
        )


def main():
    parser = argparse.ArgumentParser(description="AuthorizeAI data setup")
    parser.add_argument(
        "--mimic-dir", default=str(DEFAULT_MIMIC_DIR),
        help="Path to MIMIC-IV root directory",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("validate", help="Check MIMIC directory")

    seed_p = sub.add_parser("seed", help="Seed base rates database")
    seed_p.add_argument("--cms-puf", help="Path to CMS PUF CSV file")

    train_p = sub.add_parser("train", help="Generate training data from MIMIC")
    train_p.add_argument("--n", type=int, default=500, help="Number of cases")
    train_p.add_argument("--model", choices=["logistic", "gbm"], default="logistic")

    eval_p = sub.add_parser("eval", help="Run pipeline evaluation")
    eval_p.add_argument("--n", type=int, default=10, help="Number of cases")
    eval_p.add_argument("--imaging-only", action="store_true")

    sub.add_parser("summary", help="Show base rates database")

    args = parser.parse_args()

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "seed":
        cmd_seed(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
