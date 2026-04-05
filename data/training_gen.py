"""
AuthorizeAI — Training Data Generator
=======================================
Generates labeled training data for Agent 3's prediction model by
combining MIMIC-IV clinical cases with public CMS/KFF base rates.

The synthetic label strategy:
  1. Run Agent 1 + Agent 2 on each MIMIC case to get coverage_score
  2. Combine coverage_score with public payer denial rates
  3. Generate probabilistic labels calibrated against CMS published rates
  4. Output a feature matrix + label vector for sklearn training
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from .mimic_loader import MIMICCase, case_to_pipeline_input
from .public_rates import (
    get_payer_denial_rate, get_procedure_approval_rate,
    init_base_rate_db, seed_kff_base_rates,
)
from ..models.features import (
    FeatureVector, CPT_TO_CATEGORY,
    PAYER_DENIAL_RATES, PROCEDURE_CATEGORY_APPROVAL_RATES,
)


def generate_training_data_from_mimic(
    cases: list[MIMICCase],
    run_agents_1_2_fn=None,
    use_heuristic_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate training data from real MIMIC cases.

    Two modes:
      1. Full agent mode: run Agents 1+2 on each case to get real
         coverage scores (expensive — requires LLM calls)
      2. Heuristic mode: estimate features from MIMIC structured data
         without LLM calls (fast, free, good for prototyping)

    Args:
        cases: List of assembled MIMICCase objects
        run_agents_1_2_fn: Optional function that runs the extraction +
            matching pipeline and returns the state. If None, uses
            heuristic feature estimation.
        use_heuristic_labels: If True, generate labels from base rates
            rather than running the full pipeline.

    Returns:
        (X, y) — feature matrix and binary labels
    """
    features_list = []
    labels = []
    rng = np.random.default_rng(42)

    # Ensure base rates are seeded
    conn = init_base_rate_db()
    seed_kff_base_rates(conn)

    for case in cases:
        if not case.primary_discharge_text:
            continue

        if use_heuristic_labels or run_agents_1_2_fn is None:
            feat, label = _heuristic_features_and_label(case, conn, rng)
        else:
            feat, label = _agent_features_and_label(
                case, run_agents_1_2_fn, conn, rng
            )

        if feat is not None:
            features_list.append(feat)
            labels.append(label)

    if not features_list:
        return np.array([]), np.array([])

    X = np.array(features_list)
    y = np.array(labels, dtype=int)
    return X, y


def _heuristic_features_and_label(
    case: MIMICCase,
    conn,
    rng: np.random.Generator,
) -> tuple[list[float] | None, int]:
    """
    Estimate features directly from MIMIC structured data without LLM.

    Coverage score is approximated from diagnosis count and medication
    history — patients with more documented diagnoses and prior treatments
    tend to have better-supported PA requests.
    """
    n_dx = len(case.diagnoses)
    n_drugs = len(case.drug_list)
    n_procedures = len(case.procedures)
    n_radiology = len(case.radiology_reports)

    # Heuristic coverage score: more documentation = higher coverage
    doc_richness = min(
        (n_dx * 0.15 + n_drugs * 0.1 + n_procedures * 0.2 + n_radiology * 0.15),
        1.0,
    )
    coverage_score = 0.3 + 0.7 * doc_richness  # floor at 0.3

    # Estimate ratios
    met_ratio = coverage_score * rng.uniform(0.8, 1.0)
    not_met = rng.uniform(0, max(0.01, 1 - met_ratio))
    insuf = max(0, 1 - met_ratio - not_met)
    gaps = max(0, int(3 * (1 - coverage_score)))

    # Severity from length/complexity of discharge note
    note_len = len(case.primary_discharge_text)
    severity = min(note_len / 10000.0, 1.0)  # longer notes = sicker patients

    # Failed treatments heuristic: count distinct drug classes
    n_failed = min(int(n_drugs * 0.3), 5)  # assume ~30% were unsuccessful

    has_symptoms = 1  # discharge notes always describe symptoms
    extraction_confidence = 0.75  # assumed for heuristic mode

    # Payer rates
    payer_id = case.payer_proxy
    payer_denial = get_payer_denial_rate(payer_id, conn)

    # Procedure category — check radiology CPTs or default
    proc_category = "MRI"  # default for hackathon scope
    for cpt in case.radiology_cpt_codes:
        if cpt in CPT_TO_CATEGORY:
            proc_category = CPT_TO_CATEGORY[cpt]
            break
    proc_approval = get_procedure_approval_rate(proc_category, conn)

    # Demographics
    age = float(case.patient.anchor_age) if case.patient else 55.0
    is_female = 1 if (case.patient and case.patient.gender == "F") else 0

    features = [
        coverage_score, met_ratio, not_met, insuf,
        float(gaps), severity, float(n_drugs), float(n_failed),
        float(has_symptoms), float(n_dx), payer_denial, proc_approval,
        age, float(is_female), extraction_confidence,
    ]

    # Generate label calibrated to base rates
    logit = (
        2.0 * coverage_score
        + 1.5 * (n_failed / 3.0)
        + 1.0 * severity
        - 2.0 * not_met
        - 0.5 * gaps / 3.0
        + 1.0 * proc_approval
        - 1.5 * payer_denial
        - 0.3
    )
    prob = 1 / (1 + np.exp(-logit))
    label = int(rng.random() < prob)

    return features, label


def _agent_features_and_label(
    case: MIMICCase,
    run_agents_1_2_fn,
    conn,
    rng: np.random.Generator,
) -> tuple[list[float] | None, int]:
    """
    Run Agents 1+2 on the case to get real extraction + matching features.
    More accurate but requires LLM API calls (costs ~$0.01-0.05 per case).
    """
    from ..models.features import extract_features
    from ..state import AuthorizeState

    try:
        inputs = case_to_pipeline_input(case)
        state = run_agents_1_2_fn(**inputs)

        feat_vec = extract_features(state)
        features = feat_vec.to_list()

        # Label from coverage score + base rates
        logit = (
            2.0 * feat_vec.coverage_score
            + 1.5 * (feat_vec.num_failed_treatments / 3.0)
            + 1.0 * feat_vec.severity_score
            - 2.0 * feat_vec.criteria_not_met_ratio
            + 1.0 * feat_vec.procedure_approval_rate
            - 1.5 * feat_vec.payer_denial_rate
            - 0.3
        )
        prob = 1 / (1 + np.exp(-logit))
        label = int(rng.random() < prob)

        return features, label

    except Exception as e:
        print(f"  Agent feature extraction failed for {case.hadm_id}: {e}")
        return None, 0
