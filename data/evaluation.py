"""
AuthorizeAI — Evaluation Harness
==================================
Uses MIMIC-IV structured data (ICD codes, prescriptions) as ground truth
to evaluate Agent 1 extraction accuracy and end-to-end pipeline quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .mimic_loader import MIMICCase, case_to_pipeline_input, case_to_ground_truth


@dataclass
class ExtractionEvalResult:
    """Per-case evaluation of Agent 1 extraction vs MIMIC ground truth."""
    hadm_id: str
    # Diagnosis code matching
    true_dx_codes: list[str] = field(default_factory=list)
    extracted_dx_codes: list[str] = field(default_factory=list)
    dx_precision: float = 0.0
    dx_recall: float = 0.0
    dx_f1: float = 0.0
    # Drug matching
    true_drugs: list[str] = field(default_factory=list)
    extracted_drugs: list[str] = field(default_factory=list)
    drug_recall: float = 0.0
    # Symptom extraction (no ground truth — count only)
    n_symptoms_extracted: int = 0
    # Overall
    extraction_confidence: float = 0.0


@dataclass
class PipelineEvalResult:
    """End-to-end evaluation for a single case."""
    hadm_id: str
    extraction_eval: ExtractionEvalResult | None = None
    coverage_score: float = 0.0
    approval_probability: float = 0.0
    risk_tier: str = ""
    letter_generated: bool = False
    letter_length: int = 0
    pipeline_status: str = ""
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _icd_prefix_match(code_a: str, code_b: str) -> bool:
    """
    Check if two ICD codes match via prefix matching.
    Handles cases where extracted code may be shorter (M54 vs M544)
    or longer (M5416 vs M54) than the true code.
    """
    return code_a.startswith(code_b) or code_b.startswith(code_a)


def evaluate_extraction(
    case: MIMICCase,
    extracted_state: dict,
) -> ExtractionEvalResult:
    """
    Compare Agent 1's extraction against MIMIC structured ground truth.

    Uses fuzzy matching for diagnosis codes (prefix matching) and
    case-insensitive substring matching for drug names.
    """
    ground = case_to_ground_truth(case)
    result = ExtractionEvalResult(hadm_id=case.hadm_id)

    # ── Diagnosis code evaluation ──
    true_codes = set(_normalize_icd(c) for c in ground["diagnosis_codes"])
    extracted_codes = set(
        _normalize_icd(c) for c in extracted_state.get("diagnosis_codes", [])
    )
    result.true_dx_codes = list(true_codes)
    result.extracted_dx_codes = list(extracted_codes)

    if extracted_codes and true_codes:
        # Precision: what fraction of extracted codes match a true code
        precision_matches = sum(
            1 for ec in extracted_codes
            if any(_icd_prefix_match(ec, tc) for tc in true_codes)
        )
        # Recall: what fraction of true codes are matched by an extracted code
        recall_matches = sum(
            1 for tc in true_codes
            if any(_icd_prefix_match(ec, tc) for ec in extracted_codes)
        )
        result.dx_precision = precision_matches / len(extracted_codes)
        result.dx_recall = recall_matches / len(true_codes)
        if result.dx_precision + result.dx_recall > 0:
            result.dx_f1 = (
                2 * result.dx_precision * result.dx_recall
                / (result.dx_precision + result.dx_recall)
            )
    elif extracted_codes and not true_codes:
        # No ground truth codes — can't compute meaningful metrics
        result.dx_precision = 0.0
        result.dx_recall = 0.0
    elif true_codes and not extracted_codes:
        # Extraction found nothing — zero recall
        result.dx_precision = 0.0
        result.dx_recall = 0.0

    # ── Drug matching ──
    true_drugs = set(d.lower() for d in ground["drugs"])
    extracted_treatments = extracted_state.get("prior_treatments", [])
    extracted_drugs = set()
    for t in extracted_treatments:
        drug_name = ""
        if isinstance(t, dict):
            drug_name = t.get("drug", "")
        elif hasattr(t, "drug"):
            drug_name = t.drug
        if drug_name:
            extracted_drugs.add(drug_name.lower())

    result.true_drugs = list(true_drugs)
    result.extracted_drugs = list(extracted_drugs)

    if true_drugs:
        # Count true drugs that are matched by any extracted drug
        drug_matches = sum(
            1 for td in true_drugs
            if any(ed in td or td in ed for ed in extracted_drugs)
        )
        result.drug_recall = drug_matches / len(true_drugs)

    # ── Symptom count ──
    result.n_symptoms_extracted = len(extracted_state.get("symptoms", []))
    result.extraction_confidence = extracted_state.get("extraction_confidence", 0.0)

    return result


def evaluate_pipeline_on_cases(
    cases: list[MIMICCase],
    run_fn,
    procedure_code: str = "",
    verbose: bool = True,
) -> list[PipelineEvalResult]:
    """
    Run the full pipeline on a batch of MIMIC cases and evaluate.

    Args:
        cases: List of assembled MIMICCase objects
        run_fn: The run_pipeline function from src.pipeline
        procedure_code: Optional override CPT code
        verbose: Print progress

    Returns:
        List of per-case evaluation results
    """
    import time
    from dataclasses import asdict

    results = []

    for i, case in enumerate(cases):
        if verbose:
            print(f"  [{i+1}/{len(cases)}] hadm_id={case.hadm_id}...", end=" ")

        inputs = case_to_pipeline_input(case, procedure_code)
        if not inputs["clinical_text"]:
            if verbose:
                print("SKIP (no discharge note)")
            continue

        start = time.time()
        try:
            state = run_fn(**inputs)
            elapsed = time.time() - start

            # Evaluate extraction
            extraction_dict = {
                "diagnosis_codes": state.diagnosis_codes,
                "prior_treatments": [
                    {"drug": t.drug, "duration": t.duration, "outcome": t.outcome}
                    for t in state.prior_treatments
                ],
                "symptoms": state.symptoms,
                "extraction_confidence": state.extraction_confidence,
            }
            ext_eval = evaluate_extraction(case, extraction_dict)

            eval_result = PipelineEvalResult(
                hadm_id=case.hadm_id,
                extraction_eval=ext_eval,
                coverage_score=state.overall_coverage_score,
                approval_probability=state.prediction.approval_probability,
                risk_tier=state.prediction.risk_tier,
                letter_generated=bool(state.draft_letter),
                letter_length=len(state.draft_letter),
                pipeline_status=state.status.value,
                errors=state.errors,
                elapsed_seconds=round(elapsed, 2),
            )

            if verbose:
                print(
                    f"OK ({elapsed:.1f}s) "
                    f"dx_f1={ext_eval.dx_f1:.2f} "
                    f"prob={state.prediction.approval_probability:.2f}"
                )

        except Exception as e:
            elapsed = time.time() - start
            eval_result = PipelineEvalResult(
                hadm_id=case.hadm_id,
                pipeline_status="error",
                errors=[str(e)],
                elapsed_seconds=round(elapsed, 2),
            )
            if verbose:
                print(f"ERROR: {e}")

        results.append(eval_result)

    return results


def summarize_eval_results(results: list[PipelineEvalResult]) -> dict:
    """Aggregate evaluation metrics across all cases."""
    n = len(results)
    if n == 0:
        return {"n_cases": 0}

    successful = [r for r in results if r.pipeline_status == "success"]
    with_extraction = [r for r in successful if r.extraction_eval is not None]

    avg_dx_f1 = 0.0
    avg_dx_precision = 0.0
    avg_dx_recall = 0.0
    avg_drug_recall = 0.0
    avg_confidence = 0.0
    if with_extraction:
        avg_dx_f1 = sum(
            r.extraction_eval.dx_f1 for r in with_extraction
            if r.extraction_eval is not None
        ) / len(with_extraction)
        avg_dx_precision = sum(
            r.extraction_eval.dx_precision for r in with_extraction
            if r.extraction_eval is not None
        ) / len(with_extraction)
        avg_dx_recall = sum(
            r.extraction_eval.dx_recall for r in with_extraction
            if r.extraction_eval is not None
        ) / len(with_extraction)
        avg_drug_recall = sum(
            r.extraction_eval.drug_recall for r in with_extraction
            if r.extraction_eval is not None
        ) / len(with_extraction)
        avg_confidence = sum(
            r.extraction_eval.extraction_confidence for r in with_extraction
            if r.extraction_eval is not None
        ) / len(with_extraction)

    return {
        "n_cases": n,
        "n_successful": len(successful),
        "n_failed": n - len(successful),
        "success_rate": round(len(successful) / n, 3),
        "avg_dx_precision": round(avg_dx_precision, 3),
        "avg_dx_recall": round(avg_dx_recall, 3),
        "avg_dx_f1": round(avg_dx_f1, 3),
        "avg_drug_recall": round(avg_drug_recall, 3),
        "avg_extraction_confidence": round(avg_confidence, 3),
        "avg_coverage_score": round(
            sum(r.coverage_score for r in successful) / max(len(successful), 1), 3
        ),
        "avg_approval_prob": round(
            sum(r.approval_probability for r in successful) / max(len(successful), 1), 3
        ),
        "letters_generated": sum(1 for r in successful if r.letter_generated),
        "avg_letter_length": round(
            sum(r.letter_length for r in successful if r.letter_generated)
            / max(sum(1 for r in successful if r.letter_generated), 1)
        ),
        "avg_elapsed_seconds": round(
            sum(r.elapsed_seconds for r in results) / n, 2
        ),
    }


def _normalize_icd(code: str) -> str:
    """Normalize ICD code: strip dots, uppercase."""
    return code.replace(".", "").upper().strip()