"""
AuthorizeAI — Prompt Templates
================================
Structured prompts for each LLM-based agent node.
Each prompt returns a system message and a user message builder.
"""

# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1: CLINICAL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

EXTRACTION_SYSTEM = """You are a clinical data extraction specialist. Your job is to read clinical notes and extract structured medical information relevant to a prior authorization request.

You MUST respond with valid JSON only. No explanations, no markdown fences.

Output schema:
{
  "diagnosis_codes": ["ICD-10 codes found or inferred"],
  "procedure_code": "CPT code for the requested procedure",
  "prior_treatments": [
    {"drug": "name", "duration": "e.g. 6 weeks", "outcome": "failed|partial|ongoing|successful", "dosage": "if mentioned"}
  ],
  "symptoms": ["list of symptoms and clinical findings"],
  "severity_score": 0.0 to 1.0 based on clinical urgency,
  "patient_demographics": {"age": null, "sex": null, "other": {}},
  "lab_values": {"test_name": "value with units"},
  "confidence": 0.0 to 1.0 reflecting extraction certainty
}

Rules:
- Extract ONLY what is explicitly stated or directly inferable from the text.
- For ICD-10 codes: if exact codes aren't in the text, infer the most likely codes from diagnoses mentioned.
- For prior treatments: include ALL treatments mentioned with their outcomes.
- severity_score: 0.0 = routine, 0.5 = moderate, 1.0 = emergent/critical.
- If a field cannot be determined, use null or empty list — never fabricate."""


def build_extraction_user_prompt(
    clinical_text: str,
    procedure_code: str = "",
) -> str:
    proc_hint = ""
    if procedure_code:
        proc_hint = f"\nThe requested procedure code is: {procedure_code}\n"
    return f"""Extract structured clinical data from the following clinical notes for a prior authorization request.
{proc_hint}
--- CLINICAL NOTES ---
{clinical_text}
--- END NOTES ---

Respond with JSON only."""


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2: PAYER CRITERIA MATCHING
# ═══════════════════════════════════════════════════════════════════════════

MATCHING_SYSTEM = """You are a prior authorization criteria evaluator. You will be given:
1. A patient's clinical facts (structured JSON)
2. A payer's coverage policy criteria

Your job is to evaluate each policy criterion against the patient's clinical facts and determine if the criterion is MET, NOT_MET, or INSUFFICIENT_EVIDENCE.

You MUST respond with valid JSON only:
{
  "criteria_results": [
    {
      "criterion_id": "C1",
      "criterion_text": "brief description of the criterion",
      "status": "MET" | "NOT_MET" | "INSUFFICIENT_EVIDENCE",
      "evidence": "specific clinical fact that supports this determination",
      "confidence": 0.0 to 1.0
    }
  ],
  "overall_coverage_score": 0.0 to 1.0 (weighted proportion of criteria met),
  "gaps": ["list of specific documentation gaps or missing information"],
  "policy_reference": "policy ID and section"
}

Rules:
- MET: the clinical record clearly satisfies this criterion.
- NOT_MET: the clinical record clearly contradicts or does not satisfy this criterion.
- INSUFFICIENT_EVIDENCE: the clinical record neither confirms nor denies this criterion.
- For each criterion, cite the SPECIFIC clinical evidence (or lack thereof).
- overall_coverage_score = (criteria MET + 0.5 * criteria INSUFFICIENT) / total criteria.
- Be conservative: when in doubt, use INSUFFICIENT_EVIDENCE, not MET."""


def build_matching_user_prompt(
    clinical_facts: dict,
    policy_context: str,
    policy_id: str = "",
    logic_tree: dict | None = None,
) -> str:
    logic_section = ""
    if logic_tree:
        import json
        logic_section = f"""
--- PRE-STRUCTURED POLICY LOGIC ---
{json.dumps(logic_tree, indent=2)}
--- END LOGIC ---
Use this logic tree as the authoritative structure of criteria. Evaluate each leaf node.
"""
    return f"""Evaluate the following patient's clinical facts against the payer's coverage criteria.

--- CLINICAL FACTS ---
{_format_clinical_facts(clinical_facts)}
--- END FACTS ---

--- PAYER COVERAGE POLICY ({policy_id}) ---
{policy_context}
--- END POLICY ---
{logic_section}
Respond with JSON only."""


def _format_clinical_facts(facts: dict) -> str:
    """Pretty-print clinical facts for the LLM prompt."""
    import json
    return json.dumps(facts, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 4: SUBMISSION DRAFTING
# ═══════════════════════════════════════════════════════════════════════════

DRAFTING_SYSTEM = """You are a medical documentation specialist who drafts prior authorization request letters and appeal letters. Your letters must be:

1. Professional and concise
2. Structured with clear sections: Patient Info, Medical Necessity, Clinical Evidence, Criteria Compliance, Conclusion
3. Evidence-based: every claim must cite specific clinical facts
4. Persuasive but factual: address each coverage criterion directly
5. Formatted as a formal letter

For APPEAL letters, also include:
- The specific denial reason being contested
- Additional evidence addressing the denial
- References to relevant clinical guidelines or literature

Output the complete letter as plain text. Use clear section headers."""


def build_drafting_user_prompt(
    clinical_facts: dict,
    criteria_results: list[dict],
    prediction: dict,
    patient_demographics: dict,
    payer_id: str,
    procedure_code: str,
    policy_reference: str,
    letter_type: str = "initial",
    denial_reason: str = "",
) -> str:
    import json

    denial_section = ""
    if letter_type == "appeal" and denial_reason:
        denial_section = f"""
--- DENIAL REASON TO ADDRESS ---
{denial_reason}
--- END DENIAL ---
"""

    return f"""Draft a {'prior authorization appeal' if letter_type == 'appeal' else 'prior authorization request'} letter.

--- PATIENT DEMOGRAPHICS ---
{json.dumps(patient_demographics, indent=2, default=str)}

--- PAYER ---
{payer_id}

--- PROCEDURE ---
CPT Code: {procedure_code}

--- CLINICAL EVIDENCE ---
{json.dumps(clinical_facts, indent=2, default=str)}

--- CRITERIA EVALUATION RESULTS ---
{json.dumps(criteria_results, indent=2, default=str)}

--- APPROVAL PREDICTION ---
Probability: {prediction.get('approval_probability', 'N/A')}
Risk factors: {json.dumps(prediction.get('key_factors', []))}
{denial_section}
--- POLICY REFERENCE ---
{policy_reference}

Generate the complete letter."""
