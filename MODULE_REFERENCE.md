# AuthorizeAI — Module Reference

> One-file reference for every module in the backend. Read this before touching any code.

---

## Project Tree

```
authorizeai/
├── app.py                          # Streamlit demo entry point
├── train_model.py                  # CLI script to train prediction model (synthetic data)
├── setup_data.py                   # CLI for MIMIC setup, eval, and MIMIC-based training
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── data/
│   ├── policies/                   # Raw policy documents (.txt, .json)
│   ├── clinical_notes/             # Sample clinical notes for testing
│   ├── models/                     # Serialized ML models (.pkl)
│   ├── mimic-iv/                   # ← MIMIC-IV data (you download from PhysioNet)
│   │   ├── hosp/                   #   patients, admissions, diagnoses_icd, etc.
│   │   └── note/                   #   discharge, radiology, radiology_detail
│   ├── policy_index.db             # SQLite FTS5 index (auto-created)
│   └── base_rates.db               # SQLite base rates from CMS/KFF (auto-created)
├── src/
│   ├── __init__.py                 # Exports: run_pipeline, build_pipeline, AuthorizeState
│   ├── state.py                    # Shared pipeline state schema
│   ├── pipeline.py                 # LangGraph StateGraph definition
│   ├── config.py                   # Central configuration from env vars
│   ├── agents/
│   │   ├── extraction.py           # Agent 1: Clinical Extraction
│   │   ├── matching.py             # Agent 2: Payer Criteria Matching
│   │   ├── prediction.py           # Agent 3: Approval Prediction
│   │   └── drafting.py             # Agent 4: Submission Drafting
│   ├── data/                       # ← NEW: data loading & evaluation
│   │   ├── mimic_loader.py         # MIMIC-IV CSV reader + case assembler
│   │   ├── public_rates.py         # CMS/KFF/state base rate ingestion
│   │   ├── evaluation.py           # Pipeline eval harness (vs MIMIC ground truth)
│   │   └── training_gen.py         # Training data generator (MIMIC + base rates)
│   ├── retrieval/
│   │   ├── indexer.py              # SQLite FTS5 indexing & chunking
│   │   └── searcher.py             # BM25 search & context assembly
│   ├── models/
│   │   ├── features.py             # Feature engineering (now DB-backed)
│   │   └── predictor.py            # ML model training & inference
│   ├── prompts/
│   │   └── templates.py            # All LLM prompt templates
│   └── utils/
│       ├── llm_client.py           # LLM API wrapper (Anthropic/OpenAI)
│       └── pdf_parser.py           # PDF & text extraction
└── tests/
    └── (test files)
```

---

## Pipeline Flow

```
START → extract → [gate] → match → [gate] → predict → draft → END
                    ↘ fail ─────────────────────────────────→ END
```

Every agent reads from and writes to a single shared `AuthorizeState` dataclass. The state accumulates outputs as it flows through the graph. Conditional gates after Agents 1 and 2 can short-circuit to END on critical failures.

---

## Module-by-Module Reference

### `src/state.py`

The backbone of the pipeline. Defines `AuthorizeState`, a flat dataclass that every agent node receives and returns. Organized into five sections:

- **Inputs** — `raw_clinical_text`, `payer_id`, `procedure_code`, `patient_id`. Set at pipeline invocation.
- **Agent 1 outputs** — `diagnosis_codes`, `extracted_procedure_code`, `prior_treatments`, `symptoms`, `severity_score`, `patient_demographics`, `lab_values`, `extraction_confidence`.
- **Agent 2 outputs** — `criteria_results` (list of `CriterionResult`), `overall_coverage_score`, `gaps`, `policy_reference`, `matched_policy_text`.
- **Agent 3 outputs** — `prediction` (`PredictionResult` with probability, risk tier, factors, actions).
- **Agent 4 outputs** — `draft_letter`, `letter_type`, `cited_evidence`.
- **Metadata** — `status` (enum: pending/running/success/failed/needs_review), `errors`, `current_agent`.

Also defines supporting types: `Treatment`, `CriterionResult`, `PredictionResult`, `CriterionStatus` (MET/NOT_MET/INSUFFICIENT_EVIDENCE), and `PipelineStatus`.

---

### `src/pipeline.py`

Wires the four agents into a LangGraph `StateGraph`. Two key exports:

- **`build_pipeline()`** — Constructs and compiles the graph. Adds four nodes (`extract`, `match`, `predict`, `draft`), connects them with conditional edges. Returns a compiled graph object.
- **`run_pipeline(clinical_text, payer_id, procedure_code, ...)`** — Convenience function. Creates an `AuthorizeState`, invokes the compiled graph, returns the completed state.

Three conditional edge functions control routing:
- After extraction: fails to END if confidence is critically low (< 0.2); proceeds with a warning flag if low but usable (0.2–0.4); proceeds normally otherwise.
- After matching: fails to END if no policy found or LLM returned garbage.
- After prediction: always proceeds to drafting — prediction failure is non-fatal because the heuristic fallback in the predictor guarantees a result.

---

### `src/agents/extraction.py`

**Agent 1: Clinical Extraction.** Takes raw clinical text, sends it to the LLM with a structured extraction prompt, parses the JSON response into state fields.

- Reads: `raw_clinical_text`, `procedure_code`
- Writes: `diagnosis_codes`, `extracted_procedure_code`, `prior_treatments`, `symptoms`, `severity_score`, `patient_demographics`, `lab_values`, `extraction_confidence`
- Uses low temperature (0.05) for deterministic extraction.
- Flags `NEEDS_REVIEW` if extraction confidence < 0.4.
- Handles non-JSON LLM responses gracefully by logging the error and setting status.

---

### `src/agents/matching.py`

**Agent 2: Payer Criteria Matching.** The most complex agent. Three-step process:

1. **BM25 retrieval** — Builds search keywords from diagnosis codes, symptoms, and drug names. Queries the SQLite FTS5 index filtered by payer ID. Retrieves top-8 policy chunks.
2. **Logic tree check** — If the policy has a pre-structured JSON logic tree (stored during ingestion), loads it and provides it to the LLM as authoritative structure.
3. **LLM evaluation** — Sends clinical facts + policy context to the LLM. The LLM evaluates each criterion as MET, NOT_MET, or INSUFFICIENT_EVIDENCE with supporting evidence.

- Reads: all Agent 1 outputs + `payer_id`, `procedure_code`
- Writes: `criteria_results`, `overall_coverage_score`, `gaps`, `policy_reference`, `matched_policy_text`
- Coverage score formula: `(MET + 0.5 × INSUFFICIENT) / total_criteria`

---

### `src/agents/prediction.py`

**Agent 3: Approval Prediction.** Thin wrapper around `ApprovalPredictor`. Extracts features from the accumulated state, runs model inference (or heuristic fallback), writes the `PredictionResult`.

- Reads: full upstream state
- Writes: `prediction` (probability, risk tier, key factors, recommended actions)
- Uses a module-level singleton to avoid reloading the model per run.
- Non-fatal failure: if the model errors out, falls back to `coverage_score × 0.8` with a "review manually" recommendation. Pipeline continues to drafting regardless.

---

### `src/agents/drafting.py`

**Agent 4: Submission Drafting.** Generates a complete PA request letter (or appeal letter) using all upstream data. The letter is structured with sections for patient info, medical necessity, clinical evidence, criteria compliance, and conclusion.

- Reads: full upstream state
- Writes: `draft_letter`, `cited_evidence`
- Uses slightly higher temperature (0.3) for natural language generation.
- Post-generation, scans the letter for references to clinical facts (diagnosis codes, drug names, symptoms) and logs them as `cited_evidence` for audit trail.
- Sets pipeline status to `SUCCESS` on completion.
- For appeals: the prompt includes the denial reason and instructs the LLM to address it directly.

---

### `src/retrieval/indexer.py`

Policy document ingestion and SQLite FTS5 indexing. Two tables:

- **`policy_meta`** — One row per policy. Stores `policy_id`, `payer_id`, `procedure_code`, `policy_name`, `source_url`, and `raw_json` (the pre-structured logic tree if available).
- **`policy_chunks`** (FTS5 virtual table) — One row per criterion-level chunk. Tokenized with Porter stemming and Unicode support.

Key functions:
- **`chunk_policy_text()`** — Splits raw policy text into criterion-level chunks using regex on numbered/lettered criteria patterns. Falls back to paragraph-level splitting. Auto-detects section type (indications, contraindications, documentation).
- **`index_policy()`** — Stores metadata + all chunks for one policy. Handles upserts (deletes old chunks before re-indexing).
- **`bulk_index_from_directory()`** — Batch-indexes all `.txt` and `.json` files in a directory. JSON files can include a `logic_tree` field for pre-structured criteria.

---

### `src/retrieval/searcher.py`

BM25 query interface over the FTS5 index. Functions:

- **`search_policies(procedure_code, payer_id, clinical_keywords, top_k)`** — Main search function. Builds an FTS5 query combining the procedure code with clinical keywords via OR. Filters by payer ID. Falls back to unfiltered search if the payer filter returns nothing.
- **`get_policy_logic_tree(policy_id)`** — Retrieves the pre-structured JSON logic tree from `policy_meta`, if it exists.
- **`get_all_criteria_for_policy(policy_id)`** — Returns every chunk for a policy (used when a logic tree exists and we want the full criterion set, not just BM25-ranked).
- **`build_context_block(results)`** — Formats search results into a single text block for LLM context, separated by criterion ID and section.

---

### `src/models/features.py`

Feature engineering module. Transforms the accumulated `AuthorizeState` into a flat `FeatureVector` (15 features) for the ML model.

**Features (grouped by source):**

From Agent 2 (criteria matching):
- `coverage_score`, `criteria_met_ratio`, `criteria_not_met_ratio`, `criteria_insufficient_ratio`, `num_gaps`

From Agent 1 (clinical extraction):
- `severity_score`, `num_prior_treatments`, `num_failed_treatments`, `has_symptoms`, `num_diagnosis_codes`, `extraction_confidence`

From public base rate data (CMS/KFF):
- `payer_denial_rate` — historical denial rate for this payer (e.g., UHC = 12.8%, Elevance = 4.2%). Source: KFF analysis of CMS Part C Reporting PUF.
- `procedure_approval_rate` — historical approval rate for this procedure category (e.g., MRI = 88%, brand drug = 78%). Source: CMS OIG + published radiology PA studies.

From demographics:
- `patient_age`, `is_female`

Also contains `CPT_TO_CATEGORY` mapping (10 hackathon-scope procedures) and lookup dicts for payer/procedure base rates. In production, these would come from a SQLite table updated from CMS data refreshes.

---

### `src/models/predictor.py`

ML model wrapper. `ApprovalPredictor` class with three modes:

1. **Trained model** — Loads a pickled sklearn model + scaler from `data/models/approval_model.pkl`. Runs `predict_proba()` and extracts the approval probability.
2. **Heuristic fallback** — Used when no trained model exists. Weighted formula: 50% coverage score + 20% historical base rate + 15% failed treatment signal + 15% severity score.
3. **Training** — `train(X, y, model_type)` fits a logistic regression or gradient-boosted tree with 5-fold cross-validated AUC. Saves the model to disk.

Also provides `generate_synthetic_training_data(n_samples)` which produces realistic feature distributions calibrated against CMS/KFF published rates. The synthetic label generation uses a logistic function over feature combinations, so the trained model learns plausible decision boundaries.

---

### `src/prompts/templates.py`

All LLM prompt templates, centralized. Three prompt sets:

- **Extraction** (`EXTRACTION_SYSTEM` + `build_extraction_user_prompt`) — Instructs the LLM to output a strict JSON schema with diagnosis codes, CPT codes, treatments, symptoms, severity, demographics, and lab values. Dynamic hint: if a procedure code is pre-specified, it's included to focus extraction.
- **Matching** (`MATCHING_SYSTEM` + `build_matching_user_prompt`) — Instructs the LLM to evaluate each policy criterion against clinical facts. Outputs JSON with per-criterion status, evidence, and overall coverage score. If a pre-structured logic tree is available, it's injected as authoritative structure so the LLM only does evaluation, not parsing.
- **Drafting** (`DRAFTING_SYSTEM` + `build_drafting_user_prompt`) — Instructs the LLM to generate a formal PA letter. Provides all upstream data as context. For appeals, includes the denial reason and instructs direct rebuttal.

---

### `src/utils/llm_client.py`

Thin LLM API wrapper. Single entry point: `call_llm(system_prompt, user_prompt, ...)`.

- **Provider resolution** — Checks explicit param → env var → auto-detect from API key presence.
- **Anthropic path** — Uses the `anthropic` SDK. Sends system prompt separately (Anthropic format). Default model: `claude-sonnet-4-20250514`.
- **OpenAI path** — Uses the `openai` SDK. Standard chat completion format. Default model: `gpt-4o-mini`.
- **JSON parsing** — Strips markdown fences, tries direct parse, then scans for embedded JSON object. Returns `dict` if parseable, raw `str` otherwise.
- **Retry logic** — Up to 2 retries with linear backoff on any exception.

---

### `src/utils/pdf_parser.py`

Document text extraction utility.

- **`extract_text(file_path)`** — Auto-detects format from file extension. Supports PDF (PyMuPDF), plain text, and JSON (including FHIR-style bundles with a `text` field).
- **`extract_text_from_string(raw)`** — Cleans raw clinical text input by normalizing line endings and collapsing excessive whitespace while preserving paragraph structure.

---

### `src/config.py`

Centralized configuration loaded from environment variables. Defines paths (data dir, policy dir, model dir, DB path), LLM settings (provider, API key, model name), retrieval settings (BM25 top-k), and prediction model type. Includes a `validate()` method that returns a list of configuration issues (e.g., missing API keys).

---

### `app.py`

Streamlit demo interface. Two tabs:

- **Input tab** — Paste clinical notes or upload a file (txt/pdf/json). Select payer, CPT code, and request type (initial/appeal).
- **Results tab** — Runs the pipeline on submit. Displays: approval probability metric, coverage score, risk tier, per-criterion expandable cards (with status icon, evidence, confidence), documentation gaps, recommended actions, and the full draft letter in an editable text area. Raw JSON output available in an expander.

Sidebar includes admin tools: index policy documents from `data/policies/` and train the prediction model with synthetic data.

---

### `setup_data.py`

CLI for data initialization, MIMIC validation, evaluation, and MIMIC-based model training. Subcommands:

- **`validate`** — Checks `data/mimic-iv/` for required tables, reports what's found/missing
- **`seed`** — Seeds the `base_rates.db` with KFF/CMS/AMA published rates. Optionally loads a CMS PUF CSV with `--cms-puf`
- **`train --n 500`** — Loads N MIMIC cases, generates features using heuristic extraction, trains the prediction model
- **`eval --n 10`** — Runs the full LLM pipeline on N MIMIC cases and evaluates extraction accuracy against MIMIC ground truth
- **`summary`** — Prints all base rate data currently in the database

---

## Data Modules (NEW)

### `src/data/mimic_loader.py`

The core MIMIC-IV integration. Reads gzipped CSV files from the PhysioNet download and assembles them into `MIMICCase` objects that the pipeline can consume directly.

**Tables it reads:**

From `hosp/` module:
- `patients.csv.gz` — demographics (gender, anchor_age)
- `admissions.csv.gz` — admission details and insurance type (payer proxy)
- `diagnoses_icd.csv.gz` — ICD-9/10 diagnosis codes per admission, with priority ranking
- `procedures_icd.csv.gz` — ICD procedure codes per admission
- `prescriptions.csv.gz` — medication orders (drug name, dosage, route, duration)

From `note/` module:
- `discharge.csv.gz` — free-text discharge summaries (this is what Agent 1 processes)
- `radiology.csv.gz` — free-text radiology reports
- `radiology_detail.csv.gz` — CPT codes and exam names for radiology studies

**Key functions:**
- **`load_all_cases(mimic_dir, limit)`** — Loads all tables, assembles `MIMICCase` objects by joining on `subject_id` and `hadm_id`. Returns a list.
- **`case_to_pipeline_input(case)`** — Converts a `MIMICCase` into `run_pipeline()` kwargs. Maps MIMIC's insurance field to a payer ID and selects CPT code from radiology details.
- **`case_to_ground_truth(case)`** — Extracts structured ground truth (ICD codes, drug list, demographics) for evaluating Agent 1 accuracy.
- **`filter_cases_by_diagnosis(cases, icd_prefixes)`** — Filter to cases with specific diagnoses (e.g., `["M54", "M51"]` for back pain).
- **`filter_cases_with_imaging(cases)`** — Filter to cases with radiology reports (highest PA relevance).
- **`validate_mimic_directory()`** — Reports which tables exist, which are missing, and whether the directory is ready.

**How MIMICCase maps to the pipeline:**
- `primary_discharge_text` → `raw_clinical_text` (Agent 1 input)
- `payer_proxy` → `payer_id` (derived from MIMIC `insurance` field: Medicare/Medicaid/Other)
- `radiology_cpt_codes` → `procedure_code` (from `radiology_detail` CPT codes)
- `icd10_codes` → ground truth for evaluating `diagnosis_codes` extraction
- `drug_list` → ground truth for evaluating `prior_treatments` extraction

---

### `src/data/public_rates.py`

Manages a SQLite database (`data/base_rates.db`) of publicly available PA outcome data. Two tables:

**`payer_denial_rates`** — Payer-level denial rates with source tracking. Columns: `payer_id`, `payer_name`, `contract_id`, `year`, `total_requests`, `total_denials`, `denial_rate`, `appeal_rate`, `overturn_rate`, `source`. Sources are prioritized: `PAYER_DISCLOSURE` > `CMS_PUF` > `KFF` > `STATE_*`.

**`procedure_approval_rates`** — Procedure-category approval rates. Columns: `procedure_category`, `cpt_codes` (JSON list), `approval_rate`, `denial_rate`, `sample_size`, `year`, `source`.

**Key functions:**
- **`seed_kff_base_rates()`** — Loads hardcoded rates from KFF's 2024 analysis (UHC 12.8%, Elevance 4.2%, etc.) and procedure-category rates from CMS OIG and AMA surveys. Zero network calls — these are published figures transcribed into code.
- **`load_cms_puf(csv_path)`** — Parses the CMS Part C/D Reporting Requirements PUF (a downloadable CSV from cms.gov). Maps insurer names to canonical payer IDs via fuzzy matching.
- **`load_payer_disclosure(payer_id, approval_pct, denial_pct, ...)`** — Manual entry point for payer-published PA metrics (the March 2026 mandate). Since these aren't yet in standardized format, you enter them after scraping each payer's website.
- **`load_state_insurer_report(csv_path, state)`** — Loads Connecticut or Vermont state insurance department PA data.
- **`get_payer_denial_rate(payer_id)`** — Query function used by `features.py`. Returns the best available denial rate for a payer, with source priority ordering.
- **`get_procedure_approval_rate(category)`** — Query function used by `features.py`.

---

### `src/data/evaluation.py`

Evaluation harness that uses MIMIC structured data as ground truth to measure pipeline quality.

**`evaluate_extraction(case, extracted_state)`** — Compares Agent 1's extracted diagnosis codes and drug names against MIMIC's `diagnoses_icd` and `prescriptions` tables. Uses prefix matching for ICD codes (extracted `M54` matches true `M544`) and case-insensitive substring matching for drug names. Returns per-case precision, recall, F1 for diagnoses and recall for drugs.

**`evaluate_pipeline_on_cases(cases, run_fn)`** — Batch evaluator. Runs the full pipeline on each case, collects extraction accuracy, coverage scores, approval probabilities, letter generation success, and timing. Prints per-case progress.

**`summarize_eval_results(results)`** — Aggregates across all cases: mean diagnosis F1, mean drug recall, mean coverage score, success rate, average elapsed time.

---

### `src/data/training_gen.py`

Generates labeled training data for Agent 3 by combining MIMIC clinical cases with public base rates. Two modes:

**Heuristic mode** (default, free, fast) — Estimates features directly from MIMIC structured data without any LLM calls. Coverage score is approximated from documentation richness (diagnosis count, medication count, procedure count). Labels are generated probabilistically using the same logistic function calibrated against CMS/KFF base rates. Good for the hackathon prototype.

**Agent mode** (requires LLM API calls) — Runs Agents 1+2 on each case to get real extraction and matching features. More accurate but costs ~$0.01-0.05 per case. Use for post-hackathon model improvement.

Both modes produce `(X, y)` numpy arrays ready for `ApprovalPredictor.train()`.

---

### `src/models/features.py` (UPDATED)

Now queries the `base_rates.db` SQLite database for payer denial rates and procedure approval rates before falling back to hardcoded dictionaries. This means when you load CMS PUF data or payer disclosures into the database, Agent 3's predictions automatically incorporate the latest public data without code changes.

---

## Data Audit: What's Real vs. Synthetic

| Component | Source | Status |
|-----------|--------|--------|
| Clinical notes fed to Agent 1 | MIMIC-IV discharge summaries | **Real** — 331K de-identified notes from BIDMC |
| Ground truth ICD codes | MIMIC-IV `diagnoses_icd` | **Real** — actual coded diagnoses |
| Ground truth prescriptions | MIMIC-IV `prescriptions` | **Real** — actual medication orders |
| Radiology CPT codes | MIMIC-IV `radiology_detail` | **Real** — actual procedure codes |
| Patient demographics | MIMIC-IV `patients` | **Real** — de-identified age/gender |
| Payer denial rates | KFF analysis of CMS PUF | **Real** — published aggregate rates |
| Procedure approval rates | CMS OIG / AMA surveys | **Real** — published aggregate rates |
| Payer-published PA metrics | March 2026 mandate disclosures | **Real** — payer website scrapes |
| Coverage policy documents | Payer public websites | **Real** — but manually collected |
| PA outcome labels (approved/denied) | Generated from base rates | **Synthetic** — no case-level outcome data |
| Coverage score for training | Heuristic from MIMIC structure | **Estimated** — not from actual Agent 2 |

**The one thing that's still synthetic:** case-level approval/denial labels. There is no public dataset mapping individual PA requests to outcomes. The labels are probabilistic, calibrated against real aggregate rates. This is the honest framing for judges: "real clinical data, real base rates, synthetic labels pending real outcome data from a pilot partner."

---

## Quickstart

```bash
# 1. Install
cd authorizeai
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your API key

# 3. Download MIMIC-IV from PhysioNet into data/mimic-iv/
#    hosp module: https://physionet.org/content/mimiciv/2.2/
#    note module: https://physionet.org/content/mimic-iv-note/2.2/

# 4. Validate MIMIC directory
python setup_data.py validate

# 5. Seed public base rates
python setup_data.py seed

# 6. Add policy documents to data/policies/
#    Format: PAYER_PROCEDURE.txt or .json with logic_tree

# 7. Index policies
python -c "from src.retrieval.indexer import init_db, bulk_index_from_directory; \
           conn = init_db(); print(f'Indexed {bulk_index_from_directory(\"data/policies\", conn)} chunks')"

# 8. Train prediction model (option A: MIMIC-based)
python setup_data.py train --n 500 --model logistic

# 8. Train prediction model (option B: synthetic only, no MIMIC needed)
python train_model.py --model logistic --samples 2000

# 9. Run demo
streamlit run app.py

# 10. Evaluate pipeline against MIMIC ground truth
python setup_data.py eval --n 10
```

---

## Data Flow Summary

```
Clinical Notes (text/PDF)
        │
        ▼
┌─────────────────┐   raw_clinical_text
│  Agent 1:       │──────────────────────▶ diagnosis_codes, procedure_code,
│  Extraction     │                        prior_treatments, symptoms,
│  (LLM)          │                        severity_score, demographics
└────────┬────────┘
         │
         ▼
┌─────────────────┐   clinical facts + payer_id
│  Agent 2:       │──────────────────────▶ criteria_results (per-criterion
│  Matching       │                        MET/NOT_MET/INSUFFICIENT),
│  (BM25 + LLM)   │                        coverage_score, gaps
└────────┬────────┘
         │
         ▼
┌─────────────────┐   coverage_score + features + base rates
│  Agent 3:       │──────────────────────▶ approval_probability, risk_tier,
│  Prediction     │                        key_factors, recommended_actions
│  (sklearn)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐   all upstream data
│  Agent 4:       │──────────────────────▶ draft_letter, cited_evidence
│  Drafting       │
│  (LLM)          │
└─────────────────┘
```
