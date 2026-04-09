# AuthorizeAI

AI-powered medical prior authorization (PA) automation. Accepts clinical notes, extracts diagnoses and treatments, matches against payer criteria, predicts approval probability, and drafts the PA letter — all in one pipeline.

**Pipeline:** Clinical Extraction → Payer Criteria Matching → Approval Prediction → Letter Drafting

---

## Requirements

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) (or OpenAI key as fallback)
- _(Optional)_ A [LangSmith API key](https://smith.langchain.com/) for tracing

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `APPROVAL_MODEL_PATH` | Absolute path to your `approval_model.pkl` (optional — defaults to `data/models/approval_model.pkl`) |
| `LANGCHAIN_API_KEY` | LangSmith API key for tracing (optional) |
| `LANGCHAIN_TRACING_V2` | Set to `true` to enable LangSmith tracing (optional) |

### 3. Initialize data (first time only)

```bash
python generate_policies.py       # generates synthetic payer policies
python setup_data.py seed         # seeds base rates database
```

If you have MIMIC-IV data, you can also train and evaluate the model:

```bash
# Download MIMIC-IV from PhysioNet into data/mimic-iv/, then:
python setup_data.py validate     # confirm data is in place
python setup_data.py train --n 500
python setup_data.py eval --n 10
```

If you have a pre-trained model, copy or symlink it:

```bash
cp /path/to/approval_model.pkl data/models/approval_model.pkl
# or set APPROVAL_MODEL_PATH=/path/to/approval_model.pkl in .env
```

---

## Running locally

```bash
streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

---

## Running with Docker (team hosting)

```bash
docker compose up --build
```

The app is served at `http://<host-ip>:8501`. Your teammates can access it over the network without any local Python setup.

**Before building**, make sure your `data/models/approval_model.pkl` is in place (or set `APPROVAL_MODEL_PATH` in `.env`), and that `.env` contains valid API keys.

To run in the background:

```bash
docker compose up --build -d
docker compose logs -f          # tail logs
docker compose down             # stop
```

---

## LangSmith tracing

When `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set in `.env`, every pipeline run is automatically traced. You can view agent inputs/outputs, LLM calls, latency, and errors at [smith.langchain.com](https://smith.langchain.com).

No code changes are needed — LangGraph picks up the env vars automatically.

---

## Testing

### Smoke test — run the pipeline end-to-end

```bash
python - <<'EOF'
from dotenv import load_dotenv
load_dotenv()

from src.pipeline import run_pipeline

result = run_pipeline(
    clinical_text="""
        Patient: 45-year-old female with chronic low back pain.
        Diagnosis: L4-L5 disc herniation (M51.16).
        Tried 6 weeks of physical therapy and NSAIDs without relief.
        Requesting MRI lumbar spine (CPT 72148).
    """,
    payer_id="UHC",
    procedure_code="72148",
    letter_type="initial",
)

print("Status          :", result.status)
print("Approval prob   :", result.prediction.approval_probability if result.prediction else "N/A")
print("Risk tier       :", result.prediction.risk_tier if result.prediction else "N/A")
print("Coverage score  :", result.overall_coverage_score)
print("Draft letter    :", (result.draft_letter or "")[:200], "...")
EOF
```

### Run the test suite

```bash
pytest tests/ -v
```

### Test just the model loader

```bash
python - <<'EOF'
from dotenv import load_dotenv
load_dotenv()
from src.models.predictor import ApprovalPredictor

p = ApprovalPredictor()
print("Model loaded    :", p.model is not None)
print("Model path      :", p.model_path)
print("Has scaler      :", p.scaler is not None)
EOF
```

### Test LangSmith connection

```bash
python - <<'EOF'
import os
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
client = Client()
projects = list(client.list_projects())
print("LangSmith connected. Projects:", [p.name for p in projects])
EOF
```

---

## Project structure

```
authorize.ai/
├── app.py                    # Streamlit web UI
├── config.py                 # Central configuration (env vars)
├── generate_policies.py      # Synthetic payer policy generator
├── setup_data.py             # CLI: seed, train, eval
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── data/
│   ├── policies/             # Payer policy documents
│   ├── models/               # approval_model.pkl goes here
│   ├── policy_index.db       # SQLite FTS5 search index
│   └── base_rates.db         # CMS/KFF approval rate data
└── src/
    ├── pipeline.py           # LangGraph orchestration
    ├── state.py              # Shared pipeline state
    ├── agents/
    │   ├── extraction.py     # Agent 1: clinical data extraction
    │   ├── matching.py       # Agent 2: payer criteria matching
    │   ├── prediction.py     # Agent 3: approval prediction
    │   └── drafting.py       # Agent 4: PA letter generation
    ├── models/
    │   ├── predictor.py      # sklearn model wrapper
    │   └── features.py       # feature engineering
    ├── retrieval/
    │   ├── indexer.py        # SQLite FTS5 indexer
    │   └── searcher.py       # BM25 policy search
    ├── prompts/
    │   └── templates.py      # LLM prompt templates
    └── utils/
        ├── llm_client.py     # Anthropic / OpenAI wrapper
        └── pdf_parser.py     # PDF text extraction
```
