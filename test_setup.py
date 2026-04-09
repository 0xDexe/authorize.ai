"""Quick setup verification — run with: python test_setup.py"""
from dotenv import load_dotenv
load_dotenv(override=True)

print("\n=== 1. Model loader ===")
from src.models.predictor import ApprovalPredictor
p = ApprovalPredictor()
print("Model loaded :", p.model is not None)
print("Model path   :", p.model_path)

print("\n=== 2. LangSmith connection ===")
import os
if os.getenv("LANGCHAIN_API_KEY"):
    try:
        from langsmith import Client
        projects = list(Client().list_projects())
        print("Connected. Projects:", [proj.name for proj in projects])
    except Exception as e:
        print("Failed:", e)
else:
    print("Skipped (LANGCHAIN_API_KEY not set)")

print("\n=== 3. Pipeline smoke test ===")
print("Running pipeline... (this will use your Anthropic API key)")
from src.pipeline import run_pipeline
try:
    result = run_pipeline(
        clinical_text=(
            "45F, L4-L5 disc herniation (M51.16). "
            "Failed 6 weeks PT and NSAIDs. "
            "Requesting MRI lumbar spine (CPT 72148)."
        ),
        payer_id="UHC",
        procedure_code="72148",
        letter_type="initial",
    )
    print("Status         :", result.status)
    if result.errors:
        print("Errors         :")
        for e in result.errors:
            print("  -", e)
    pred = result.prediction
    print("Approval prob  :", pred.approval_probability if pred else "N/A")
    print("Risk tier      :", pred.risk_tier if pred else "N/A")
    print("Coverage score :", result.overall_coverage_score)
    print("Letter preview :", (result.draft_letter or "")[:300], "...")
except Exception as e:
    print("Pipeline raised an exception:", e)
    import traceback; traceback.print_exc()
print("\nAll done!")
