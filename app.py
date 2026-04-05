"""
AuthorizeAI — Streamlit Demo App
==================================
Hackathon demo interface for the PA automation pipeline.
Run: streamlit run app.py
"""

import streamlit as st
import json
import time

from src.pipeline import run_pipeline
from src.state import PipelineStatus, CriterionStatus
from src.retrieval.indexer import init_db, bulk_index_from_directory
from src.models.predictor import ApprovalPredictor, generate_synthetic_training_data
from src.utils.pdf_parser import extract_text
from src.config import config


# ── Page config ──
st.set_page_config(
    page_title="AuthorizeAI",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 AuthorizeAI")
st.caption("Autonomous Prior Authorization — AI-Powered PA Workflow")

# ── Sidebar ──
with st.sidebar:
    st.header("Configuration")

    payer_id = st.selectbox(
        "Payer",
        ["UHC", "AETNA", "CIGNA", "HUMANA", "ELEVANCE", "CENTENE", "KAISER"],
        index=0,
    )

    procedure_code = st.text_input(
        "CPT Procedure Code",
        value="72148",
        help="e.g. 72148 (MRI lumbar), 70553 (MRI brain), 74177 (CT abdomen)",
    )

    letter_type = st.radio(
        "Request Type",
        ["initial", "appeal"],
        horizontal=True,
    )

    st.divider()

    # Admin tools
    with st.expander("🔧 Admin Tools"):
        if st.button("Index Policy Documents"):
            with st.spinner("Indexing policies..."):
                conn = init_db()
                count = bulk_index_from_directory(config.policy_dir, conn)
                st.success(f"Indexed {count} policy chunks")

        if st.button("Train Prediction Model"):
            with st.spinner("Generating synthetic data & training..."):
                X, y = generate_synthetic_training_data(n_samples=2000)
                predictor = ApprovalPredictor()
                metrics = predictor.train(X, y, model_type=config.prediction_model_type)
                st.success(
                    f"Model trained — AUC: {metrics['cv_auc_mean']:.3f} "
                    f"(±{metrics['cv_auc_std']:.3f})"
                )

# ── Main content ──
tab_input, tab_results = st.tabs(["📋 Input", "📊 Results"])

with tab_input:
    input_method = st.radio(
        "Input Method",
        ["Paste clinical notes", "Upload file"],
        horizontal=True,
    )

    clinical_text = ""

    if input_method == "Paste clinical notes":
        clinical_text = st.text_area(
            "Clinical Notes",
            height=300,
            placeholder="Paste clinical notes, discharge summary, or medical record here...",
        )
    else:
        uploaded = st.file_uploader(
            "Upload clinical document",
            type=["txt", "pdf", "json"],
        )
        if uploaded:
            if uploaded.name.endswith(".pdf"):
                # Save temporarily and extract
                tmp = config.data_dir / "tmp_upload.pdf"
                tmp.write_bytes(uploaded.read())
                clinical_text = extract_text(tmp)
                tmp.unlink()
            else:
                clinical_text = uploaded.read().decode("utf-8", errors="replace")

            with st.expander("Preview extracted text"):
                st.text(clinical_text[:2000])

    run_button = st.button(
        "🚀 Run Authorization Pipeline",
        type="primary",
        disabled=not clinical_text,
    )

with tab_results:
    if run_button and clinical_text:
        progress = st.progress(0, text="Starting pipeline...")

        try:
            # Run the pipeline
            progress.progress(10, text="Agent 1: Extracting clinical data...")
            start = time.time()

            result = run_pipeline(
                clinical_text=clinical_text,
                payer_id=payer_id,
                procedure_code=procedure_code,
                letter_type=letter_type,
            )

            elapsed = time.time() - start
            progress.progress(100, text=f"Complete in {elapsed:.1f}s")

            # Status banner
            if result.status == PipelineStatus.SUCCESS:
                st.success(f"Pipeline completed successfully in {elapsed:.1f}s")
            elif result.status == PipelineStatus.NEEDS_REVIEW:
                st.warning("Pipeline completed with review flags")
            else:
                st.error("Pipeline encountered errors")

            if result.errors:
                with st.expander("⚠️ Errors & Warnings"):
                    for err in result.errors:
                        st.warning(err)

            # ── Results display ──
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Approval Probability",
                    f"{result.prediction.approval_probability:.0%}",
                )

            with col2:
                st.metric(
                    "Coverage Score",
                    f"{result.overall_coverage_score:.0%}",
                )

            with col3:
                risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                tier = result.prediction.risk_tier
                st.metric(
                    "Risk Tier",
                    f"{risk_colors.get(tier, '⚪')} {tier.upper()}",
                )

            # Criteria breakdown
            st.subheader("Coverage Criteria Evaluation")
            if result.criteria_results:
                for cr in result.criteria_results:
                    icon = {
                        CriterionStatus.MET: "✅",
                        CriterionStatus.NOT_MET: "❌",
                        CriterionStatus.INSUFFICIENT: "❓",
                    }.get(cr.status, "❓")

                    with st.expander(
                        f"{icon} {cr.criterion_id}: {cr.criterion_text[:80]}"
                    ):
                        st.write(f"**Status:** {cr.status.value}")
                        st.write(f"**Evidence:** {cr.evidence}")
                        st.write(f"**Confidence:** {cr.confidence:.0%}")

            # Gaps
            if result.gaps:
                st.subheader("Documentation Gaps")
                for gap in result.gaps:
                    st.write(f"• {gap}")

            # Recommended actions
            if result.prediction.recommended_actions:
                st.subheader("Recommended Actions")
                for action in result.prediction.recommended_actions:
                    st.info(action)

            # Draft letter
            st.subheader("Draft PA Letter")
            st.text_area(
                "Generated Letter (editable)",
                value=result.draft_letter,
                height=400,
            )

            # Raw JSON output
            with st.expander("🔍 Full Pipeline Output (JSON)"):
                from dataclasses import asdict
                st.json(json.loads(json.dumps(asdict(result), default=str)))

        except Exception as e:
            progress.progress(100, text="Failed")
            st.error(f"Pipeline error: {str(e)}")
            st.exception(e)

    elif not clinical_text:
        st.info("Paste or upload clinical notes in the Input tab, then run the pipeline.")
