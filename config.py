"""
AuthorizeAI — Configuration
==============================
Central configuration loaded from environment variables.
"""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    # LLM settings
    llm_provider: str = os.getenv("AUTHORIZEAI_LLM_PROVIDER", "anthropic")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Paths
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = project_root / "data"
    policy_dir: Path = data_dir / "policies"
    clinical_dir: Path = data_dir / "clinical_notes"
    model_dir: Path = data_dir / "models"
    db_path: Path = data_dir / "policy_index.db"

    # Retrieval settings
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "5"))

    # Prediction model
    prediction_model_type: str = os.getenv("PREDICTION_MODEL", "logistic")
    approval_model_path: str = os.getenv("APPROVAL_MODEL_PATH", "")

    # LangSmith tracing (auto-picked up by LangGraph via env vars; listed here for visibility)
    langchain_tracing_v2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "authorize-ai")

    def validate(self) -> list[str]:
        """Check for missing required config. Returns list of issues."""
        issues = []
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY not set")
        if self.llm_provider == "openai" and not self.openai_api_key:
            issues.append("OPENAI_API_KEY not set")
        return issues


config = Config()
