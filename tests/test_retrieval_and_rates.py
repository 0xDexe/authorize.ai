"""Tests for policy indexing, BM25 search, and public base rates."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.retrieval.indexer import (
    init_db, PolicyChunk, index_policy, chunk_policy_text,
    bulk_index_from_directory,
)
from src.retrieval.searcher import (
    search_policies, get_policy_logic_tree,
    get_all_criteria_for_policy, build_context_block,
)
from src.data.public_rates import (
    init_base_rate_db, seed_kff_base_rates,
    load_payer_disclosure, get_payer_denial_rate,
    get_procedure_approval_rate, _normalize_payer_name,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    conn = init_db(db_path)
    yield conn
    conn.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def tmp_rates_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    conn = init_base_rate_db(db_path)
    yield conn
    conn.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def indexed_db(tmp_db):
    """Database with sample policy data indexed."""
    chunks = [
        PolicyChunk(
            policy_id="UHC_MRI_LUMBAR",
            payer_id="UHC",
            procedure_code="72148",
            criterion_id="C1",
            criterion_text="Conservative treatment must be trialed for at least 6 weeks including physical therapy or NSAIDs",
            section="indications",
        ),
        PolicyChunk(
            policy_id="UHC_MRI_LUMBAR",
            payer_id="UHC",
            procedure_code="72148",
            criterion_id="C2",
            criterion_text="Radiculopathy or neurological symptoms must be present including numbness tingling or motor weakness",
            section="indications",
        ),
        PolicyChunk(
            policy_id="UHC_MRI_LUMBAR",
            payer_id="UHC",
            procedure_code="72148",
            criterion_id="C3",
            criterion_text="Red flag symptoms such as cauda equina syndrome bowel or bladder dysfunction require urgent imaging",
            section="indications",
        ),
    ]

    logic_tree = {
        "operator": "AND",
        "criteria": [
            {"id": "C1", "text": "Conservative treatment >= 6 weeks"},
            {"operator": "OR", "criteria": [
                {"id": "C2", "text": "Neurological symptoms present"},
                {"id": "C3", "text": "Red flag symptoms"},
            ]},
        ],
    }

    index_policy(
        tmp_db, "UHC_MRI_LUMBAR", "UHC", "72148", chunks,
        policy_name="UHC Lumbar MRI Policy",
        logic_tree=logic_tree,
    )
    return tmp_db


# ── Indexer tests ──────────────────────────────────────────────────────────

class TestIndexer:
    def test_init_creates_tables(self, tmp_db):
        tables = tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "policy_meta" in table_names
        assert "policy_chunks" in table_names

    def test_index_policy_stores_metadata(self, indexed_db):
        row = indexed_db.execute(
            "SELECT * FROM policy_meta WHERE policy_id = 'UHC_MRI_LUMBAR'"
        ).fetchone()
        assert row is not None

    def test_index_policy_stores_chunks(self, indexed_db):
        rows = indexed_db.execute(
            "SELECT COUNT(*) FROM policy_chunks WHERE policy_id = 'UHC_MRI_LUMBAR'"
        ).fetchone()
        assert rows[0] == 3

    def test_chunk_policy_text_numbered(self):
        text = """1. Patient must have tried conservative treatment for 6 weeks.
2. Neurological symptoms such as radiculopathy must be documented.
3. Prior imaging must be referenced if available."""
        chunks = chunk_policy_text(text, "TEST_POLICY", "TEST", "72148")
        assert len(chunks) >= 2
        assert all(c.policy_id == "TEST_POLICY" for c in chunks)

    def test_chunk_policy_text_paragraphs(self):
        text = """This is the first paragraph about coverage indications. It describes when MRI is appropriate.

This is the second paragraph about documentation requirements. It specifies what records are needed.

This is the third paragraph about exclusions. It lists when MRI is not covered."""
        chunks = chunk_policy_text(text, "P1", "UHC", "72148")
        assert len(chunks) >= 2

    def test_reindex_replaces_old_chunks(self, indexed_db):
        new_chunks = [
            PolicyChunk("UHC_MRI_LUMBAR", "UHC", "72148", "C1", "Updated criterion", "indications"),
        ]
        index_policy(indexed_db, "UHC_MRI_LUMBAR", "UHC", "72148", new_chunks)
        rows = indexed_db.execute(
            "SELECT COUNT(*) FROM policy_chunks WHERE policy_id = 'UHC_MRI_LUMBAR'"
        ).fetchone()
        assert rows[0] == 1

    def test_bulk_index_from_directory(self, tmp_db):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_file = Path(tmpdir) / "UHC_CT_ABDOMEN.txt"
            policy_file.write_text(
                "1. Abdominal pain lasting more than 2 weeks.\n"
                "2. Prior ultrasound was inconclusive.\n"
                "3. Patient has not had CT in the past 6 months."
            )
            count = bulk_index_from_directory(tmpdir, tmp_db)
            assert count >= 2


# ── Searcher tests ─────────────────────────────────────────────────────────

class TestSearcher:
    def test_search_returns_results(self, indexed_db):
        results = search_policies("72148", "UHC", conn=indexed_db)
        assert len(results) > 0
        assert results[0].payer_id == "UHC"

    def test_search_with_clinical_keywords(self, indexed_db):
        results = search_policies(
            "72148", "UHC",
            clinical_keywords=["radiculopathy", "numbness"],
            conn=indexed_db,
        )
        assert len(results) > 0

    def test_search_respects_top_k(self, indexed_db):
        results = search_policies("72148", "UHC", top_k=2, conn=indexed_db)
        assert len(results) <= 2

    def test_search_wrong_payer_falls_back(self, indexed_db):
        results = search_policies("72148", "NONEXISTENT", conn=indexed_db)
        # Should fall back to unfiltered search
        assert len(results) >= 0

    def test_get_logic_tree(self, indexed_db):
        tree = get_policy_logic_tree("UHC_MRI_LUMBAR", conn=indexed_db)
        assert tree is not None
        assert tree["operator"] == "AND"
        assert len(tree["criteria"]) == 2

    def test_get_logic_tree_missing(self, indexed_db):
        tree = get_policy_logic_tree("NONEXISTENT", conn=indexed_db)
        assert tree is None

    def test_get_all_criteria(self, indexed_db):
        results = get_all_criteria_for_policy("UHC_MRI_LUMBAR", conn=indexed_db)
        assert len(results) == 3

    def test_build_context_block_formats(self, indexed_db):
        results = search_policies("72148", "UHC", conn=indexed_db)
        block = build_context_block(results)
        assert "[C" in block
        assert "---" in block

    def test_build_context_block_empty(self):
        block = build_context_block([])
        assert "No matching" in block


# ── Public rates tests ─────────────────────────────────────────────────────

class TestPublicRates:
    def test_seed_kff_rates(self, tmp_rates_db):
        n = seed_kff_base_rates(tmp_rates_db)
        assert n > 0

        rows = tmp_rates_db.execute(
            "SELECT COUNT(*) FROM payer_denial_rates"
        ).fetchone()
        assert rows[0] >= 6  # at least 6 major payers

    def test_get_payer_denial_rate(self, tmp_rates_db):
        seed_kff_base_rates(tmp_rates_db)
        rate = get_payer_denial_rate("UHC", tmp_rates_db)
        assert abs(rate - 0.128) < 0.001

    def test_get_unknown_payer_returns_default(self, tmp_rates_db):
        seed_kff_base_rates(tmp_rates_db)
        rate = get_payer_denial_rate("NONEXISTENT", tmp_rates_db)
        assert rate == 0.077  # national average fallback

    def test_get_procedure_approval_rate(self, tmp_rates_db):
        seed_kff_base_rates(tmp_rates_db)
        rate = get_procedure_approval_rate("MRI", tmp_rates_db)
        assert abs(rate - 0.88) < 0.01

    def test_payer_disclosure_overrides_kff(self, tmp_rates_db):
        seed_kff_base_rates(tmp_rates_db)
        load_payer_disclosure(
            "UHC", "UnitedHealthcare", 85.0, 15.0,
            total_requests=10_000_000, conn=tmp_rates_db,
        )
        rate = get_payer_denial_rate("UHC", tmp_rates_db)
        # PAYER_DISCLOSURE should take priority over KFF
        assert abs(rate - 0.15) < 0.001

    def test_normalize_payer_name(self):
        assert _normalize_payer_name("UnitedHealthcare") == "UHC"
        assert _normalize_payer_name("Elevance Health") == "ELEVANCE"
        assert _normalize_payer_name("Aetna Inc") == "AETNA"
        assert _normalize_payer_name("Kaiser Permanente") == "KAISER"
        assert _normalize_payer_name("Cigna Group") == "CIGNA"
        assert _normalize_payer_name("WellCare Health") == "CENTENE"

    def test_procedure_rates_seeded(self, tmp_rates_db):
        seed_kff_base_rates(tmp_rates_db)
        rows = tmp_rates_db.execute(
            "SELECT COUNT(*) FROM procedure_approval_rates"
        ).fetchone()
        assert rows[0] >= 6
