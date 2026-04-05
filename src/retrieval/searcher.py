"""
AuthorizeAI — BM25 Policy Searcher
====================================
Queries the SQLite FTS5 index to retrieve the top-k most relevant policy
criterion chunks for a given procedure code, payer, and clinical keywords.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .indexer import DB_PATH


@dataclass
class SearchResult:
    policy_id: str
    payer_id: str
    procedure_code: str
    criterion_id: str
    criterion_text: str
    section: str
    bm25_score: float


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def search_policies(
    procedure_code: str,
    payer_id: str,
    clinical_keywords: list[str] | None = None,
    top_k: int = 5,
    conn: sqlite3.Connection | None = None,
) -> list[SearchResult]:
    """
    BM25 search over policy chunks.

    Builds a query combining the procedure code with optional clinical
    keywords (diagnosis terms, symptom names) for targeted retrieval.
    Returns top-k results ranked by BM25 relevance.
    """
    if conn is None:
        conn = get_connection()

    # Build FTS5 query: procedure code + payer filter + clinical terms
    query_parts = [procedure_code]
    if clinical_keywords:
        query_parts.extend(clinical_keywords[:10])  # cap keyword count

    fts_query = " OR ".join(query_parts)

    sql = """
        SELECT
            policy_id,
            payer_id,
            procedure_code,
            criterion_id,
            criterion_text,
            section,
            bm25(policy_chunks) AS score
        FROM policy_chunks
        WHERE policy_chunks MATCH ?
          AND payer_id = ?
        ORDER BY bm25(policy_chunks)
        LIMIT ?
    """

    try:
        rows = conn.execute(sql, (fts_query, payer_id, top_k)).fetchall()
    except sqlite3.OperationalError:
        # Fallback: broader search without payer filter
        sql_fallback = """
            SELECT
                policy_id, payer_id, procedure_code,
                criterion_id, criterion_text, section,
                bm25(policy_chunks) AS score
            FROM policy_chunks
            WHERE policy_chunks MATCH ?
            ORDER BY bm25(policy_chunks)
            LIMIT ?
        """
        rows = conn.execute(sql_fallback, (fts_query, top_k)).fetchall()

    return [
        SearchResult(
            policy_id=r["policy_id"],
            payer_id=r["payer_id"],
            procedure_code=r["procedure_code"],
            criterion_id=r["criterion_id"],
            criterion_text=r["criterion_text"],
            section=r["section"],
            bm25_score=r["score"],
        )
        for r in rows
    ]


def get_policy_logic_tree(
    policy_id: str,
    conn: sqlite3.Connection | None = None,
) -> dict | None:
    """
    Retrieve the pre-structured JSON logic tree for a policy, if available.
    Falls back to None if the policy was indexed without a logic tree.
    """
    if conn is None:
        conn = get_connection()

    row = conn.execute(
        "SELECT raw_json FROM policy_meta WHERE policy_id = ?",
        (policy_id,),
    ).fetchone()

    if row and row["raw_json"]:
        return json.loads(row["raw_json"])
    return None


def get_all_criteria_for_policy(
    policy_id: str,
    conn: sqlite3.Connection | None = None,
) -> list[SearchResult]:
    """Retrieve every criterion chunk for a specific policy (unranked)."""
    if conn is None:
        conn = get_connection()

    rows = conn.execute(
        "SELECT policy_id, payer_id, procedure_code, criterion_id, "
        "criterion_text, section FROM policy_chunks WHERE policy_id = ? "
        "ORDER BY criterion_id",
        (policy_id,),
    ).fetchall()

    return [
        SearchResult(
            policy_id=r["policy_id"],
            payer_id=r["payer_id"],
            procedure_code=r["procedure_code"],
            criterion_id=r["criterion_id"],
            criterion_text=r["criterion_text"],
            section=r["section"],
            bm25_score=0.0,
        )
        for r in rows
    ]


def build_context_block(results: list[SearchResult]) -> str:
    """
    Format search results into a single text block suitable for LLM context.
    Used by Agent 2 to provide policy context for criteria evaluation.
    """
    if not results:
        return "[No matching policy criteria found]"

    lines = []
    for r in results:
        lines.append(
            f"[{r.criterion_id} | {r.section}]\n{r.criterion_text}"
        )
    return "\n\n---\n\n".join(lines)
