"""
AuthorizeAI — Policy Document Indexer
======================================
Ingests payer coverage policy documents, chunks them at the criterion level,
and indexes them into a SQLite FTS5 full-text search table with BM25 ranking.
"""

import json
import re
import sqlite3
from pathlib import Path
from dataclasses import dataclass


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "policy_index.db"


@dataclass
class PolicyChunk:
    policy_id: str        # e.g. "UHC_MRI_LUMBAR"
    payer_id: str         # e.g. "UHC"
    procedure_code: str   # CPT code this policy covers
    criterion_id: str     # e.g. "C1", "C2"
    criterion_text: str   # the actual policy criterion text
    section: str          # e.g. "indications", "contraindications", "documentation"
    metadata: dict | None = None


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create the FTS5 table and metadata table if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")

    # Metadata table for structured lookups
    conn.execute("""
        CREATE TABLE IF NOT EXISTS policy_meta (
            policy_id   TEXT PRIMARY KEY,
            payer_id    TEXT NOT NULL,
            procedure_code TEXT NOT NULL,
            policy_name TEXT,
            source_url  TEXT,
            raw_json    TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # FTS5 virtual table for full-text BM25 retrieval
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS policy_chunks USING fts5(
            policy_id,
            payer_id,
            procedure_code,
            criterion_id,
            criterion_text,
            section,
            tokenize='porter unicode61'
        )
    """)

    conn.commit()
    return conn


def index_chunk(conn: sqlite3.Connection, chunk: PolicyChunk) -> None:
    """Insert a single policy chunk into the FTS5 index."""
    conn.execute(
        "INSERT INTO policy_chunks (policy_id, payer_id, procedure_code, "
        "criterion_id, criterion_text, section) VALUES (?, ?, ?, ?, ?, ?)",
        (chunk.policy_id, chunk.payer_id, chunk.procedure_code,
         chunk.criterion_id, chunk.criterion_text, chunk.section),
    )


def index_policy(
    conn: sqlite3.Connection,
    policy_id: str,
    payer_id: str,
    procedure_code: str,
    chunks: list[PolicyChunk],
    policy_name: str = "",
    source_url: str = "",
    logic_tree: dict | None = None,
) -> int:
    """
    Index a complete policy: store metadata + all criterion-level chunks.
    Returns the number of chunks indexed.
    """
    # Upsert metadata
    conn.execute(
        "INSERT OR REPLACE INTO policy_meta "
        "(policy_id, payer_id, procedure_code, policy_name, source_url, raw_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (policy_id, payer_id, procedure_code, policy_name, source_url,
         json.dumps(logic_tree) if logic_tree else None),
    )

    # Delete old chunks for this policy, then re-insert
    conn.execute(
        "DELETE FROM policy_chunks WHERE policy_id = ?", (policy_id,)
    )

    for chunk in chunks:
        index_chunk(conn, chunk)

    conn.commit()
    return len(chunks)


def chunk_policy_text(
    raw_text: str,
    policy_id: str,
    payer_id: str,
    procedure_code: str,
) -> list[PolicyChunk]:
    """
    Split raw policy text into criterion-level chunks.

    Heuristic: split on numbered criteria patterns like "1.", "(1)", "(a)",
    or header-like patterns ("INDICATIONS:", "DOCUMENTATION REQUIREMENTS:").
    Each resulting segment becomes one retrievable chunk.
    """
    chunks: list[PolicyChunk] = []

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", raw_text.strip())

    # Try to split on numbered/lettered criteria
    pattern = r"(?:^|\n)(?:\d+\.\s+|\(\d+\)\s*|\([a-z]\)\s*|[A-Z][A-Z ]{3,}:)"
    segments = re.split(pattern, text)
    headers = re.findall(pattern, text)

    if len(segments) <= 1:
        # Fallback: split on double newlines (paragraph-level)
        segments = text.split("\n\n")
        headers = [f"P{i}" for i in range(len(segments))]

    section = "general"
    for i, seg in enumerate(segments):
        seg = seg.strip()
        if not seg or len(seg) < 20:
            continue

        # Detect section from header keywords
        header = headers[i].strip() if i < len(headers) else ""
        lower_header = header.lower()
        if any(k in lower_header for k in ["indication", "covered when", "criteria"]):
            section = "indications"
        elif any(k in lower_header for k in ["contraindic", "exclusion", "not covered"]):
            section = "contraindications"
        elif any(k in lower_header for k in ["document", "required", "submission"]):
            section = "documentation"

        chunks.append(PolicyChunk(
            policy_id=policy_id,
            payer_id=payer_id,
            procedure_code=procedure_code,
            criterion_id=f"C{len(chunks) + 1}",
            criterion_text=seg[:2000],  # cap chunk size
            section=section,
        ))

    return chunks


def bulk_index_from_directory(
    policy_dir: str | Path,
    conn: sqlite3.Connection | None = None,
) -> int:
    """
    Index all .txt or .json policy files from a directory.
    JSON files should have: policy_id, payer_id, procedure_code, text, and
    optionally logic_tree.
    Returns total chunks indexed.
    """
    policy_dir = Path(policy_dir)
    if conn is None:
        conn = init_db()

    total = 0
    for fpath in sorted(policy_dir.glob("*")):
        if fpath.suffix == ".json":
            data = json.loads(fpath.read_text())
            chunks = chunk_policy_text(
                data["text"], data["policy_id"],
                data["payer_id"], data["procedure_code"],
            )
            total += index_policy(
                conn, data["policy_id"], data["payer_id"],
                data["procedure_code"], chunks,
                policy_name=data.get("policy_name", ""),
                source_url=data.get("source_url", ""),
                logic_tree=data.get("logic_tree"),
            )
        elif fpath.suffix == ".txt":
            # Derive IDs from filename: PAYER_PROCEDURE.txt
            stem = fpath.stem
            parts = stem.split("_", 1)
            payer_id = parts[0] if len(parts) > 1 else "UNKNOWN"
            proc = parts[1] if len(parts) > 1 else stem
            policy_id = stem
            chunks = chunk_policy_text(
                fpath.read_text(), policy_id, payer_id, proc,
            )
            total += index_policy(conn, policy_id, payer_id, proc, chunks)

    return total
