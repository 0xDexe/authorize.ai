"""
AuthorizeAI — Public Data Ingestion
=====================================
Loads publicly available prior authorization outcome data from:
  1. CMS Part C/D Reporting Requirements PUF (contract-level PA rates)
  2. Payer-published PA metrics (March 2026 mandate, scraped from payer sites)
  3. CMS FFS Prior Auth statistics (procedure-level traditional Medicare rates)
  4. KFF/AMA published base rates (hardcoded from reports)

All sources are public domain — no credentials required except MIMIC-IV.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "base_rates.db"


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class PayerDenialRate:
    """Contract-level denial rate from CMS PUF or payer disclosures."""
    payer_id: str              # normalized: "UHC", "AETNA", etc.
    payer_name: str            # full name
    contract_id: str = ""      # CMS MA contract ID (e.g. H1234)
    year: int = 2024
    total_requests: int = 0
    total_denials: int = 0
    denial_rate: float = 0.0
    appeal_rate: float = 0.0   # % of denials appealed
    overturn_rate: float = 0.0 # % of appeals overturned
    source: str = ""           # "CMS_PUF", "PAYER_DISCLOSURE", "KFF"


@dataclass
class ProcedureApprovalRate:
    """Procedure-category-level approval rates from CMS FFS or literature."""
    procedure_category: str    # "MRI", "CT", "SPECIALTY_REFERRAL", etc.
    cpt_codes: list[str] = field(default_factory=list)
    approval_rate: float = 0.0
    denial_rate: float = 0.0
    sample_size: int = 0
    year: int = 2024
    source: str = ""


# ── Database setup ─────────────────────────────────────────────────────────

def init_base_rate_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create tables for public PA data."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS payer_denial_rates (
            payer_id       TEXT,
            payer_name     TEXT,
            contract_id    TEXT,
            year           INTEGER,
            total_requests INTEGER,
            total_denials  INTEGER,
            denial_rate    REAL,
            appeal_rate    REAL,
            overturn_rate  REAL,
            source         TEXT,
            PRIMARY KEY (payer_id, contract_id, year)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS procedure_approval_rates (
            procedure_category TEXT,
            cpt_codes          TEXT,
            approval_rate      REAL,
            denial_rate        REAL,
            sample_size        INTEGER,
            year               INTEGER,
            source             TEXT,
            PRIMARY KEY (procedure_category, year, source)
        )
    """)

    conn.commit()
    return conn


# ── CMS PUF loader ─────────────────────────────────────────────────────────

def load_cms_puf(
    puf_csv_path: str | Path,
    conn: sqlite3.Connection | None = None,
) -> int:
    """
    Load the CMS Part C Reporting Requirements Public Use File.

    The PUF is a CSV with contract-level PA data. Download from:
    https://www.cms.gov/data-research/cms-data/limited-data-set-lds-files/
    parts-c-d-reporting-requirements-limited-data-set

    Relevant columns vary by year but typically include:
    - contract_id, organization_name
    - prior_auth_requests, prior_auth_approvals, prior_auth_denials
    - appeals_filed, appeals_overturned

    Returns number of records loaded.
    """
    if conn is None:
        conn = init_base_rate_db()

    path = Path(puf_csv_path)
    count = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Column names vary — normalize common variants
            contract_id = (
                row.get("contract_id", "") or
                row.get("Contract ID", "") or
                row.get("CONTRACT_ID", "")
            ).strip()

            org_name = (
                row.get("organization_name", "") or
                row.get("Organization Name", "") or
                row.get("ORGANIZATION_NAME", "")
            ).strip()

            requests = _safe_int(
                row.get("prior_auth_requests", "") or
                row.get("Total PA Requests", "") or
                row.get("total_requests", "")
            )
            denials = _safe_int(
                row.get("prior_auth_denials", "") or
                row.get("Total PA Denials", "") or
                row.get("total_denials", "")
            )

            if requests == 0:
                continue

            payer_id = _normalize_payer_name(org_name)
            denial_rate = denials / requests if requests > 0 else 0.0

            appeals = _safe_int(
                row.get("appeals_filed", "") or
                row.get("Total Appeals", "") or "0"
            )
            overturned = _safe_int(
                row.get("appeals_overturned", "") or
                row.get("Appeals Overturned", "") or "0"
            )

            conn.execute(
                """INSERT OR REPLACE INTO payer_denial_rates
                   (payer_id, payer_name, contract_id, year, total_requests,
                    total_denials, denial_rate, appeal_rate, overturn_rate, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payer_id, org_name, contract_id, 2024,
                    requests, denials, round(denial_rate, 4),
                    round(appeals / denials, 4) if denials > 0 else 0.0,
                    round(overturned / appeals, 4) if appeals > 0 else 0.0,
                    "CMS_PUF",
                ),
            )
            count += 1

    conn.commit()
    return count


# ── Payer disclosure loader (March 2026 mandate) ───────────────────────────

def load_payer_disclosure(
    payer_id: str,
    payer_name: str,
    approval_pct: float,
    denial_pct: float,
    appeal_overturn_pct: float = 0.0,
    total_requests: int = 0,
    year: int = 2025,
    conn: sqlite3.Connection | None = None,
) -> None:
    """
    Manually enter payer-published PA metrics from their public websites.

    As of March 31, 2026, payers must post aggregated PA metrics annually.
    Since these aren't yet in a standardized downloadable format, you'll
    scrape or manually enter them from each payer's website.

    Example:
        load_payer_disclosure("UHC", "UnitedHealthcare", 87.2, 12.8, 80.7, 12000000)
    """
    if conn is None:
        conn = init_base_rate_db()

    denial_rate = denial_pct / 100.0
    total_denials = int(total_requests * denial_rate) if total_requests else 0

    conn.execute(
        """INSERT OR REPLACE INTO payer_denial_rates
           (payer_id, payer_name, contract_id, year, total_requests,
            total_denials, denial_rate, appeal_rate, overturn_rate, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            payer_id, payer_name, "AGGREGATE", year,
            total_requests, total_denials, round(denial_rate, 4),
            0.0, round(appeal_overturn_pct / 100.0, 4),
            "PAYER_DISCLOSURE",
        ),
    )
    conn.commit()


# ── KFF/AMA hardcoded base rates ──────────────────────────────────────────

def seed_kff_base_rates(conn: sqlite3.Connection | None = None) -> int:
    """
    Seed the database with published base rates from KFF analysis of
    CMS data (2024) and AMA physician survey data.

    These are aggregated, publicly available figures — no credentials needed.
    Sources:
      - KFF: "MA Insurers Made Nearly 53M PA Determinations in 2024"
      - AMA: "2023 AMA Prior Authorization Physician Survey"
      - CMS OIG: "MAO Denials of PA Requests" (2022 report)
    """
    if conn is None:
        conn = init_base_rate_db()

    # Payer-level rates from KFF 2024 analysis
    kff_payer_rates = [
        ("UHC", "UnitedHealthcare", 0.128, 53_000_000),
        ("ELEVANCE", "Elevance Health (Anthem)", 0.042, 53_000_000),
        ("HUMANA", "Humana", 0.058, 53_000_000),
        ("AETNA", "Aetna (CVS Health)", 0.119, 53_000_000),
        ("CENTENE", "Centene Corporation", 0.123, 53_000_000),
        ("KAISER", "Kaiser Permanente", 0.109, 53_000_000),
    ]

    for pid, pname, drate, total in kff_payer_rates:
        conn.execute(
            """INSERT OR REPLACE INTO payer_denial_rates
               (payer_id, payer_name, contract_id, year, total_requests,
                total_denials, denial_rate, appeal_rate, overturn_rate, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pid, pname, "KFF_AGG", 2024, total,
             int(total * drate), drate, 0.115, 0.807, "KFF"),
        )

    # Procedure-category rates from CMS FFS + radiology PA studies
    proc_rates = [
        ("MRI", ["72148", "70553", "75557"], 0.88, 2024, "CMS_FFS_OIG"),
        ("CT", ["74177", "71260"], 0.91, 2024, "CMS_FFS_OIG"),
        ("SPECIALTY_REFERRAL", ["99242", "99243"], 0.85, 2024, "AMA_SURVEY"),
        ("BRAND_DRUG", ["J0717", "J2357"], 0.78, 2024, "AMA_SURVEY"),
        ("SURGERY", [], 0.82, 2024, "CMS_FFS_OIG"),
        ("DME", [], 0.80, 2024, "CMS_FFS_OIG"),
        ("INPATIENT", [], 0.75, 2024, "CMS_FFS_OIG"),
        ("HOME_HEALTH", [], 0.77, 2024, "CMS_FFS_OIG"),
    ]

    for cat, cpts, approval, year, source in proc_rates:
        conn.execute(
            """INSERT OR REPLACE INTO procedure_approval_rates
               (procedure_category, cpt_codes, approval_rate, denial_rate,
                sample_size, year, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (cat, json.dumps(cpts), approval, round(1 - approval, 4),
             0, year, source),
        )

    conn.commit()
    return len(kff_payer_rates) + len(proc_rates)


# ── Query interface (used by features.py) ──────────────────────────────────

def get_payer_denial_rate(
    payer_id: str,
    conn: sqlite3.Connection | None = None,
) -> float:
    """
    Look up the best available denial rate for a payer.
    Priority: PAYER_DISCLOSURE > CMS_PUF > KFF
    Falls back to national average (0.077) if not found.
    """
    if conn is None:
        try:
            conn = sqlite3.connect(str(DB_PATH))
        except Exception:
            return 0.077

    row = conn.execute(
        """SELECT denial_rate FROM payer_denial_rates
           WHERE payer_id = ?
           ORDER BY
             CASE source
               WHEN 'PAYER_DISCLOSURE' THEN 1
               WHEN 'CMS_PUF' THEN 2
               WHEN 'KFF' THEN 3
               ELSE 4
             END,
             year DESC
           LIMIT 1""",
        (payer_id.upper(),),
    ).fetchone()

    return row[0] if row else 0.077


def get_procedure_approval_rate(
    procedure_category: str,
    conn: sqlite3.Connection | None = None,
) -> float:
    """Look up procedure-category approval rate. Falls back to 0.85."""
    if conn is None:
        try:
            conn = sqlite3.connect(str(DB_PATH))
        except Exception:
            return 0.85

    row = conn.execute(
        """SELECT approval_rate FROM procedure_approval_rates
           WHERE procedure_category = ?
           ORDER BY year DESC LIMIT 1""",
        (procedure_category.upper(),),
    ).fetchone()

    return row[0] if row else 0.85


# ── Connecticut / Vermont state data loader ────────────────────────────────

def load_state_insurer_report(
    csv_path: str | Path,
    state: str,
    year: int,
    conn: sqlite3.Connection | None = None,
) -> int:
    """
    Load state insurance department PA data (CT or VT report card format).

    Connecticut publishes annual insurer report cards since 2011.
    Vermont publishes PA data for insurers covering 2,000+ residents.

    Expected CSV columns: insurer_name, total_claims, denied_claims,
    denial_rate, appeals, appeals_overturned

    Download from:
      CT: portal.ct.gov/cid (Health Insurance section)
      VT: dfr.vermont.gov (Health Insurance section)
    """
    if conn is None:
        conn = init_base_rate_db()

    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("insurer_name", "").strip()
            if not name:
                continue

            payer_id = _normalize_payer_name(name)
            denial_rate = _safe_float(row.get("denial_rate", "0"))
            if denial_rate > 1:
                denial_rate /= 100.0  # handle percentage format

            conn.execute(
                """INSERT OR REPLACE INTO payer_denial_rates
                   (payer_id, payer_name, contract_id, year,
                    total_requests, total_denials, denial_rate,
                    appeal_rate, overturn_rate, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payer_id, name, f"STATE_{state.upper()}", year,
                    _safe_int(row.get("total_claims", "0")),
                    _safe_int(row.get("denied_claims", "0")),
                    round(denial_rate, 4),
                    0.0, 0.0,
                    f"STATE_{state.upper()}",
                ),
            )
            count += 1

    conn.commit()
    return count


# ── Helpers ────────────────────────────────────────────────────────────────

def _normalize_payer_name(name: str) -> str:
    """Map insurer names to canonical payer IDs."""
    upper = name.upper()
    if "UNITED" in upper or "UHC" in upper:
        return "UHC"
    if "AETNA" in upper or "CVS" in upper:
        return "AETNA"
    if "CIGNA" in upper or "EVERNORTH" in upper:
        return "CIGNA"
    if "HUMANA" in upper:
        return "HUMANA"
    if "ANTHEM" in upper or "ELEVANCE" in upper or "WELLPOINT" in upper:
        return "ELEVANCE"
    if "CENTENE" in upper or "WELLCARE" in upper:
        return "CENTENE"
    if "KAISER" in upper:
        return "KAISER"
    if "BLUE CROSS" in upper or "BCBS" in upper:
        return "BCBS"
    if "MOLINA" in upper:
        return "MOLINA"
    return name.upper().replace(" ", "_")[:20]


def _safe_int(val: str) -> int:
    try:
        return int(val.replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0


def _safe_float(val: str) -> float:
    try:
        return float(val.replace(",", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return 0.0
