from __future__ import annotations

"""
AuthorizeAI — MIMIC-IV Data Loader
====================================
Reads MIMIC-IV hosp module (structured EHR) and MIMIC-IV-Note module
(discharge summaries, radiology reports) to assemble real clinical cases
for pipeline testing, evaluation, and model training.

Expected directory layout (after PhysioNet download):
    data/mimic-iv/
    ├── hosp/
    │   ├── patients.csv.gz
    │   ├── admissions.csv.gz
    │   ├── diagnoses_icd.csv.gz
    │   ├── procedures_icd.csv.gz
    │   ├── prescriptions.csv.gz
    │   └── labevents.csv.gz        (optional — large file)
    └── note/
        ├── discharge.csv.gz
        ├── discharge_detail.csv.gz
        ├── radiology.csv.gz
        └── radiology_detail.csv.gz
"""
"""
AuthorizeAI — MIMIC-IV Loader
================================
Reads MIMIC-IV CSVs from a local PhysioNet download and assembles
MIMICCase objects for the pipeline.

Loophole fixes applied:
  #1 OOM: load_all_cases now streams discharge notes with early stop,
         then filters all downstream tables by the collected hadm_ids.
         Previously, every table was read fully into memory before
         `limit` was applied — OOM on 16GB laptops.
  #2 Payer proxy: MIMIC's insurance field is de-identified. Values are
         now mapped via PAYER_PROXY_MAP to explicit _PROXY-suffixed labels
         and a one-shot UserWarning is emitted on first use.
  #3 CPT selection: radiology_cpt_codes is now deduped + sorted before
         picking [0], making the choice deterministic across runs.
  #4 Retrospective notes: case_to_pipeline_input emits a one-shot
         UserWarning explaining that discharge summaries are not
         prospective PA documentation.

"""


import csv
import gzip
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ── Default paths ──────────────────────────────────────────────────────────
# This file lives at <project>/data/mimic_loader.py, so .parent is data/.
DEFAULT_MIMIC_DIR = Path(__file__).resolve().parent / "mimic-iv"


# ── Payer proxy mapping (loophole #2) ──────────────────────────────────────
# MIMIC's admissions.insurance column is NOT a real payer identifier.
# BIDMC de-identified it. You CANNOT recover "UHC" or "Aetna" from MIMIC.
# These proxy values are explicitly _PROXY-suffixed so downstream code
# cannot accidentally mistake them for real payer data.
PAYER_PROXY_MAP = {
    "medicare":   "CMS_MEDICARE_PROXY",
    "medicaid":   "CMS_MEDICAID_PROXY",
    "private":    "COMMERCIAL_GENERIC_PROXY",
    "other":      "COMMERCIAL_GENERIC_PROXY",
    "self pay":   "SELF_PAY_PROXY",
    "self-pay":   "SELF_PAY_PROXY",
    "government": "GOVERNMENT_OTHER_PROXY",
}
_DEFAULT_PAYER_PROXY = "UNKNOWN_PROXY"

# One-shot warning flags
_payer_proxy_warning_shown = False
_retrospective_warning_shown = False


def _warn_payer_proxy_once() -> None:
    global _payer_proxy_warning_shown
    if not _payer_proxy_warning_shown:
        warnings.warn(
            "MIMIC does not contain real payer identifiers. payer_proxy "
            "values (CMS_MEDICARE_PROXY, COMMERCIAL_GENERIC_PROXY, etc.) "
            "are synthetic and must NOT be interpreted as real payer data. "
            "Any Agent 2 policy match against these values is for pipeline "
            "validation only, not a real payer-policy evaluation.",
            UserWarning,
            stacklevel=3,
        )
        _payer_proxy_warning_shown = True


def _warn_retrospective_once() -> None:
    global _retrospective_warning_shown
    if not _retrospective_warning_shown:
        warnings.warn(
            "MIMIC discharge summaries are RETROSPECTIVE end-of-stay "
            "narratives, not PROSPECTIVE PA request documentation. "
            "Agent 1 will conflate in-admission treatments with truly "
            "prior treatments. Treat extraction/pipeline evaluation "
            "results as an upper bound, not a faithful reflection of "
            "real-world PA performance.",
            UserWarning,
            stacklevel=3,
        )
        _retrospective_warning_shown = True


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class MIMICPatient:
    subject_id: str
    gender: str = ""
    anchor_age: int = 0
    anchor_year: int = 0


@dataclass
class MIMICAdmission:
    hadm_id: str
    subject_id: str
    admittime: str = ""
    dischtime: str = ""
    admission_type: str = ""
    insurance: str = ""       # de-identified payer proxy — see PAYER_PROXY_MAP
    race: str = ""


@dataclass
class MIMICDiagnosis:
    hadm_id: str
    subject_id: str
    icd_code: str = ""
    icd_version: int = 10
    seq_num: int = 0


@dataclass
class MIMICProcedure:
    hadm_id: str
    subject_id: str
    icd_code: str = ""
    icd_version: int = 10
    seq_num: int = 0


@dataclass
class MIMICPrescription:
    hadm_id: str
    subject_id: str
    drug: str = ""
    drug_type: str = ""
    route: str = ""
    dose_val_rx: str = ""
    dose_unit_rx: str = ""
    starttime: str = ""
    stoptime: str = ""


@dataclass
class MIMICDischargeNote:
    note_id: str
    subject_id: str
    hadm_id: str
    charttime: str = ""
    text: str = ""


@dataclass
class MIMICRadiologyReport:
    note_id: str
    subject_id: str
    hadm_id: str
    charttime: str = ""
    text: str = ""


@dataclass
class MIMICRadiologyDetail:
    note_id: str
    subject_id: str
    field_name: str = ""
    field_value: str = ""


@dataclass
class MIMICCase:
    """
    A fully assembled clinical case combining structured data with
    free-text notes for a single hospitalization.
    """
    subject_id: str
    hadm_id: str
    patient: MIMICPatient | None = None
    admission: MIMICAdmission | None = None
    diagnoses: list[MIMICDiagnosis] = field(default_factory=list)
    procedures: list[MIMICProcedure] = field(default_factory=list)
    prescriptions: list[MIMICPrescription] = field(default_factory=list)
    discharge_notes: list[MIMICDischargeNote] = field(default_factory=list)
    radiology_reports: list[MIMICRadiologyReport] = field(default_factory=list)
    radiology_details: list[MIMICRadiologyDetail] = field(default_factory=list)

    @property
    def primary_discharge_text(self) -> str:
        if not self.discharge_notes:
            return ""
        return max(self.discharge_notes, key=lambda n: len(n.text)).text

    @property
    def icd10_codes(self) -> list[str]:
        return [
            d.icd_code for d in sorted(self.diagnoses, key=lambda d: d.seq_num)
            if d.icd_version == 10
        ]

    @property
    def icd9_codes(self) -> list[str]:
        return [
            d.icd_code for d in sorted(self.diagnoses, key=lambda d: d.seq_num)
            if d.icd_version == 9
        ]

    @property
    def drug_list(self) -> list[str]:
        seen = set()
        drugs = []
        for p in self.prescriptions:
            name = p.drug.strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                drugs.append(name)
        return drugs

    @property
    def radiology_cpt_codes(self) -> list[str]:
        """CPT codes from radiology studies. Deduped + sorted for determinism."""
        codes = {
            d.field_value for d in self.radiology_details
            if d.field_name == "cpt_code" and d.field_value
        }
        return sorted(codes)

    @property
    def payer_proxy(self) -> str:
        """
        Map MIMIC insurance field to a proxy payer ID.

        WARNING: This is NOT a real payer. MIMIC's insurance field is
        de-identified. All return values are _PROXY-suffixed to signal
        this to downstream code. First call emits a UserWarning.
        """
        _warn_payer_proxy_once()
        if not self.admission:
            return _DEFAULT_PAYER_PROXY
        ins = self.admission.insurance.lower().strip()
        if not ins:
            return _DEFAULT_PAYER_PROXY
        for key, proxy in PAYER_PROXY_MAP.items():
            if key in ins:
                return proxy
        return "COMMERCIAL_GENERIC_PROXY"


# ── CSV readers ────────────────────────────────────────────────────────────

def _read_gz_csv(filepath: Path) -> Iterator[dict]:
    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        yield from reader


def _read_csv(filepath: Path) -> Iterator[dict]:
    if filepath.suffix == ".gz":
        return _read_gz_csv(filepath)
    else:
        f = open(filepath, "r", encoding="utf-8", errors="replace")
        reader = csv.DictReader(f)
        return reader


# ── Loaders for individual tables ──────────────────────────────────────────
# Big tables now accept optional `target_hadm_ids` (or `target_note_ids`)
# set for row-level filtering during streaming. Pass None to load everything
# (preserves the old behavior for anyone who calls these directly).

def load_patients(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict[str, MIMICPatient]:
    path = _find_file(mimic_dir / "hosp", "patients")
    patients = {}
    for row in _read_csv(path):
        sid = row["subject_id"]
        patients[sid] = MIMICPatient(
            subject_id=sid,
            gender=row.get("gender", ""),
            anchor_age=int(row.get("anchor_age", 0) or 0),
            anchor_year=int(row.get("anchor_year", 0) or 0),
        )
    return patients


def load_admissions(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict[str, list[MIMICAdmission]]:
    path = _find_file(mimic_dir / "hosp", "admissions")
    admissions: dict[str, list[MIMICAdmission]] = defaultdict(list)
    for row in _read_csv(path):
        sid = row["subject_id"]
        admissions[sid].append(MIMICAdmission(
            hadm_id=row["hadm_id"],
            subject_id=sid,
            admittime=row.get("admittime", ""),
            dischtime=row.get("dischtime", ""),
            admission_type=row.get("admission_type", ""),
            insurance=row.get("insurance", ""),
            race=row.get("race", ""),
        ))
    return dict(admissions)


def load_diagnoses(
    mimic_dir: Path = DEFAULT_MIMIC_DIR,
    target_hadm_ids: set[str] | None = None,
) -> dict[str, list[MIMICDiagnosis]]:
    path = _find_file(mimic_dir / "hosp", "diagnoses_icd")
    diag: dict[str, list[MIMICDiagnosis]] = defaultdict(list)
    for row in _read_csv(path):
        hid = row["hadm_id"]
        if target_hadm_ids is not None and hid not in target_hadm_ids:
            continue
        diag[hid].append(MIMICDiagnosis(
            hadm_id=hid,
            subject_id=row["subject_id"],
            icd_code=row.get("icd_code", ""),
            icd_version=int(row.get("icd_version", 10) or 10),
            seq_num=int(row.get("seq_num", 0) or 0),
        ))
    return dict(diag)


def load_procedures(
    mimic_dir: Path = DEFAULT_MIMIC_DIR,
    target_hadm_ids: set[str] | None = None,
) -> dict[str, list[MIMICProcedure]]:
    path = _find_file(mimic_dir / "hosp", "procedures_icd")
    procs: dict[str, list[MIMICProcedure]] = defaultdict(list)
    for row in _read_csv(path):
        hid = row["hadm_id"]
        if target_hadm_ids is not None and hid not in target_hadm_ids:
            continue
        procs[hid].append(MIMICProcedure(
            hadm_id=hid,
            subject_id=row["subject_id"],
            icd_code=row.get("icd_code", ""),
            icd_version=int(row.get("icd_version", 10) or 10),
            seq_num=int(row.get("seq_num", 0) or 0),
        ))
    return dict(procs)


def load_prescriptions(
    mimic_dir: Path = DEFAULT_MIMIC_DIR,
    target_hadm_ids: set[str] | None = None,
) -> dict[str, list[MIMICPrescription]]:
    path = _find_file(mimic_dir / "hosp", "prescriptions")
    meds: dict[str, list[MIMICPrescription]] = defaultdict(list)
    for row in _read_csv(path):
        hid = row.get("hadm_id", "")
        if not hid:
            continue
        if target_hadm_ids is not None and hid not in target_hadm_ids:
            continue
        meds[hid].append(MIMICPrescription(
            hadm_id=hid,
            subject_id=row["subject_id"],
            drug=row.get("drug", ""),
            drug_type=row.get("drug_type", ""),
            route=row.get("route", ""),
            dose_val_rx=row.get("dose_val_rx", ""),
            dose_unit_rx=row.get("dose_unit_rx", ""),
            starttime=row.get("starttime", ""),
            stoptime=row.get("stoptime", ""),
        ))
    return dict(meds)


def load_discharge_notes(
    mimic_dir: Path = DEFAULT_MIMIC_DIR,
    limit: int | None = None,
    target_hadm_ids: set[str] | None = None,
) -> dict[str, list[MIMICDischargeNote]]:
    """
    Stream discharge summaries. If `limit` is set, stop after that many
    distinct hadm_ids have been collected (early termination — critical
    for avoiding OOM on the full note.discharge.csv.gz file).
    """
    path = _find_file(mimic_dir / "note", "discharge")
    notes: dict[str, list[MIMICDischargeNote]] = defaultdict(list)
    for row in _read_csv(path):
        hid = row.get("hadm_id", "")
        if not hid:
            continue
        if target_hadm_ids is not None and hid not in target_hadm_ids:
            continue
        notes[hid].append(MIMICDischargeNote(
            note_id=row.get("note_id", ""),
            subject_id=row["subject_id"],
            hadm_id=hid,
            charttime=row.get("charttime", ""),
            text=row.get("text", ""),
        ))
        if limit is not None and len(notes) >= limit:
            break
    return dict(notes)


def load_radiology_reports(
    mimic_dir: Path = DEFAULT_MIMIC_DIR,
    target_hadm_ids: set[str] | None = None,
) -> dict[str, list[MIMICRadiologyReport]]:
    path = _find_file(mimic_dir / "note", "radiology")
    reports: dict[str, list[MIMICRadiologyReport]] = defaultdict(list)
    for row in _read_csv(path):
        hid = row.get("hadm_id", "")
        if not hid:
            continue
        if target_hadm_ids is not None and hid not in target_hadm_ids:
            continue
        reports[hid].append(MIMICRadiologyReport(
            note_id=row.get("note_id", ""),
            subject_id=row["subject_id"],
            hadm_id=hid,
            charttime=row.get("charttime", ""),
            text=row.get("text", ""),
        ))
    return dict(reports)


def load_radiology_details(
    mimic_dir: Path = DEFAULT_MIMIC_DIR,
    target_note_ids: set[str] | None = None,
) -> dict[str, list[MIMICRadiologyDetail]]:
    path = _find_file(mimic_dir / "note", "radiology_detail")
    details: dict[str, list[MIMICRadiologyDetail]] = defaultdict(list)
    for row in _read_csv(path):
        nid = row.get("note_id", "")
        if target_note_ids is not None and nid not in target_note_ids:
            continue
        details[nid].append(MIMICRadiologyDetail(
            note_id=nid,
            subject_id=row.get("subject_id", ""),
            field_name=row.get("field_name", ""),
            field_value=row.get("field_value", ""),
        ))
    return dict(details)


# ── Case assembler ─────────────────────────────────────────────────────────

def assemble_case(
    hadm_id: str,
    patients: dict[str, MIMICPatient],
    admissions_by_subject: dict[str, list[MIMICAdmission]],
    diagnoses: dict[str, list[MIMICDiagnosis]],
    procedures: dict[str, list[MIMICProcedure]],
    prescriptions: dict[str, list[MIMICPrescription]],
    discharge_notes: dict[str, list[MIMICDischargeNote]],
    radiology_reports: dict[str, list[MIMICRadiologyReport]] | None = None,
    radiology_details: dict[str, list[MIMICRadiologyDetail]] | None = None,
) -> MIMICCase | None:
    notes = discharge_notes.get(hadm_id, [])
    if not notes:
        return None

    subject_id = notes[0].subject_id
    patient = patients.get(subject_id)

    admission = None
    for adm in admissions_by_subject.get(subject_id, []):
        if adm.hadm_id == hadm_id:
            admission = adm
            break

    rad_details = []
    if radiology_details:
        for report in (radiology_reports or {}).get(hadm_id, []):
            rad_details.extend(radiology_details.get(report.note_id, []))

    return MIMICCase(
        subject_id=subject_id,
        hadm_id=hadm_id,
        patient=patient,
        admission=admission,
        diagnoses=diagnoses.get(hadm_id, []),
        procedures=procedures.get(hadm_id, []),
        prescriptions=prescriptions.get(hadm_id, []),
        discharge_notes=notes,
        radiology_reports=(radiology_reports or {}).get(hadm_id, []),
        radiology_details=rad_details,
    )


def load_all_cases(
    mimic_dir: Path = DEFAULT_MIMIC_DIR,
    limit: int | None = None,
    require_discharge_note: bool = True,
    require_radiology: bool = False,
) -> list[MIMICCase]:
    """
    Load and assemble clinical cases from MIMIC-IV using streaming reads
    with target-set filtering to bound memory.

    Strategy (loophole #1 fix):
      1. Load small tables (patients, admissions) in full.
      2. Stream discharge notes with early stop at `limit * 3` (oversample
         so the downstream require_radiology filter can still hit `limit`
         cases).
      3. Use the collected hadm_ids as a filter set for every large table.
      4. Assemble cases and stop once `limit` is reached.

    Memory footprint is now bounded by `limit`, not by MIMIC table sizes.
    """
    print("Loading MIMIC-IV tables (streaming mode)...")

    # Small tables — load in full
    patients = load_patients(mimic_dir)
    print(f"  patients: {len(patients)}")

    admissions = load_admissions(mimic_dir)
    print(f"  admissions: {sum(len(v) for v in admissions.values())}")

    # Discharge notes first, with early stop — this anchors the cohort.
    # Oversample 3x if we have a radiology requirement, since many cases
    # won't have radiology and will be filtered out downstream.
    if limit is not None:
        note_load_limit = limit * 3 if require_radiology else limit * 2
    else:
        note_load_limit = None

    discharge_notes = load_discharge_notes(mimic_dir, limit=note_load_limit)
    print(f"  discharge notes loaded: {len(discharge_notes)} hadm_ids")

    target_hadm_ids: set[str] = set(discharge_notes.keys())
    print(f"  target cohort: {len(target_hadm_ids)} hadm_ids")

    # Big tables — filter to target cohort during streaming
    diagnoses = load_diagnoses(mimic_dir, target_hadm_ids=target_hadm_ids)
    print(f"  diagnoses (filtered): {sum(len(v) for v in diagnoses.values())}")

    procedures = load_procedures(mimic_dir, target_hadm_ids=target_hadm_ids)
    print(f"  procedures (filtered): {sum(len(v) for v in procedures.values())}")

    prescriptions = load_prescriptions(mimic_dir, target_hadm_ids=target_hadm_ids)
    print(f"  prescriptions (filtered): {sum(len(v) for v in prescriptions.values())}")

    radiology_reports = None
    radiology_details = None
    rad_path = mimic_dir / "note"
    if (rad_path / "radiology.csv.gz").exists() or (rad_path / "radiology.csv").exists():
        radiology_reports = load_radiology_reports(
            mimic_dir, target_hadm_ids=target_hadm_ids,
        )
        print(f"  radiology reports (filtered): {sum(len(v) for v in radiology_reports.values())}")

        # Collect note_ids from the filtered reports for detail filtering
        target_note_ids: set[str] = set()
        for reports in radiology_reports.values():
            for r in reports:
                if r.note_id:
                    target_note_ids.add(r.note_id)

        try:
            radiology_details = load_radiology_details(
                mimic_dir, target_note_ids=target_note_ids,
            )
            print(f"  radiology details (filtered): {sum(len(v) for v in radiology_details.values())}")
        except FileNotFoundError:
            pass

    # Assemble cases
    print("\nAssembling cases...")
    cases = []
    for hadm_id in discharge_notes.keys():
        case = assemble_case(
            hadm_id, patients, admissions, diagnoses,
            procedures, prescriptions, discharge_notes,
            radiology_reports, radiology_details,
        )
        if case is None:
            continue
        if require_radiology and not case.radiology_reports:
            continue
        cases.append(case)
        if limit and len(cases) >= limit:
            break

    print(f"  Assembled {len(cases)} cases")
    return cases


# ── Filtering helpers ──────────────────────────────────────────────────────

def filter_cases_by_procedure_codes(
    cases: list[MIMICCase],
    target_cpt_codes: list[str] | None = None,
    target_icd_proc_codes: list[str] | None = None,
) -> list[MIMICCase]:
    filtered = []
    for case in cases:
        if target_cpt_codes:
            if any(c in target_cpt_codes for c in case.radiology_cpt_codes):
                filtered.append(case)
                continue
        if target_icd_proc_codes:
            if any(p.icd_code in target_icd_proc_codes for p in case.procedures):
                filtered.append(case)
                continue
    return filtered


def filter_cases_by_diagnosis(
    cases: list[MIMICCase],
    icd_prefixes: list[str],
) -> list[MIMICCase]:
    filtered = []
    for case in cases:
        all_codes = case.icd10_codes + case.icd9_codes
        if any(
            code.startswith(prefix)
            for code in all_codes
            for prefix in icd_prefixes
        ):
            filtered.append(case)
    return filtered


def filter_cases_with_imaging(cases: list[MIMICCase]) -> list[MIMICCase]:
    return [c for c in cases if c.radiology_reports]


# ── Pipeline integration ──────────────────────────────────────────────────

def case_to_pipeline_input(
    case: MIMICCase,
    procedure_code: str = "",
) -> dict:
    """
    Convert a MIMICCase into kwargs for run_pipeline().

    WARNING (loophole #4): MIMIC discharge summaries are retrospective
    end-of-stay narratives, not prospective PA request documentation.
    This function emits a one-shot UserWarning on first call to make
    sure callers know what they're feeding into Agent 1.

    CPT selection (loophole #3): if no procedure_code is explicitly
    provided, the first CPT from radiology_cpt_codes is used. That list
    is now deduped + sorted, so the choice is deterministic across runs.
    Fallback is CPT 72148 (MRI lumbar) if the case has no radiology.
    """
    _warn_retrospective_once()

    if not procedure_code:
        cpts = case.radiology_cpt_codes  # already sorted + deduped
        procedure_code = cpts[0] if cpts else "72148"

    return {
        "clinical_text": case.primary_discharge_text,
        "payer_id": case.payer_proxy,
        "procedure_code": procedure_code,
        "patient_id": case.subject_id,
    }


def case_to_ground_truth(case: MIMICCase) -> dict:
    return {
        "diagnosis_codes": case.icd10_codes or case.icd9_codes,
        "procedure_codes": [p.icd_code for p in case.procedures],
        "drugs": case.drug_list,
        "radiology_cpt_codes": case.radiology_cpt_codes,
        "patient_age": case.patient.anchor_age if case.patient else None,
        "patient_gender": case.patient.gender if case.patient else None,
        "insurance": case.admission.insurance if case.admission else None,
        "n_diagnoses": len(case.diagnoses),
        "n_prescriptions": len(case.prescriptions),
        "n_radiology_reports": len(case.radiology_reports),
    }


# ── Utility ────────────────────────────────────────────────────────────────

def _find_file(directory: Path, table_name: str) -> Path:
    for ext in [".csv.gz", ".csv"]:
        path = directory / f"{table_name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Table '{table_name}' not found in {directory}. "
        f"Expected: {table_name}.csv.gz or {table_name}.csv"
    )


def validate_mimic_directory(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict:
    required_hosp = ["patients", "admissions", "diagnoses_icd",
                     "procedures_icd", "prescriptions"]
    required_note = ["discharge"]
    optional_note = ["radiology", "radiology_detail", "discharge_detail"]
    optional_hosp = ["labevents"]

    report = {"found": [], "missing": [], "optional_missing": [], "ready": False}

    for table in required_hosp:
        try:
            _find_file(mimic_dir / "hosp", table)
            report["found"].append(f"hosp/{table}")
        except FileNotFoundError:
            report["missing"].append(f"hosp/{table}")

    for table in required_note:
        try:
            _find_file(mimic_dir / "note", table)
            report["found"].append(f"note/{table}")
        except FileNotFoundError:
            report["missing"].append(f"note/{table}")

    for table in optional_note:
        try:
            _find_file(mimic_dir / "note", table)
            report["found"].append(f"note/{table}")
        except FileNotFoundError:
            report["optional_missing"].append(f"note/{table}")

    for table in optional_hosp:
        try:
            _find_file(mimic_dir / "hosp", table)
            report["found"].append(f"hosp/{table}")
        except FileNotFoundError:
            report["optional_missing"].append(f"hosp/{table}")

    report["ready"] = len(report["missing"]) == 0
    return report