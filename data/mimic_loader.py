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

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ── Default paths ──────────────────────────────────────────────────────────

DEFAULT_MIMIC_DIR = Path(__file__).resolve().parents[2] / "data" / "mimic-iv"


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
    insurance: str = ""       # payer proxy: "Medicare", "Medicaid", "Other"
    race: str = ""


@dataclass
class MIMICDiagnosis:
    hadm_id: str
    subject_id: str
    icd_code: str = ""
    icd_version: int = 10     # 9 or 10
    seq_num: int = 0          # priority rank


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
    drug_type: str = ""       # "MAIN", "BASE", "ADDITIVE"
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
    # field_name includes: "exam_name", "cpt_code", "parent_note_id"


@dataclass
class MIMICCase:
    """
    A fully assembled clinical case combining structured data with
    free-text notes for a single hospitalization. This is what gets
    fed into the AuthorizeAI pipeline.
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
        """Return the longest discharge note (typically the main summary)."""
        if not self.discharge_notes:
            return ""
        return max(self.discharge_notes, key=lambda n: len(n.text)).text

    @property
    def icd10_codes(self) -> list[str]:
        """Return ICD-10 diagnosis codes sorted by priority."""
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
        """Unique drug names prescribed during this admission."""
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
        """CPT codes from radiology studies during this admission."""
        return [
            d.field_value for d in self.radiology_details
            if d.field_name == "cpt_code" and d.field_value
        ]

    @property
    def payer_proxy(self) -> str:
        """
        Map MIMIC insurance field to a payer ID usable by the pipeline.
        MIMIC uses: 'Medicare', 'Medicaid', 'Other' (commercial).
        """
        if not self.admission:
            return "UHC"
        ins = self.admission.insurance.lower()
        if "medicare" in ins:
            return "MEDICARE"
        elif "medicaid" in ins:
            return "MEDICAID"
        else:
            return "UHC"  # default commercial payer for prototype


# ── CSV readers ────────────────────────────────────────────────────────────

def _read_gz_csv(filepath: Path) -> Iterator[dict]:
    """Stream rows from a gzipped CSV file."""
    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        yield from reader


def _read_csv(filepath: Path) -> Iterator[dict]:
    """Read from either .csv.gz or .csv."""
    if filepath.suffix == ".gz":
        return _read_gz_csv(filepath)
    else:
        f = open(filepath, "r", encoding="utf-8", errors="replace")
        reader = csv.DictReader(f)
        return reader


# ── Loaders for individual tables ──────────────────────────────────────────

def load_patients(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict[str, MIMICPatient]:
    """Load patients table → dict keyed by subject_id."""
    path = _find_file(mimic_dir / "hosp", "patients")
    patients = {}
    for row in _read_csv(path):
        sid = row["subject_id"]
        patients[sid] = MIMICPatient(
            subject_id=sid,
            gender=row.get("gender", ""),
            anchor_age=int(row.get("anchor_age", 0)),
            anchor_year=int(row.get("anchor_year", 0)),
        )
    return patients


def load_admissions(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict[str, list[MIMICAdmission]]:
    """Load admissions → dict keyed by subject_id (one patient can have many admissions)."""
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


def load_diagnoses(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict[str, list[MIMICDiagnosis]]:
    """Load diagnoses_icd → dict keyed by hadm_id."""
    path = _find_file(mimic_dir / "hosp", "diagnoses_icd")
    diag: dict[str, list[MIMICDiagnosis]] = defaultdict(list)
    for row in _read_csv(path):
        hid = row["hadm_id"]
        diag[hid].append(MIMICDiagnosis(
            hadm_id=hid,
            subject_id=row["subject_id"],
            icd_code=row.get("icd_code", ""),
            icd_version=int(row.get("icd_version", 10)),
            seq_num=int(row.get("seq_num", 0)),
        ))
    return dict(diag)


def load_procedures(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict[str, list[MIMICProcedure]]:
    """Load procedures_icd → dict keyed by hadm_id."""
    path = _find_file(mimic_dir / "hosp", "procedures_icd")
    procs: dict[str, list[MIMICProcedure]] = defaultdict(list)
    for row in _read_csv(path):
        hid = row["hadm_id"]
        procs[hid].append(MIMICProcedure(
            hadm_id=hid,
            subject_id=row["subject_id"],
            icd_code=row.get("icd_code", ""),
            icd_version=int(row.get("icd_version", 10)),
            seq_num=int(row.get("seq_num", 0)),
        ))
    return dict(procs)


def load_prescriptions(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict[str, list[MIMICPrescription]]:
    """Load prescriptions → dict keyed by hadm_id."""
    path = _find_file(mimic_dir / "hosp", "prescriptions")
    meds: dict[str, list[MIMICPrescription]] = defaultdict(list)
    for row in _read_csv(path):
        hid = row.get("hadm_id", "")
        if not hid:
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


def load_discharge_notes(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict[str, list[MIMICDischargeNote]]:
    """Load discharge summaries → dict keyed by hadm_id."""
    path = _find_file(mimic_dir / "note", "discharge")
    notes: dict[str, list[MIMICDischargeNote]] = defaultdict(list)
    for row in _read_csv(path):
        hid = row.get("hadm_id", "")
        if not hid:
            continue
        notes[hid].append(MIMICDischargeNote(
            note_id=row.get("note_id", ""),
            subject_id=row["subject_id"],
            hadm_id=hid,
            charttime=row.get("charttime", ""),
            text=row.get("text", ""),
        ))
    return dict(notes)


def load_radiology_reports(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict[str, list[MIMICRadiologyReport]]:
    """Load radiology reports → dict keyed by hadm_id."""
    path = _find_file(mimic_dir / "note", "radiology")
    reports: dict[str, list[MIMICRadiologyReport]] = defaultdict(list)
    for row in _read_csv(path):
        hid = row.get("hadm_id", "")
        if not hid:
            continue
        reports[hid].append(MIMICRadiologyReport(
            note_id=row.get("note_id", ""),
            subject_id=row["subject_id"],
            hadm_id=hid,
            charttime=row.get("charttime", ""),
            text=row.get("text", ""),
        ))
    return dict(reports)


def load_radiology_details(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict[str, list[MIMICRadiologyDetail]]:
    """Load radiology_detail → dict keyed by note_id."""
    path = _find_file(mimic_dir / "note", "radiology_detail")
    details: dict[str, list[MIMICRadiologyDetail]] = defaultdict(list)
    for row in _read_csv(path):
        nid = row.get("note_id", "")
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
    """
    Assemble a complete MIMICCase for a single hospitalization.
    Returns None if the hadm_id cannot be found in discharge notes.
    """
    notes = discharge_notes.get(hadm_id, [])
    if not notes:
        return None

    subject_id = notes[0].subject_id
    patient = patients.get(subject_id)

    # Find the matching admission
    admission = None
    for adm in admissions_by_subject.get(subject_id, []):
        if adm.hadm_id == hadm_id:
            admission = adm
            break

    # Collect radiology details by note_id
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
    Load and assemble all clinical cases from MIMIC-IV.

    Args:
        mimic_dir: Root directory containing hosp/ and note/ subdirs
        limit: Max number of cases to load (None = all)
        require_discharge_note: Only include cases with discharge notes
        require_radiology: Only include cases with radiology reports

    Returns:
        List of assembled MIMICCase objects
    """
    print("Loading MIMIC-IV tables...")
    patients = load_patients(mimic_dir)
    print(f"  patients: {len(patients)}")

    admissions = load_admissions(mimic_dir)
    print(f"  admissions: {sum(len(v) for v in admissions.values())}")

    diagnoses = load_diagnoses(mimic_dir)
    print(f"  diagnoses: {sum(len(v) for v in diagnoses.values())}")

    procedures = load_procedures(mimic_dir)
    print(f"  procedures: {sum(len(v) for v in procedures.values())}")

    prescriptions = load_prescriptions(mimic_dir)
    print(f"  prescriptions: {sum(len(v) for v in prescriptions.values())}")

    discharge_notes = load_discharge_notes(mimic_dir)
    print(f"  discharge notes: {sum(len(v) for v in discharge_notes.values())}")

    radiology_reports = None
    radiology_details = None
    rad_path = mimic_dir / "note"
    if (rad_path / "radiology.csv.gz").exists() or (rad_path / "radiology.csv").exists():
        radiology_reports = load_radiology_reports(mimic_dir)
        print(f"  radiology reports: {sum(len(v) for v in radiology_reports.values())}")
        try:
            radiology_details = load_radiology_details(mimic_dir)
            print(f"  radiology details: {sum(len(v) for v in radiology_details.values())}")
        except FileNotFoundError:
            pass

    # Assemble cases
    print("\nAssembling cases...")
    cases = []
    hadm_ids = list(discharge_notes.keys())

    for hadm_id in hadm_ids:
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
    """
    Filter cases that involve specific procedures.
    Useful for finding cases relevant to AuthorizeAI's 10 target procedures.
    """
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
    """
    Filter cases where any diagnosis code starts with the given prefixes.
    Example: icd_prefixes=["M54", "M51"] for low back pain / disc disorders.
    """
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
    """Return only cases that have radiology reports — high PA relevance."""
    return [c for c in cases if c.radiology_reports]


# ── Pipeline integration ──────────────────────────────────────────────────

def case_to_pipeline_input(
    case: MIMICCase,
    procedure_code: str = "",
) -> dict:
    """
    Convert a MIMICCase into kwargs for run_pipeline().

    If no procedure_code is provided, attempts to infer one from
    radiology CPT codes or defaults to a generic imaging code.
    """
    # Determine procedure code
    if not procedure_code:
        cpts = case.radiology_cpt_codes
        if cpts:
            procedure_code = cpts[0]
        else:
            procedure_code = "72148"  # default: MRI lumbar

    return {
        "clinical_text": case.primary_discharge_text,
        "payer_id": case.payer_proxy,
        "procedure_code": procedure_code,
        "patient_id": case.subject_id,
    }


def case_to_ground_truth(case: MIMICCase) -> dict:
    """
    Extract structured ground truth from MIMIC structured data.
    Used to evaluate Agent 1's extraction accuracy.
    """
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
    """Find a CSV file by table name, checking .csv.gz and .csv variants."""
    for ext in [".csv.gz", ".csv"]:
        path = directory / f"{table_name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Table '{table_name}' not found in {directory}. "
        f"Expected: {table_name}.csv.gz or {table_name}.csv"
    )


def validate_mimic_directory(mimic_dir: Path = DEFAULT_MIMIC_DIR) -> dict:
    """
    Check which MIMIC-IV tables are available and return a status report.
    """
    required_hosp = ["patients", "admissions", "diagnoses_icd",
                     "procedures_icd", "prescriptions"]
    required_note = ["discharge"]
    optional_note = ["radiology", "radiology_detail", "discharge_detail"]
    optional_hosp = ["labevents"]

    report = {"found": [], "missing": [], "optional_missing": []}

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
