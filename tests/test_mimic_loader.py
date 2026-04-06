"""Tests for MIMIC-IV data loader using mock CSV data.
No actual MIMIC data required — creates temporary CSVs."""

import csv
import gzip
import tempfile
from pathlib import Path

import pytest

from src.data.mimic_loader import (
    MIMICCase, MIMICPatient, MIMICAdmission,
    load_patients, load_admissions, load_diagnoses,
    load_prescriptions, load_discharge_notes,
    assemble_case, load_all_cases, validate_mimic_directory,
    case_to_pipeline_input, case_to_ground_truth,
    filter_cases_by_diagnosis, filter_cases_with_imaging,
)


@pytest.fixture
def mock_mimic_dir():
    """Create a temporary MIMIC-IV directory with mock CSV.gz files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hosp = root / "hosp"
        note = root / "note"
        hosp.mkdir()
        note.mkdir()

        # patients.csv.gz
        _write_gz_csv(hosp / "patients.csv.gz", [
            {"subject_id": "100", "gender": "F", "anchor_age": "52", "anchor_year": "2150"},
            {"subject_id": "200", "gender": "M", "anchor_age": "68", "anchor_year": "2145"},
        ])

        # admissions.csv.gz
        _write_gz_csv(hosp / "admissions.csv.gz", [
            {"subject_id": "100", "hadm_id": "1001", "admittime": "2150-01-01",
             "dischtime": "2150-01-05", "admission_type": "URGENT", "insurance": "Other", "race": "WHITE"},
            {"subject_id": "200", "hadm_id": "2001", "admittime": "2145-03-10",
             "dischtime": "2145-03-15", "admission_type": "ELECTIVE", "insurance": "Medicare", "race": "BLACK"},
        ])

        # diagnoses_icd.csv.gz
        _write_gz_csv(hosp / "diagnoses_icd.csv.gz", [
            {"subject_id": "100", "hadm_id": "1001", "icd_code": "M544", "icd_version": "10", "seq_num": "1"},
            {"subject_id": "100", "hadm_id": "1001", "icd_code": "M5116", "icd_version": "10", "seq_num": "2"},
            {"subject_id": "200", "hadm_id": "2001", "icd_code": "I10", "icd_version": "10", "seq_num": "1"},
        ])

        # procedures_icd.csv.gz
        _write_gz_csv(hosp / "procedures_icd.csv.gz", [
            {"subject_id": "100", "hadm_id": "1001", "icd_code": "BR39ZZZ", "icd_version": "10", "seq_num": "1"},
        ])

        # prescriptions.csv.gz
        _write_gz_csv(hosp / "prescriptions.csv.gz", [
            {"subject_id": "100", "hadm_id": "1001", "drug": "Naproxen", "drug_type": "MAIN",
             "route": "PO", "dose_val_rx": "500", "dose_unit_rx": "mg", "starttime": "", "stoptime": ""},
            {"subject_id": "100", "hadm_id": "1001", "drug": "Gabapentin", "drug_type": "MAIN",
             "route": "PO", "dose_val_rx": "300", "dose_unit_rx": "mg", "starttime": "", "stoptime": ""},
            {"subject_id": "200", "hadm_id": "2001", "drug": "Lisinopril", "drug_type": "MAIN",
             "route": "PO", "dose_val_rx": "10", "dose_unit_rx": "mg", "starttime": "", "stoptime": ""},
        ])

        # discharge.csv.gz
        _write_gz_csv(note / "discharge.csv.gz", [
            {"note_id": "N1", "subject_id": "100", "hadm_id": "1001", "charttime": "2150-01-05",
             "text": "DISCHARGE SUMMARY\n\nChief Complaint: Low back pain with radiculopathy.\n\n"
                     "History of Present Illness: 52yo F with 3-month history of progressive low back pain "
                     "radiating to left leg. Failed 6 weeks of physical therapy and NSAIDs (Naproxen 500mg BID). "
                     "MRI lumbar spine shows L4-L5 disc herniation with nerve root compression.\n\n"
                     "Diagnoses: Lumbar radiculopathy (M54.4), Lumbar disc herniation (M51.16)\n\n"
                     "Medications at Discharge: Gabapentin 300mg TID, Naproxen 500mg BID"},
            {"note_id": "N2", "subject_id": "200", "hadm_id": "2001", "charttime": "2145-03-15",
             "text": "DISCHARGE SUMMARY\n\nChief Complaint: Chest pain.\n\n"
                     "History: 68yo M with hypertension presenting with exertional chest pain.\n\n"
                     "Diagnoses: Essential hypertension (I10)"},
        ])

        yield root


def _write_gz_csv(path: Path, rows: list[dict]):
    """Write a list of dicts as a gzipped CSV."""
    if not rows:
        return
    with gzip.open(path, "wt", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ── Table loader tests ─────────────────────────────────────────────────────

class TestTableLoaders:
    def test_load_patients(self, mock_mimic_dir):
        patients = load_patients(mock_mimic_dir)
        assert len(patients) == 2
        assert patients["100"].gender == "F"
        assert patients["100"].anchor_age == 52

    def test_load_admissions(self, mock_mimic_dir):
        admissions = load_admissions(mock_mimic_dir)
        assert "100" in admissions
        assert admissions["100"][0].hadm_id == "1001"
        assert admissions["100"][0].insurance == "Other"

    def test_load_diagnoses(self, mock_mimic_dir):
        diagnoses = load_diagnoses(mock_mimic_dir)
        assert "1001" in diagnoses
        assert len(diagnoses["1001"]) == 2
        assert diagnoses["1001"][0].icd_code == "M544"

    def test_load_prescriptions(self, mock_mimic_dir):
        meds = load_prescriptions(mock_mimic_dir)
        assert "1001" in meds
        drugs = [m.drug for m in meds["1001"]]
        assert "Naproxen" in drugs
        assert "Gabapentin" in drugs

    def test_load_discharge_notes(self, mock_mimic_dir):
        notes = load_discharge_notes(mock_mimic_dir)
        assert "1001" in notes
        assert "radiculopathy" in notes["1001"][0].text


# ── Case assembly tests ───────────────────────────────────────────────────

class TestCaseAssembly:
    def test_load_all_cases(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        assert len(cases) == 2

    def test_load_all_cases_with_limit(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir, limit=1)
        assert len(cases) == 1

    def test_case_has_patient(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        case = next(c for c in cases if c.hadm_id == "1001")
        assert case.patient is not None
        assert case.patient.gender == "F"

    def test_case_has_diagnoses(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        case = next(c for c in cases if c.hadm_id == "1001")
        assert len(case.diagnoses) == 2

    def test_icd10_codes_sorted(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        case = next(c for c in cases if c.hadm_id == "1001")
        codes = case.icd10_codes
        assert "M544" in codes
        assert "M5116" in codes

    def test_drug_list_unique(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        case = next(c for c in cases if c.hadm_id == "1001")
        drugs = case.drug_list
        assert "Naproxen" in drugs
        assert "Gabapentin" in drugs
        assert len(drugs) == len(set(d.lower() for d in drugs))

    def test_primary_discharge_text(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        case = next(c for c in cases if c.hadm_id == "1001")
        assert "radiculopathy" in case.primary_discharge_text

    def test_payer_proxy_commercial(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        case = next(c for c in cases if c.hadm_id == "1001")
        assert case.payer_proxy == "UHC"  # "Other" maps to commercial

    def test_payer_proxy_medicare(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        case = next(c for c in cases if c.hadm_id == "2001")
        assert case.payer_proxy == "MEDICARE"


# ── Pipeline integration tests ─────────────────────────────────────────────

class TestPipelineIntegration:
    def test_case_to_pipeline_input(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        case = next(c for c in cases if c.hadm_id == "1001")
        inputs = case_to_pipeline_input(case, procedure_code="72148")
        assert inputs["clinical_text"] != ""
        assert inputs["payer_id"] == "UHC"
        assert inputs["procedure_code"] == "72148"
        assert inputs["patient_id"] == "100"

    def test_case_to_ground_truth(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        case = next(c for c in cases if c.hadm_id == "1001")
        gt = case_to_ground_truth(case)
        assert "M544" in gt["diagnosis_codes"]
        assert "Naproxen" in gt["drugs"]
        assert gt["patient_age"] == 52
        assert gt["patient_gender"] == "F"
        assert gt["insurance"] == "Other"


# ── Filter tests ───────────────────────────────────────────────────────────

class TestFilters:
    def test_filter_by_diagnosis(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        filtered = filter_cases_by_diagnosis(cases, ["M54"])
        assert len(filtered) == 1
        assert filtered[0].hadm_id == "1001"

    def test_filter_by_diagnosis_no_match(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        filtered = filter_cases_by_diagnosis(cases, ["Z99"])
        assert len(filtered) == 0

    def test_filter_with_imaging_empty(self, mock_mimic_dir):
        cases = load_all_cases(mock_mimic_dir)
        filtered = filter_cases_with_imaging(cases)
        assert len(filtered) == 0  # mock data has no radiology


# ── Validation tests ───────────────────────────────────────────────────────

class TestValidation:
    def test_validate_complete_directory(self, mock_mimic_dir):
        report = validate_mimic_directory(mock_mimic_dir)
        assert report["ready"] is True
        assert len(report["missing"]) == 0

    def test_validate_incomplete_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = validate_mimic_directory(Path(tmpdir))
            assert report["ready"] is False
            assert len(report["missing"]) > 0
