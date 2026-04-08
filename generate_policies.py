"""
AuthorizeAI — Synthetic Policy Document Generator
====================================================
Generates realistic coverage policy documents for all supported
payer/CPT combinations based on publicly known medical necessity
criteria from clinical guidelines (ACR, AAN, AMA, CMS NCDs).

DISCLAIMER: These are synthetic policy documents for development
and testing purposes. They do not represent actual payer coverage
determinations or official policy positions.

Usage: python generate_policies.py
Output: data/policies/ (one .json file per payer-procedure pair)
"""

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "policies"

PAYERS = ["UHC", "AETNA", "CIGNA", "HUMANA", "ELEVANCE", "CENTENE", "KAISER"]

# ── Policy templates by CPT code ──────────────────────────────────────────
# Based on widely published clinical guidelines and common PA criteria.

POLICY_TEMPLATES = {
    "72148": {
        "procedure_name": "MRI Lumbar Spine without Contrast",
        "category": "MRI",
        "indications": [
            {
                "id": "C1",
                "text": "Patient presents with persistent low back pain lasting at least 6 weeks that has not responded to conservative treatment including physical therapy, chiropractic care, or pharmacotherapy.",
                "section": "indications",
            },
            {
                "id": "C2",
                "text": "Patient exhibits neurological deficits such as radiculopathy, progressive motor weakness, sensory loss, bowel or bladder dysfunction, or cauda equina syndrome symptoms.",
                "section": "indications",
            },
            {
                "id": "C3",
                "text": "Clinical suspicion of serious underlying pathology including but not limited to spinal infection, malignancy, fracture, or inflammatory spondyloarthropathy based on red flag findings.",
                "section": "indications",
            },
            {
                "id": "C4",
                "text": "Pre-surgical planning when surgical intervention for a documented spinal condition has been recommended by the treating or consulting surgeon.",
                "section": "indications",
            },
        ],
        "contraindications": [
            {
                "id": "C5",
                "text": "MRI is not covered for routine screening of low back pain without clinical indication or red flag symptoms.",
                "section": "contraindications",
            },
            {
                "id": "C6",
                "text": "Repeat MRI within 12 months is not covered unless there is a documented significant change in clinical status, new neurological findings, or post-surgical evaluation need.",
                "section": "contraindications",
            },
        ],
        "documentation": [
            {
                "id": "C7",
                "text": "Clinical notes must document the duration, severity, and character of symptoms including onset date and functional impact.",
                "section": "documentation",
            },
            {
                "id": "C8",
                "text": "Records of prior conservative treatments attempted, including specific therapies, duration of each treatment, and documented outcomes or reasons for failure.",
                "section": "documentation",
            },
            {
                "id": "C9",
                "text": "Physical examination findings including neurological assessment, straight leg raise test results, reflexes, motor strength testing, and sensory examination.",
                "section": "documentation",
            },
        ],
        "logic_tree": {
            "type": "AND",
            "children": [
                {
                    "type": "OR",
                    "label": "Clinical Indication",
                    "children": ["C1", "C2", "C3", "C4"],
                },
                {
                    "type": "AND",
                    "label": "No Exclusions",
                    "children": [
                        {"type": "NOT", "child": "C5"},
                        {"type": "NOT", "child": "C6"},
                    ],
                },
                {
                    "type": "AND",
                    "label": "Documentation Complete",
                    "children": ["C7", "C8", "C9"],
                },
            ],
        },
    },
    "70553": {
        "procedure_name": "MRI Brain with and without Contrast",
        "category": "MRI",
        "indications": [
            {
                "id": "C1",
                "text": "Patient presents with new onset seizures, unexplained headaches persisting more than 4 weeks with neurological signs, or focal neurological deficits suggesting intracranial pathology.",
                "section": "indications",
            },
            {
                "id": "C2",
                "text": "Evaluation of suspected or known intracranial neoplasm, including initial diagnosis, treatment planning, or surveillance for recurrence per NCCN guidelines.",
                "section": "indications",
            },
            {
                "id": "C3",
                "text": "Clinical presentation consistent with demyelinating disease such as multiple sclerosis, including optic neuritis, transverse myelitis, or clinically isolated syndrome meeting McDonald criteria.",
                "section": "indications",
            },
            {
                "id": "C4",
                "text": "Evaluation of suspected cerebrovascular disease including stroke workup, vascular malformation, or aneurysm when CT angiography is insufficient or contraindicated.",
                "section": "indications",
            },
            {
                "id": "C5",
                "text": "Pre-operative planning for intracranial surgery or stereotactic procedures requiring detailed anatomical mapping.",
                "section": "indications",
            },
        ],
        "contraindications": [
            {
                "id": "C6",
                "text": "Not covered for routine headache evaluation without red flag symptoms, neurological deficits, or failure of appropriate first-line treatment.",
                "section": "contraindications",
            },
            {
                "id": "C7",
                "text": "Not covered for screening of asymptomatic patients or for cognitive complaints without focal neurological findings in patients under age 65.",
                "section": "contraindications",
            },
        ],
        "documentation": [
            {
                "id": "C8",
                "text": "Detailed neurological examination findings including mental status, cranial nerve assessment, motor and sensory examination, and cerebellar function testing.",
                "section": "documentation",
            },
            {
                "id": "C9",
                "text": "Documentation of symptom onset, progression, duration, and any prior imaging results with dates.",
                "section": "documentation",
            },
            {
                "id": "C10",
                "text": "For headache indications: documentation of headache characteristics, failed treatments, and presence or absence of red flag features per AAN guidelines.",
                "section": "documentation",
            },
        ],
        "logic_tree": {
            "type": "AND",
            "children": [
                {
                    "type": "OR",
                    "label": "Clinical Indication",
                    "children": ["C1", "C2", "C3", "C4", "C5"],
                },
                {
                    "type": "AND",
                    "label": "No Exclusions",
                    "children": [
                        {"type": "NOT", "child": "C6"},
                        {"type": "NOT", "child": "C7"},
                    ],
                },
                {
                    "type": "AND",
                    "label": "Documentation Complete",
                    "children": ["C8", "C9"],
                },
            ],
        },
    },
    "75557": {
        "procedure_name": "Cardiac MRI for Morphology and Function",
        "category": "MRI",
        "indications": [
            {
                "id": "C1",
                "text": "Evaluation of known or suspected cardiomyopathy including hypertrophic, dilated, restrictive, or arrhythmogenic right ventricular cardiomyopathy when echocardiography is inconclusive.",
                "section": "indications",
            },
            {
                "id": "C2",
                "text": "Assessment of myocardial viability in patients with coronary artery disease being considered for revascularization when other non-invasive testing is inconclusive.",
                "section": "indications",
            },
            {
                "id": "C3",
                "text": "Evaluation of cardiac masses, tumors, or thrombus when echocardiography provides inadequate characterization.",
                "section": "indications",
            },
            {
                "id": "C4",
                "text": "Assessment of congenital heart disease in adults when anatomical detail beyond echocardiography is required for surgical or interventional planning.",
                "section": "indications",
            },
            {
                "id": "C5",
                "text": "Evaluation of pericardial disease including constrictive pericarditis when clinical and echocardiographic findings are equivocal.",
                "section": "indications",
            },
        ],
        "contraindications": [
            {
                "id": "C6",
                "text": "Not covered as initial evaluation when echocardiography has not yet been performed and is not contraindicated.",
                "section": "contraindications",
            },
            {
                "id": "C7",
                "text": "Not covered for routine follow-up of stable, well-characterized cardiac conditions without change in clinical status.",
                "section": "contraindications",
            },
        ],
        "documentation": [
            {
                "id": "C8",
                "text": "Prior echocardiography report with date and findings, including documentation of why echocardiography was insufficient for clinical decision-making.",
                "section": "documentation",
            },
            {
                "id": "C9",
                "text": "Relevant cardiac history including prior interventions, current medications, and functional status (NYHA class or equivalent).",
                "section": "documentation",
            },
            {
                "id": "C10",
                "text": "Specific clinical question the cardiac MRI is expected to answer that cannot be addressed by other available imaging modalities.",
                "section": "documentation",
            },
        ],
        "logic_tree": {
            "type": "AND",
            "children": [
                {
                    "type": "OR",
                    "label": "Clinical Indication",
                    "children": ["C1", "C2", "C3", "C4", "C5"],
                },
                {
                    "type": "AND",
                    "label": "No Exclusions",
                    "children": [
                        {"type": "NOT", "child": "C6"},
                        {"type": "NOT", "child": "C7"},
                    ],
                },
                {
                    "type": "AND",
                    "label": "Documentation Complete",
                    "children": ["C8", "C9", "C10"],
                },
            ],
        },
    },
    "74177": {
        "procedure_name": "CT Abdomen and Pelvis with Contrast",
        "category": "CT",
        "indications": [
            {
                "id": "C1",
                "text": "Acute abdominal pain with clinical signs suggesting surgical emergency including appendicitis, diverticulitis, bowel obstruction, or visceral perforation.",
                "section": "indications",
            },
            {
                "id": "C2",
                "text": "Evaluation or staging of known or suspected abdominal or pelvic malignancy per NCCN staging guidelines.",
                "section": "indications",
            },
            {
                "id": "C3",
                "text": "Assessment of suspected abdominal abscess, fluid collection, or unexplained fever with localizing abdominal signs when ultrasound is nondiagnostic.",
                "section": "indications",
            },
            {
                "id": "C4",
                "text": "Evaluation of abdominal trauma with hemodynamic instability or high-energy mechanism in accordance with ATLS protocols.",
                "section": "indications",
            },
            {
                "id": "C5",
                "text": "Surveillance imaging for previously treated abdominal malignancy at intervals consistent with NCCN surveillance guidelines.",
                "section": "indications",
            },
        ],
        "contraindications": [
            {
                "id": "C6",
                "text": "Not covered for evaluation of chronic, stable, non-specific abdominal pain without red flag symptoms or abnormal laboratory findings.",
                "section": "contraindications",
            },
            {
                "id": "C7",
                "text": "Not covered when ultrasound is the appropriate first-line imaging modality and has not yet been attempted (e.g., right upper quadrant pain, pelvic pathology in reproductive-age females).",
                "section": "contraindications",
            },
        ],
        "documentation": [
            {
                "id": "C8",
                "text": "Clinical history including symptom onset, duration, associated symptoms, and relevant physical examination findings.",
                "section": "documentation",
            },
            {
                "id": "C9",
                "text": "Relevant laboratory results including CBC, metabolic panel, and tumor markers as applicable.",
                "section": "documentation",
            },
            {
                "id": "C10",
                "text": "Results of prior imaging studies with dates, including explanation of why further imaging with CT is clinically necessary.",
                "section": "documentation",
            },
        ],
        "logic_tree": {
            "type": "AND",
            "children": [
                {
                    "type": "OR",
                    "label": "Clinical Indication",
                    "children": ["C1", "C2", "C3", "C4", "C5"],
                },
                {
                    "type": "AND",
                    "label": "No Exclusions",
                    "children": [
                        {"type": "NOT", "child": "C6"},
                        {"type": "NOT", "child": "C7"},
                    ],
                },
                {
                    "type": "AND",
                    "label": "Documentation Complete",
                    "children": ["C8", "C9", "C10"],
                },
            ],
        },
    },
    "71260": {
        "procedure_name": "CT Chest with Contrast",
        "category": "CT",
        "indications": [
            {
                "id": "C1",
                "text": "Evaluation of suspected pulmonary embolism in patients with moderate to high pre-test probability per Wells criteria or positive D-dimer.",
                "section": "indications",
            },
            {
                "id": "C2",
                "text": "Staging or surveillance of known thoracic malignancy including lung cancer, lymphoma, or mediastinal tumors per NCCN guidelines.",
                "section": "indications",
            },
            {
                "id": "C3",
                "text": "Evaluation of persistent pulmonary nodule detected on prior imaging requiring follow-up per Fleischner Society guidelines.",
                "section": "indications",
            },
            {
                "id": "C4",
                "text": "Assessment of thoracic aortic pathology including suspected dissection, aneurysm, or post-surgical follow-up.",
                "section": "indications",
            },
            {
                "id": "C5",
                "text": "Evaluation of complicated pneumonia, empyema, or mediastinitis not adequately assessed by chest radiograph.",
                "section": "indications",
            },
        ],
        "contraindications": [
            {
                "id": "C6",
                "text": "Not covered for routine evaluation of uncomplicated community-acquired pneumonia responding to treatment.",
                "section": "contraindications",
            },
            {
                "id": "C7",
                "text": "Not covered for low pre-test probability PE when D-dimer is negative per PERC rule.",
                "section": "contraindications",
            },
        ],
        "documentation": [
            {
                "id": "C8",
                "text": "Relevant clinical symptoms and physical examination findings including vital signs, oxygen saturation, and respiratory assessment.",
                "section": "documentation",
            },
            {
                "id": "C9",
                "text": "Prior chest imaging results with dates and findings. For PE evaluation, Wells score calculation or D-dimer result.",
                "section": "documentation",
            },
            {
                "id": "C10",
                "text": "For nodule follow-up: prior imaging comparison with nodule measurements and Fleischner Society risk category.",
                "section": "documentation",
            },
        ],
        "logic_tree": {
            "type": "AND",
            "children": [
                {
                    "type": "OR",
                    "label": "Clinical Indication",
                    "children": ["C1", "C2", "C3", "C4", "C5"],
                },
                {
                    "type": "AND",
                    "label": "No Exclusions",
                    "children": [
                        {"type": "NOT", "child": "C6"},
                        {"type": "NOT", "child": "C7"},
                    ],
                },
                {
                    "type": "AND",
                    "label": "Documentation Complete",
                    "children": ["C8", "C9"],
                },
            ],
        },
    },
    "99242": {
        "procedure_name": "Office Consultation — Cardiology",
        "category": "SPECIALTY_REFERRAL",
        "indications": [
            {
                "id": "C1",
                "text": "Patient presents with cardiovascular symptoms such as chest pain, palpitations, syncope, or dyspnea on exertion that require specialist evaluation beyond primary care scope.",
                "section": "indications",
            },
            {
                "id": "C2",
                "text": "Abnormal cardiac diagnostic findings including ECG abnormalities, elevated troponin, abnormal echocardiogram, or positive stress test requiring specialist interpretation and management.",
                "section": "indications",
            },
            {
                "id": "C3",
                "text": "Management of established cardiovascular disease including heart failure, valvular disease, or arrhythmia requiring medication optimization or procedural evaluation.",
                "section": "indications",
            },
            {
                "id": "C4",
                "text": "Pre-operative cardiac risk assessment for patients with known cardiac disease or multiple cardiac risk factors undergoing intermediate or high-risk non-cardiac surgery.",
                "section": "indications",
            },
        ],
        "contraindications": [
            {
                "id": "C5",
                "text": "Not covered for routine cardiovascular screening in asymptomatic low-risk patients without abnormal findings.",
                "section": "contraindications",
            },
        ],
        "documentation": [
            {
                "id": "C6",
                "text": "Referring physician clinical notes documenting the reason for referral, relevant symptoms, and clinical findings.",
                "section": "documentation",
            },
            {
                "id": "C7",
                "text": "Relevant diagnostic test results including ECG, echocardiogram, stress test, or laboratory values as applicable.",
                "section": "documentation",
            },
            {
                "id": "C8",
                "text": "Current medication list and any cardiac medications previously tried with their outcomes.",
                "section": "documentation",
            },
        ],
        "logic_tree": {
            "type": "AND",
            "children": [
                {
                    "type": "OR",
                    "label": "Clinical Indication",
                    "children": ["C1", "C2", "C3", "C4"],
                },
                {"type": "NOT", "child": "C5"},
                {
                    "type": "AND",
                    "label": "Documentation Complete",
                    "children": ["C6", "C7"],
                },
            ],
        },
    },
    "99243": {
        "procedure_name": "Office Consultation — Neurology",
        "category": "SPECIALTY_REFERRAL",
        "indications": [
            {
                "id": "C1",
                "text": "Patient presents with neurological symptoms such as recurrent seizures, progressive headaches with red flags, movement disorders, or unexplained cognitive decline requiring specialist evaluation.",
                "section": "indications",
            },
            {
                "id": "C2",
                "text": "Abnormal neurological findings on examination including focal deficits, papilledema, asymmetric reflexes, or abnormal gait requiring specialist assessment.",
                "section": "indications",
            },
            {
                "id": "C3",
                "text": "Evaluation and management of established neurological conditions including epilepsy, multiple sclerosis, Parkinson disease, or peripheral neuropathy requiring treatment adjustment.",
                "section": "indications",
            },
            {
                "id": "C4",
                "text": "Interpretation of specialized neurological testing including EEG, EMG/NCS, or advanced neuroimaging findings that require neurologist expertise.",
                "section": "indications",
            },
        ],
        "contraindications": [
            {
                "id": "C5",
                "text": "Not covered for isolated tension-type headaches without red flag features that are manageable in primary care.",
                "section": "contraindications",
            },
            {
                "id": "C6",
                "text": "Not covered for routine cognitive screening in patients without documented cognitive complaints or functional decline.",
                "section": "contraindications",
            },
        ],
        "documentation": [
            {
                "id": "C7",
                "text": "Referring physician clinical notes with neurological review of systems and focused neurological examination findings.",
                "section": "documentation",
            },
            {
                "id": "C8",
                "text": "Results of any prior neurological testing or imaging with dates and findings.",
                "section": "documentation",
            },
            {
                "id": "C9",
                "text": "Current medication list including any neurological medications previously tried, their dosages, duration, and outcomes.",
                "section": "documentation",
            },
        ],
        "logic_tree": {
            "type": "AND",
            "children": [
                {
                    "type": "OR",
                    "label": "Clinical Indication",
                    "children": ["C1", "C2", "C3", "C4"],
                },
                {
                    "type": "AND",
                    "label": "No Exclusions",
                    "children": [
                        {"type": "NOT", "child": "C5"},
                        {"type": "NOT", "child": "C6"},
                    ],
                },
                {
                    "type": "AND",
                    "label": "Documentation Complete",
                    "children": ["C7", "C8"],
                },
            ],
        },
    },
    "J0717": {
        "procedure_name": "Biologic Injection — Adalimumab (Humira)",
        "category": "BRAND_DRUG",
        "indications": [
            {
                "id": "C1",
                "text": "Patient has a confirmed diagnosis of moderate to severe rheumatoid arthritis (ICD-10 M05-M06) with inadequate response to at least one conventional DMARD such as methotrexate at therapeutic doses for a minimum of 3 months.",
                "section": "indications",
            },
            {
                "id": "C2",
                "text": "Patient has a confirmed diagnosis of moderate to severe Crohn disease (ICD-10 K50) or ulcerative colitis (ICD-10 K51) with inadequate response to conventional therapy including corticosteroids and immunomodulators.",
                "section": "indications",
            },
            {
                "id": "C3",
                "text": "Patient has a confirmed diagnosis of moderate to severe plaque psoriasis (ICD-10 L40.0) involving more than 10 percent body surface area or affecting critical areas, with failure of at least one systemic therapy such as methotrexate, cyclosporine, or phototherapy.",
                "section": "indications",
            },
            {
                "id": "C4",
                "text": "Patient has a confirmed diagnosis of ankylosing spondylitis (ICD-10 M45) with inadequate response to at least two NSAIDs at therapeutic doses over a combined period of at least 4 weeks.",
                "section": "indications",
            },
        ],
        "contraindications": [
            {
                "id": "C5",
                "text": "Active serious infection including tuberculosis. Patients must have a negative TB screening test (PPD or interferon-gamma release assay) within the past 12 months before initiation.",
                "section": "contraindications",
            },
            {
                "id": "C6",
                "text": "Not covered when a biosimilar equivalent (e.g., adalimumab-atto, adalimumab-adbm) is available and has not been tried, unless documented medical reason for brand-specific requirement.",
                "section": "contraindications",
            },
            {
                "id": "C7",
                "text": "Not covered as first-line therapy without documented trial and failure of conventional treatments appropriate for the specific indication.",
                "section": "contraindications",
            },
        ],
        "documentation": [
            {
                "id": "C8",
                "text": "Confirmed diagnosis with supporting clinical findings, relevant laboratory results (e.g., RF, anti-CCP, ESR, CRP for RA), and disease severity assessment using validated scoring tools.",
                "section": "documentation",
            },
            {
                "id": "C9",
                "text": "Complete step therapy documentation: names of prior conventional treatments tried, dosages, duration of each trial, and specific reasons for failure or intolerance.",
                "section": "documentation",
            },
            {
                "id": "C10",
                "text": "TB screening results within the past 12 months, hepatitis B screening, and documentation that patient is not currently experiencing active serious infection.",
                "section": "documentation",
            },
            {
                "id": "C11",
                "text": "Prescribing physician must be a relevant specialist (rheumatologist, gastroenterologist, or dermatologist) or have documented consultation with one.",
                "section": "documentation",
            },
        ],
        "logic_tree": {
            "type": "AND",
            "children": [
                {
                    "type": "OR",
                    "label": "Approved Indication",
                    "children": ["C1", "C2", "C3", "C4"],
                },
                {
                    "type": "AND",
                    "label": "No Contraindications",
                    "children": [
                        {"type": "NOT", "child": "C5"},
                        {"type": "NOT", "child": "C6"},
                        {"type": "NOT", "child": "C7"},
                    ],
                },
                {
                    "type": "AND",
                    "label": "Documentation Complete",
                    "children": ["C8", "C9", "C10", "C11"],
                },
            ],
        },
    },
    "J2357": {
        "procedure_name": "Specialty Injection — Omalizumab (Xolair)",
        "category": "BRAND_DRUG",
        "indications": [
            {
                "id": "C1",
                "text": "Patient has moderate to severe persistent allergic asthma (ICD-10 J45.40-J45.51) with positive skin test or in vitro reactivity to a perennial aeroallergen and inadequate control despite high-dose inhaled corticosteroid plus long-acting beta-agonist therapy.",
                "section": "indications",
            },
            {
                "id": "C2",
                "text": "Patient has chronic idiopathic urticaria (ICD-10 L50.1) refractory to H1 antihistamine therapy at up to four times the standard dose for at least 6 weeks.",
                "section": "indications",
            },
            {
                "id": "C3",
                "text": "Patient has chronic rhinosinusitis with nasal polyps (ICD-10 J33.0-J33.9) with inadequate response to intranasal corticosteroids and at least one prior sinus surgery or contraindication to surgery.",
                "section": "indications",
            },
        ],
        "contraindications": [
            {
                "id": "C4",
                "text": "Not covered for patients with serum IgE levels outside the dosing range specified in the FDA-approved labeling (30-1500 IU/mL for asthma indication).",
                "section": "contraindications",
            },
            {
                "id": "C5",
                "text": "Not covered as first-line therapy without documented failure of step therapy with appropriate controller medications for the specific indication.",
                "section": "contraindications",
            },
        ],
        "documentation": [
            {
                "id": "C6",
                "text": "Confirmed diagnosis with pulmonary function testing (FEV1, FVC) for asthma indication, or clinical documentation for urticaria/nasal polyp indications.",
                "section": "documentation",
            },
            {
                "id": "C7",
                "text": "Baseline serum total IgE level with date, body weight, and allergy testing results documenting sensitization to perennial aeroallergen (for asthma indication).",
                "section": "documentation",
            },
            {
                "id": "C8",
                "text": "Documentation of current controller medication regimen including inhaled corticosteroid dose, adherence assessment, and inhaler technique evaluation.",
                "section": "documentation",
            },
            {
                "id": "C9",
                "text": "Record of exacerbation frequency, emergency department visits, hospitalizations, and oral corticosteroid courses in the preceding 12 months.",
                "section": "documentation",
            },
        ],
        "logic_tree": {
            "type": "AND",
            "children": [
                {
                    "type": "OR",
                    "label": "Approved Indication",
                    "children": ["C1", "C2", "C3"],
                },
                {
                    "type": "AND",
                    "label": "No Contraindications",
                    "children": [
                        {"type": "NOT", "child": "C4"},
                        {"type": "NOT", "child": "C5"},
                    ],
                },
                {
                    "type": "AND",
                    "label": "Documentation Complete",
                    "children": ["C6", "C7", "C8", "C9"],
                },
            ],
        },
    },
}

# ── Payer-specific variation rules ─────────────────────────────────────────
# Real payers have slightly different thresholds and requirements.
# These variations are applied on top of the base templates.

PAYER_VARIATIONS = {
    "UHC": {
        "extra_note": "UnitedHealthcare requires use of their online prior authorization portal at uhcprovider.com for all imaging and specialty drug requests.",
        "step_therapy_strict": True,
        "appeal_window_days": 180,
        "peer_to_peer_available": True,
    },
    "AETNA": {
        "extra_note": "Aetna utilizes Carelon (formerly AIM Specialty Health) for radiology prior authorization management.",
        "step_therapy_strict": True,
        "appeal_window_days": 180,
        "peer_to_peer_available": True,
    },
    "CIGNA": {
        "extra_note": "Cigna utilizes eviCore healthcare for medical benefit management including imaging and specialty drug review.",
        "step_therapy_strict": True,
        "appeal_window_days": 180,
        "peer_to_peer_available": True,
    },
    "HUMANA": {
        "extra_note": "Humana requires prior authorization through their provider portal or by calling the PA department directly.",
        "step_therapy_strict": False,
        "appeal_window_days": 60,
        "peer_to_peer_available": True,
    },
    "ELEVANCE": {
        "extra_note": "Elevance Health (Anthem) uses AIM Specialty Health for radiology management and Express Scripts for pharmacy benefit.",
        "step_therapy_strict": True,
        "appeal_window_days": 180,
        "peer_to_peer_available": True,
    },
    "CENTENE": {
        "extra_note": "Centene subsidiary plans may have state-specific PA requirements. Verify with the specific plan for Medicaid managed care members.",
        "step_therapy_strict": False,
        "appeal_window_days": 30,
        "peer_to_peer_available": False,
    },
    "KAISER": {
        "extra_note": "Kaiser Permanente integrates PA within its closed-system model. Referrals for out-of-network services require additional authorization.",
        "step_therapy_strict": True,
        "appeal_window_days": 180,
        "peer_to_peer_available": True,
    },
}


def generate_policy_json(payer_id: str, cpt_code: str, template: dict) -> dict:
    """Build a complete policy JSON document for one payer-procedure pair."""
    payer_info = PAYER_VARIATIONS.get(payer_id, {})
    policy_id = f"{payer_id}_{cpt_code}"

    all_criteria = (
        template["indications"]
        + template["contraindications"]
        + template["documentation"]
    )

    # Build full text from criteria
    sections = {"indications": [], "contraindications": [], "documentation": []}
    for c in all_criteria:
        sections[c["section"]].append(f"{c['id']}. {c['text']}")

    text_parts = []
    if sections["indications"]:
        text_parts.append("INDICATIONS AND COVERAGE CRITERIA:\n" + "\n\n".join(sections["indications"]))
    if sections["contraindications"]:
        text_parts.append("CONTRAINDICATIONS AND EXCLUSIONS:\n" + "\n\n".join(sections["contraindications"]))
    if sections["documentation"]:
        text_parts.append("DOCUMENTATION REQUIREMENTS:\n" + "\n\n".join(sections["documentation"]))

    extra = payer_info.get("extra_note", "")
    if extra:
        text_parts.append(f"PAYER-SPECIFIC NOTES:\n{extra}")

    if payer_info.get("appeal_window_days"):
        text_parts.append(
            f"APPEAL INFORMATION:\nAppeals must be filed within {payer_info['appeal_window_days']} days of the adverse determination. "
            + ("Peer-to-peer review is available upon request." if payer_info.get("peer_to_peer_available") else "")
        )

    full_text = "\n\n".join(text_parts)

    return {
        "policy_id": policy_id,
        "payer_id": payer_id,
        "procedure_code": cpt_code,
        "policy_name": f"{payer_id} Coverage Policy — {template['procedure_name']} (CPT {cpt_code})",
        "source_url": f"https://example.com/policies/{policy_id.lower()}",
        "text": full_text,
        "logic_tree": template.get("logic_tree"),
    }


def generate_all_policies():
    """Generate policy files for all payer-CPT combinations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for payer_id in PAYERS:
        for cpt_code, template in POLICY_TEMPLATES.items():
            policy = generate_policy_json(payer_id, cpt_code, template)
            filename = f"{payer_id}_{cpt_code}.json"
            filepath = OUTPUT_DIR / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(policy, f, indent=2, ensure_ascii=False)

            count += 1

    print(f"Generated {count} policy files in {OUTPUT_DIR}/")
    print(f"  Payers: {', '.join(PAYERS)}")
    print(f"  CPT codes: {', '.join(POLICY_TEMPLATES.keys())}")
    print(f"\nNext steps:")
    print(f"  1. Click 'Index Policy Documents' in the Streamlit sidebar")
    print(f"  2. Or run: python -c \"from src.retrieval.indexer import init_db, bulk_index_from_directory; "
          f"conn = init_db(); print(f'Indexed {{bulk_index_from_directory(\\\"data/policies\\\", conn)}} chunks')\"")


if __name__ == "__main__":
    generate_all_policies()