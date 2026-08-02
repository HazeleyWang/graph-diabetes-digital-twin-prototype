"""Validate and render the modelling data dictionary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "data_dictionary.md"

# name: (stage, modelling role, explanation)
VARIABLES: dict[str, tuple[str, str, str]] = {
    "encounter_id": ("identifier", "exclude", "Encounter key; retain only for traceability."),
    "patient_nbr": ("identifier", "grouping", "Patient key for grouped splits and longitudinal edges."),
    "race": ("admission-known", "feature+audit", "Sensitive demographic; include cautiously and audit subgroup performance."),
    "gender": ("admission-known", "feature+audit", "Sensitive demographic with three invalid/unknown records."),
    "age": ("admission-known", "feature+audit", "Age band; ordinal encoding is possible but should preserve non-linearity."),
    "weight": ("admission-known", "sensitivity-only", "96.86% missing; exclude from the primary model."),
    "admission_type_id": ("admission-known", "feature", "Emergency, urgent, elective, or other admission type."),
    "discharge_disposition_id": ("discharge-known", "cohort-rule", "Can encode death/hospice and create an outcome shortcut; use for exclusions/sensitivity, not the primary feature set."),
    "admission_source_id": ("admission-known", "feature", "Referral, transfer, emergency room, or other source."),
    "time_in_hospital": ("during-stay", "feature", "Length of stay; available at discharge."),
    "payer_code": ("admission-known", "sensitivity-only", "Administrative, 39.56% missing, and vulnerable to site/time shift."),
    "medical_specialty": ("during-stay", "optional-feature", "Clinically useful but high-cardinality and 49.08% missing."),
    "num_lab_procedures": ("during-stay", "feature", "Number of laboratory procedures during the encounter."),
    "num_procedures": ("during-stay", "feature", "Number of non-laboratory procedures."),
    "num_medications": ("during-stay", "feature", "Number of distinct generic medication names."),
    "number_outpatient": ("history-known", "feature", "Outpatient visits in the preceding year."),
    "number_emergency": ("history-known", "feature", "Emergency visits in the preceding year."),
    "number_inpatient": ("history-known", "feature", "Inpatient visits in the preceding year."),
    "diag_1": ("during-stay", "feature", "Primary diagnosis; group ICD-9 codes before modelling."),
    "diag_2": ("during-stay", "feature", "Secondary diagnosis; group ICD-9 codes before modelling."),
    "diag_3": ("during-stay", "feature", "Additional diagnosis; group ICD-9 codes before modelling."),
    "number_diagnoses": ("during-stay", "feature", "Number of diagnoses entered into the system."),
    "max_glu_serum": ("during-stay", "feature", "Test result; `None` usually means not measured, not a random missing value."),
    "A1Cresult": ("during-stay", "feature", "HbA1c result; `None` usually means not measured."),
    "metformin": ("discharge-known", "feature", "Drug status/dose change during the encounter."),
    "repaglinide": ("discharge-known", "feature", "Drug status/dose change during the encounter."),
    "nateglinide": ("discharge-known", "feature", "Drug status/dose change during the encounter."),
    "chlorpropamide": ("discharge-known", "feature", "Drug status/dose change during the encounter."),
    "glimepiride": ("discharge-known", "feature", "Drug status/dose change during the encounter."),
    "acetohexamide": ("discharge-known", "optional-feature", "Extremely rare drug status; may be removed after training-only frequency checks."),
    "glipizide": ("discharge-known", "feature", "Drug status/dose change during the encounter."),
    "glyburide": ("discharge-known", "feature", "Drug status/dose change during the encounter."),
    "tolbutamide": ("discharge-known", "optional-feature", "Rare drug status; may be grouped or removed."),
    "pioglitazone": ("discharge-known", "feature", "Drug status/dose change during the encounter."),
    "rosiglitazone": ("discharge-known", "feature", "Drug status/dose change during the encounter."),
    "acarbose": ("discharge-known", "optional-feature", "Rare drug status; may be grouped or removed."),
    "miglitol": ("discharge-known", "optional-feature", "Rare drug status; may be grouped or removed."),
    "troglitazone": ("discharge-known", "optional-feature", "Rare drug status; may be grouped or removed."),
    "tolazamide": ("discharge-known", "optional-feature", "Rare drug status; may be grouped or removed."),
    "examide": ("discharge-known", "exclude", "Constant in this dataset; contains no predictive information."),
    "citoglipton": ("discharge-known", "exclude", "Constant in this dataset; contains no predictive information."),
    "insulin": ("discharge-known", "feature", "Insulin status/dose change during the encounter."),
    "glyburide-metformin": ("discharge-known", "optional-feature", "Combination-drug status; sparse categories expected."),
    "glipizide-metformin": ("discharge-known", "optional-feature", "Combination-drug status; sparse categories expected."),
    "glimepiride-pioglitazone": ("discharge-known", "optional-feature", "Combination-drug status; sparse categories expected."),
    "metformin-rosiglitazone": ("discharge-known", "optional-feature", "Combination-drug status; sparse categories expected."),
    "metformin-pioglitazone": ("discharge-known", "optional-feature", "Combination-drug status; sparse categories expected."),
    "change": ("discharge-known", "feature", "Whether diabetes medication changed during the encounter."),
    "diabetesMed": ("discharge-known", "feature", "Whether any diabetes medication was prescribed."),
    "readmitted": ("future-outcome", "target", "Outcome: `<30` is positive for 30-day readmission."),
}


def main() -> None:
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    actual = list(df.columns)
    missing = sorted(set(actual) - set(VARIABLES))
    extra = sorted(set(VARIABLES) - set(actual))
    if missing or extra:
        raise ValueError(f"Dictionary/schema mismatch; missing={missing}, extra={extra}")

    rows = []
    for name in actual:
        stage, role, explanation = VARIABLES[name]
        rows.append(
            f"| `{name}` | {stage} | {role} | {df[name].nunique():,} | {explanation} |"
        )

    report = """# Modelling data dictionary

**Locked prediction time:** discharge from the index hospital encounter.  
**Target:** whether the patient is readmitted in fewer than 30 days.

This timing permits admission, prior-utilisation, during-stay, and discharge-known information. It forbids future outcome information and requires special handling of discharge disposition because death or hospice can make readmission structurally impossible.

| Variable | Available stage | Proposed role | Unique values | Interpretation / decision |
|---|---|---|---:|---|
""" + "\n".join(rows) + """

## Locked primary-analysis rules

1. Group train/validation/test splits by `patient_nbr`.
2. Exclude `encounter_id`, `patient_nbr`, `examide`, and `citoglipton` from model inputs.
3. Exclude `weight` and `payer_code` from the primary model; revisit only in sensitivity analyses.
4. Use `discharge_disposition_id` to define clinically eligible cohorts and sensitivity analyses, not as a primary predictor.
5. Preserve “test not performed” for `A1Cresult` and `max_glu_serum` as an informative category rather than generic imputation.
6. Derive every frequency threshold, imputation rule, encoding, and graph edge definition using training data only.
"""
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Validated {len(actual)} variables and wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

