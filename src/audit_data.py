"""Generate a reproducible first-pass audit of the raw UCI diabetes dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_audit.md"
MISSING_TOKENS = {"?", "Unknown/Invalid", "NULL", "None", ""}


def pct(value: int, total: int) -> str:
    return f"{100 * value / total:.2f}%"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Follow data/README.md to download the dataset."
        )

    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    n_rows, n_cols = df.shape
    unique_encounters = df["encounter_id"].nunique()
    unique_patients = df["patient_nbr"].nunique()
    encounters_per_patient = df.groupby("patient_nbr", sort=False).size()
    repeated_patients = int((encounters_per_patient > 1).sum())

    readmission_counts = df["readmitted"].value_counts(dropna=False)
    missing_rows = []
    for column in df.columns:
        count = int(df[column].isin(MISSING_TOKENS).sum())
        if count:
            missing_rows.append((column, count, 100 * count / n_rows))
    missing_rows.sort(key=lambda item: item[1], reverse=True)

    exact_duplicates = int(df.duplicated().sum())
    duplicate_encounters = n_rows - unique_encounters
    positive_30d = int((df["readmitted"] == "<30").sum())

    target_table = "\n".join(
        f"| `{label}` | {count:,} | {pct(int(count), n_rows)} |"
        for label, count in readmission_counts.items()
    )
    missing_table = "\n".join(
        f"| `{column}` | {count:,} | {percentage:.2f}% |"
        for column, count, percentage in missing_rows
    )

    report = f"""# First-pass data audit

Generated from the local raw data by `python -m src.audit_data`.

## Dataset integrity

| Check | Result |
|---|---:|
| Encounter rows | {n_rows:,} |
| Columns | {n_cols} |
| Unique encounter IDs | {unique_encounters:,} |
| Duplicate encounter IDs | {duplicate_encounters:,} |
| Exact duplicate rows | {exact_duplicates:,} |
| Unique patients | {unique_patients:,} |
| Patients with multiple encounters | {repeated_patients:,} ({pct(repeated_patients, unique_patients)}) |
| Maximum encounters for one patient | {int(encounters_per_patient.max()):,} |

## Readmission target

| Original label | Encounters | Share |
|---|---:|---:|
{target_table}

For a binary 30-day-readmission endpoint, `<30` is the positive class: **{positive_30d:,} encounters ({pct(positive_30d, n_rows)})**. The class imbalance means accuracy alone is unsuitable; report AUROC, average precision, calibration, sensitivity/specificity, and confidence intervals.

## Explicit missing-value tokens

The source uses strings such as `?` and `Unknown/Invalid` rather than ordinary null values. Columns without one of the configured tokens are omitted below.

| Column | Missing/unknown | Share |
|---|---:|---:|
{missing_table}

## Leakage and modelling decisions

1. **Split by `patient_nbr`, never by encounter row.** Repeated encounters from one patient must not cross train, validation, and test partitions.
2. **Do not use identifiers as predictive features.** Keep `encounter_id` only for traceability and `patient_nbr` only for grouping/graph construction.
3. **Define the prediction time explicitly.** Audit discharge-related variables before modelling; any field only known after the prediction point must be excluded.
4. **Fit preprocessing on training data only.** Imputation, category grouping, scaling, feature selection, and graph-neighbour construction must not inspect validation/test outcomes.
5. **Treat demographics as sensitive.** Evaluate race, gender, and age subgroups, but avoid causal or clinical claims from observational associations.
6. **Avoid a transductive evaluation shortcut.** If graph edges use patient similarity, test nodes must not leak labels or outcome-derived features into training.

## Recommended next step

Create a data dictionary that classifies every variable as identifier, demographic, admission-known, during-stay, discharge-known, outcome, or mapping-only. Then lock the prediction time and feature set before building a patient-grouped baseline split.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

