"""Create deterministic, leakage-safe patient-grouped dataset splits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
ASSIGNMENT_PATH = PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "split_audit.md"
INELIGIBLE_DISPOSITIONS = {"11", "13", "14", "19", "20", "21"}


def patient_split(patient_nbr: str) -> str:
    """Map a patient deterministically to an approximately 70/15/15 split."""
    bucket = int(hashlib.sha256(f"graph-twin-v1:{patient_nbr}".encode()).hexdigest()[:8], 16) % 1000
    if bucket < 700:
        return "train"
    if bucket < 850:
        return "validation"
    return "test"


def main() -> None:
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    df["split"] = df["patient_nbr"].map(patient_split)
    df["eligible_primary"] = ~df["discharge_disposition_id"].isin(INELIGIBLE_DISPOSITIONS)
    df["target_30d"] = (df["readmitted"] == "<30").astype(int)

    patient_split_counts = df[["patient_nbr", "split"]].drop_duplicates()["split"].value_counts()
    if df.groupby("patient_nbr")["split"].nunique().max() != 1:
        raise AssertionError("Patient leakage detected across splits")

    assignments = df[["encounter_id", "patient_nbr", "split", "eligible_primary"]]
    ASSIGNMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(ASSIGNMENT_PATH, index=False)

    eligible = df[df["eligible_primary"]].copy()
    summary = (
        eligible.groupby("split")
        .agg(
            encounters=("encounter_id", "size"),
            patients=("patient_nbr", "nunique"),
            positives_30d=("target_30d", "sum"),
            positive_rate=("target_30d", "mean"),
        )
        .reindex(["train", "validation", "test"])
    )

    table_rows = []
    for split, row in summary.iterrows():
        table_rows.append(
            f"| {split} | {int(row.encounters):,} | {int(row.patients):,} | "
            f"{int(row.positives_30d):,} | {100 * row.positive_rate:.2f}% |"
        )

    excluded = int((~df["eligible_primary"]).sum())
    max_rate_gap = 100 * (summary["positive_rate"].max() - summary["positive_rate"].min())
    patient_total = int(patient_split_counts.sum())
    patient_distribution = ", ".join(
        f"{name}={count:,} ({100 * count / patient_total:.2f}%)"
        for name, count in patient_split_counts.reindex(["train", "validation", "test"]).items()
    )

    report = f"""# Patient-grouped split audit

## Definition

- Split ratio: approximately 70% train, 15% validation, 15% test.
- Assignment unit: patient, using a salted deterministic SHA-256 bucket.
- Primary cohort excludes discharge dispositions representing death or hospice: {sorted(INELIGIBLE_DISPOSITIONS)}.
- The row-level assignment file is stored locally at `data/processed/split_assignments.csv` and is ignored by Git.

## Integrity checks

- Patients present in more than one split: **0**.
- Patient allocation: {patient_distribution}.
- Encounters excluded from the primary cohort: **{excluded:,}**.
- Maximum difference in 30-day-positive rate between splits: **{max_rate_gap:.2f} percentage points**.

## Primary-cohort distribution

| Split | Encounters | Patients | 30-day positives | Positive rate |
|---|---:|---:|---:|---:|
{"\n".join(table_rows)}

## Interpretation

The test set is now a genuine unseen-patient evaluation. Validation is used for model and threshold choices; test remains untouched until the analysis pipeline is locked. Similar target rates across splits indicate that deterministic grouping did not introduce a material class-balance shift.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote local assignments to {ASSIGNMENT_PATH}")
    print(f"Wrote audit report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

