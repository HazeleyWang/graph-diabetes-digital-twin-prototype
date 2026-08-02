"""Test whether causal previous-encounter information adds validation value."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.train_baseline import (
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    fit_encoder,
    fit_logistic,
    metric_bundle,
    prepare_data,
    sigmoid,
    transform,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "temporal_baseline_results.md"
LOCAL_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "temporal_logistic_baseline.pkl"

TEMPORAL_NUMERIC = [
    "observed_prior_encounters",
    "previous_time_in_hospital",
    "previous_num_lab_procedures",
    "previous_num_medications",
    "previous_number_diagnoses",
]
TEMPORAL_CATEGORICAL = [
    "previous_diag_1_group",
    "previous_A1Cresult",
    "previous_insulin",
    "previous_change",
]


def add_causal_history(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["encounter_order"] = pd.to_numeric(data["encounter_id"], errors="raise")
    data = data.sort_values(["patient_nbr", "encounter_order"], kind="stable")
    data["observed_prior_encounters"] = data.groupby("patient_nbr").cumcount()

    numeric_sources = {
        "time_in_hospital": "previous_time_in_hospital",
        "num_lab_procedures": "previous_num_lab_procedures",
        "num_medications": "previous_num_medications",
        "number_diagnoses": "previous_number_diagnoses",
    }
    categorical_sources = {
        "diag_1_group": "previous_diag_1_group",
        "A1Cresult": "previous_A1Cresult",
        "insulin": "previous_insulin",
        "change": "previous_change",
    }
    for source, target in numeric_sources.items():
        data[target] = data.groupby("patient_nbr")[source].shift(1)
    for source, target in categorical_sources.items():
        data[target] = data.groupby("patient_nbr")[source].shift(1).fillna("No_prior_encounter")
    return data.sort_index()


def main() -> None:
    data = add_causal_history(prepare_data())
    train = data[data["split"] == "train"].reset_index(drop=True)
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    y_validation = validation["target_30d"].to_numpy()

    with MODEL_PATH.open("rb") as model_file:
        tabular = pickle.load(model_file)
    tabular_probability = sigmoid(
        transform(validation, tabular["encoder"]) @ tabular["weights"] + tabular["bias"]
    )
    tabular_metrics = metric_bundle(y_validation, tabular_probability)

    numeric_features = NUMERIC_FEATURES + TEMPORAL_NUMERIC
    categorical_features = CATEGORICAL_FEATURES + TEMPORAL_CATEGORICAL
    temporal_encoder = fit_encoder(
        train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    temporal_x = transform(train, temporal_encoder)
    temporal_weights, temporal_bias = fit_logistic(
        temporal_x, train["target_30d"].to_numpy()
    )
    temporal_probability = sigmoid(
        transform(validation, temporal_encoder) @ temporal_weights + temporal_bias
    )
    temporal_metrics = metric_bundle(y_validation, temporal_probability)

    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {
                "encoder": temporal_encoder,
                "weights": temporal_weights,
                "bias": temporal_bias,
            },
            model_file,
        )

    auc_gain = temporal_metrics["auroc"] - tabular_metrics["auroc"]
    ap_gain = temporal_metrics["average_precision"] - tabular_metrics["average_precision"]
    repeated = validation["observed_prior_encounters"] > 0
    repeated_tabular = metric_bundle(y_validation[repeated], tabular_probability[repeated])
    repeated_temporal = metric_bundle(y_validation[repeated], temporal_probability[repeated])

    report = f"""# Causal one-step temporal baseline

## Question

Does adding information from the immediately preceding observed encounter improve ranking beyond the current-encounter tabular baseline?

The added history features are observed prior-encounter count plus the previous encounter's length of stay, laboratory-procedure count, medication count, diagnosis count, primary diagnosis group, HbA1c-test category, insulin status, and diabetes-medication-change status. Previous readmission labels are deliberately excluded.

## Validation results

| Population / model | N | AUROC | Average precision | Brier |
|---|---:|---:|---:|---:|
| All encounters — tabular | {len(validation):,} | {tabular_metrics['auroc']:.3f} | {tabular_metrics['average_precision']:.3f} | {tabular_metrics['brier_score']:.3f} |
| All encounters — + causal history | {len(validation):,} | {temporal_metrics['auroc']:.3f} | {temporal_metrics['average_precision']:.3f} | {temporal_metrics['brier_score']:.3f} |
| Repeated encounters — tabular | {int(repeated.sum()):,} | {repeated_tabular['auroc']:.3f} | {repeated_tabular['average_precision']:.3f} | {repeated_tabular['brier_score']:.3f} |
| Repeated encounters — + causal history | {int(repeated.sum()):,} | {repeated_temporal['auroc']:.3f} | {repeated_temporal['average_precision']:.3f} | {repeated_temporal['brier_score']:.3f} |

Across all validation encounters, causal history changed AUROC by **{100 * auc_gain:+.2f} percentage points** and average precision by **{100 * ap_gain:+.2f} percentage points**.

## Interpretation

- This is a graph-derived lag-feature baseline, not a graph neural network.
- A gain would show that the previous node carries incremental information; no gain would warn against assuming that more complex message passing will help.
- The repeated-encounter subgroup is the population that can actually receive a temporal message. Its result is therefore more informative than the overall average.
- Model design uses validation data only. Test performance remains uninspected until the temporal architecture is locked.
- Encounter-ID ordering is a chronology surrogate, not a verified timestamp.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

