"""Evaluate the locked temporal model once on test with patient bootstrap CIs."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.train_baseline import (
    MODEL_PATH,
    metric_bundle,
    prepare_data,
    select_youden_threshold,
    sigmoid,
    transform,
)
from src.train_temporal_baseline import LOCAL_MODEL_PATH, add_causal_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "locked_temporal_test.md"


def patient_bootstrap(
    y: np.ndarray,
    probability: np.ndarray,
    patients: np.ndarray,
    repeats: int = 500,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(20260803)
    unique_patients, patient_codes = np.unique(patients, return_inverse=True)
    groups = [np.flatnonzero(patient_codes == index) for index in range(len(unique_patients))]
    draws = {"auroc": [], "average_precision": [], "brier_score": []}
    for _ in range(repeats):
        sampled = rng.integers(0, len(unique_patients), len(unique_patients))
        row_indices = np.concatenate([groups[index] for index in sampled])
        result = metric_bundle(y[row_indices], probability[row_indices])
        for metric in draws:
            draws[metric].append(result[metric])
    return {
        metric: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for metric, values in draws.items()
    }


def main() -> None:
    data = add_causal_history(prepare_data())
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    y_validation = validation["target_30d"].to_numpy()
    y_test = test["target_30d"].to_numpy()

    with LOCAL_MODEL_PATH.open("rb") as model_file:
        temporal = pickle.load(model_file)
    validation_probability = sigmoid(
        transform(validation, temporal["encoder"]) @ temporal["weights"] + temporal["bias"]
    )
    test_probability = sigmoid(
        transform(test, temporal["encoder"]) @ temporal["weights"] + temporal["bias"]
    )
    threshold = select_youden_threshold(y_validation, validation_probability)
    temporal_test = metric_bundle(y_test, test_probability, threshold)
    confidence = patient_bootstrap(
        y_test,
        test_probability,
        test["patient_nbr"].to_numpy(),
    )

    with MODEL_PATH.open("rb") as model_file:
        tabular = pickle.load(model_file)
    tabular_probability = sigmoid(
        transform(test, tabular["encoder"]) @ tabular["weights"] + tabular["bias"]
    )
    tabular_test = metric_bundle(y_test, tabular_probability)

    has_history = test["observed_prior_encounters"].to_numpy() > 0
    history_metrics = metric_bundle(y_test[has_history], test_probability[has_history], threshold)
    first_metrics = metric_bundle(y_test[~has_history], test_probability[~has_history], threshold)

    report = f"""# Locked causal temporal model: test evaluation

## Pre-test lock

The hand-selected one-step temporal logistic model was selected before this evaluation because it had the strongest stable validation AUROC (0.648), while neural GraphSAGE was initialization-sensitive. No test result was used to choose this specification.

The operating threshold **{threshold:.3f}** was selected on validation data by Youden's J and then applied unchanged to test. Confidence intervals use 500 patient-level bootstrap samples, keeping repeated encounters clustered by patient.

## Test results

| Metric | Estimate | Patient-bootstrap 95% CI |
|---|---:|---:|
| AUROC | {temporal_test['auroc']:.3f} | {confidence['auroc'][0]:.3f}–{confidence['auroc'][1]:.3f} |
| Average precision | {temporal_test['average_precision']:.3f} | {confidence['average_precision'][0]:.3f}–{confidence['average_precision'][1]:.3f} |
| Brier score | {temporal_test['brier_score']:.3f} | {confidence['brier_score'][0]:.3f}–{confidence['brier_score'][1]:.3f} |
| Precision | {temporal_test['precision']:.3f} | — |
| Sensitivity | {temporal_test['recall_sensitivity']:.3f} | — |
| Specificity | {temporal_test['specificity']:.3f} | — |

For context, the locked tabular test AUROC was **{tabular_test['auroc']:.3f}** and average precision was **{tabular_test['average_precision']:.3f}**. The temporal model changed test AUROC by **{100 * (temporal_test['auroc'] - tabular_test['auroc']):+.2f} percentage points** and average precision by **{100 * (temporal_test['average_precision'] - tabular_test['average_precision']):+.2f} percentage points**.

## History-availability diagnostic

| Population | N | AUROC | Average precision | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Has previous observed encounter | {int(has_history.sum()):,} | {history_metrics['auroc']:.3f} | {history_metrics['average_precision']:.3f} | {history_metrics['recall_sensitivity']:.3f} | {history_metrics['specificity']:.3f} |
| First observed encounter | {int((~has_history).sum()):,} | {first_metrics['auroc']:.3f} | {first_metrics['average_precision']:.3f} | {first_metrics['recall_sensitivity']:.3f} | {first_metrics['specificity']:.3f} |

## Interpretation boundary

This is a research prototype on retrospective, de-identified data from 1999–2008. It is not clinically validated. Encounter ordering is a surrogate, calibration requires deeper assessment, and external validation is absent. The honest contribution is a leakage-aware comparison showing whether limited observed history adds incremental predictive value.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

