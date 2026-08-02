"""Quantify the added value of prior healthcare-utilisation features."""

from __future__ import annotations

import pickle
from pathlib import Path

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
REPORT_PATH = PROJECT_ROOT / "reports" / "history_ablation.md"
HISTORY_FEATURES = ["number_outpatient", "number_emergency", "number_inpatient"]


def main() -> None:
    data = prepare_data()
    train = data[data["split"] == "train"].reset_index(drop=True)
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    y_validation = validation["target_30d"].to_numpy()

    with MODEL_PATH.open("rb") as model_file:
        full = pickle.load(model_file)
    full_probability = sigmoid(
        transform(validation, full["encoder"]) @ full["weights"] + full["bias"]
    )
    full_metrics = metric_bundle(y_validation, full_probability)

    reduced_numeric = [feature for feature in NUMERIC_FEATURES if feature not in HISTORY_FEATURES]
    reduced_encoder = fit_encoder(
        train,
        numeric_features=reduced_numeric,
        categorical_features=CATEGORICAL_FEATURES,
    )
    reduced_x = transform(train, reduced_encoder)
    reduced_weights, reduced_bias = fit_logistic(
        reduced_x, train["target_30d"].to_numpy()
    )
    reduced_probability = sigmoid(
        transform(validation, reduced_encoder) @ reduced_weights + reduced_bias
    )
    reduced_metrics = metric_bundle(y_validation, reduced_probability)

    auc_loss = full_metrics["auroc"] - reduced_metrics["auroc"]
    ap_loss = full_metrics["average_precision"] - reduced_metrics["average_precision"]
    report = f"""# Prior-utilisation ablation

## Question

How much predictive information is lost when the model cannot use outpatient, emergency, and inpatient visit counts from the preceding year?

## Validation results

| Model | AUROC | Average precision | Brier score |
|---|---:|---:|---:|
| Full tabular baseline | {full_metrics['auroc']:.3f} | {full_metrics['average_precision']:.3f} | {full_metrics['brier_score']:.3f} |
| Without prior-utilisation counts | {reduced_metrics['auroc']:.3f} | {reduced_metrics['average_precision']:.3f} | {reduced_metrics['brier_score']:.3f} |

Removing the three prior-utilisation counts reduced AUROC by **{100 * auc_loss:.2f} percentage points** and average precision by **{100 * ap_loss:.2f} percentage points**.

## Interpretation

This controlled ablation supports the hypothesis that longitudinal healthcare history contains meaningful information beyond the current admission. It motivates a patient–encounter graph that represents earlier encounters explicitly rather than relying only on three aggregate counts. It does not yet prove that a graph neural network will outperform the baseline; that remains the next empirical question.

Only validation data is used for this model comparison. The test set is not used to decide whether longitudinal graph features should be developed.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

