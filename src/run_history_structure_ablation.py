"""Separate history availability, utilisation and previous-encounter detail."""

from __future__ import annotations

from pathlib import Path

from src.run_history_ablation import HISTORY_FEATURES
from src.train_baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    fit_encoder,
    fit_logistic,
    metric_bundle,
    prepare_data,
    sigmoid,
    transform,
)
from src.train_temporal_baseline import (
    TEMPORAL_CATEGORICAL,
    TEMPORAL_NUMERIC,
    add_causal_history,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "history_structure_ablation.md"


def fit_and_evaluate(train, validation, numeric_features, categorical_features):
    encoder = fit_encoder(
        train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    weights, bias = fit_logistic(
        transform(train, encoder), train["target_30d"].to_numpy()
    )
    probability = sigmoid(transform(validation, encoder) @ weights + bias)
    return metric_bundle(validation["target_30d"].to_numpy(), probability)


def main() -> None:
    data = add_causal_history(prepare_data())
    data["has_observed_history"] = (
        data["observed_prior_encounters"] > 0
    ).astype(float)
    train = data[data["split"] == "train"].reset_index(drop=True)
    validation = data[data["split"] == "validation"].reset_index(drop=True)

    no_utilisation = [feature for feature in NUMERIC_FEATURES if feature not in HISTORY_FEATURES]
    previous_detail_numeric = [
        feature for feature in TEMPORAL_NUMERIC if feature != "observed_prior_encounters"
    ]
    specifications = [
        ("Current encounter, no prior-utilisation counts", no_utilisation, CATEGORICAL_FEATURES),
        ("+ history availability only", no_utilisation + ["has_observed_history"], CATEGORICAL_FEATURES),
        ("+ observed history count", no_utilisation + ["observed_prior_encounters"], CATEGORICAL_FEATURES),
        ("+ previous-encounter detail", no_utilisation + previous_detail_numeric, CATEGORICAL_FEATURES + TEMPORAL_CATEGORICAL),
        ("+ count and previous-encounter detail", no_utilisation + TEMPORAL_NUMERIC, CATEGORICAL_FEATURES + TEMPORAL_CATEGORICAL),
        ("Full tabular utilisation + causal history", NUMERIC_FEATURES + TEMPORAL_NUMERIC, CATEGORICAL_FEATURES + TEMPORAL_CATEGORICAL),
    ]
    results = [
        (name, fit_and_evaluate(train, validation, numeric, categorical))
        for name, numeric, categorical in specifications
    ]
    rows = "\n".join(
        f"| {name} | {metrics['auroc']:.3f} | {metrics['average_precision']:.3f} | {metrics['brier_score']:.3f} |"
        for name, metrics in results
    )
    report = f"""# History-structure ablation

## Question

Does explicit encounter history contribute detailed clinical information, or mainly indicate that a patient has previously used hospital care?

All specifications use the same training and validation patients. Test data are not evaluated. Previous readmission outcomes are excluded from every specification.

## Validation results

| Specification | AUROC | Average precision | Brier |
|---|---:|---:|---:|
{rows}

## Reading the ablation

- **History availability** tests whether edge existence alone carries signal.
- **Observed history count** tests trajectory length without previous-node clinical content.
- **Previous-encounter detail** tests the latest node's measurements without prior-year utilisation counts.
- **Full utilisation + causal history** tests whether explicit history adds information after the existing aggregate history variables are already known.

These comparisons diagnose the source of predictive information; they do not establish a causal effect of healthcare utilisation or treatment.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
