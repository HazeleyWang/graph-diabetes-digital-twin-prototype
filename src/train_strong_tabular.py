"""Evaluate nonlinear tabular and temporal baselines on validation only."""

from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

from src.train_baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    fit_encoder,
    metric_bundle,
    prepare_data,
    transform,
)
from src.train_temporal_baseline import (
    TEMPORAL_CATEGORICAL,
    TEMPORAL_NUMERIC,
    add_causal_history,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "strong_tabular_validation.pkl"
REPORT_PATH = PROJECT_ROOT / "reports" / "strong_tabular_results.md"


def fit_boosted_model(x, y) -> HistGradientBoostingClassifier:
    """Fit a deterministic nonlinear reference without validation-set tuning."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=20260803,
    )
    return model.fit(x, y)


def evaluate_specification(train, validation, numeric_features, categorical_features):
    encoder = fit_encoder(
        train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    model = fit_boosted_model(
        transform(train, encoder),
        train["target_30d"].to_numpy(),
    )
    probability = model.predict_proba(transform(validation, encoder))[:, 1]
    return encoder, model, probability, metric_bundle(
        validation["target_30d"].to_numpy(), probability
    )


def main() -> None:
    data = add_causal_history(prepare_data())
    train = data[data["split"] == "train"].reset_index(drop=True)
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    y_validation = validation["target_30d"].to_numpy()

    current = evaluate_specification(
        train,
        validation,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )
    temporal = evaluate_specification(
        train,
        validation,
        NUMERIC_FEATURES + TEMPORAL_NUMERIC,
        CATEGORICAL_FEATURES + TEMPORAL_CATEGORICAL,
    )
    _, _, current_probability, current_metrics = current
    temporal_encoder, temporal_model, temporal_probability, temporal_metrics = temporal

    has_history = validation["observed_prior_encounters"].to_numpy() > 0
    repeated_current = metric_bundle(y_validation[has_history], current_probability[has_history])
    repeated_temporal = metric_bundle(y_validation[has_history], temporal_probability[has_history])

    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {
                "encoder": temporal_encoder,
                "model": temporal_model,
                "development_split": "validation",
                "test_evaluated": False,
            },
            model_file,
        )

    auc_change = temporal_metrics["auroc"] - current_metrics["auroc"]
    ap_change = temporal_metrics["average_precision"] - current_metrics["average_precision"]
    report = f"""# Strong nonlinear tabular baseline

## Purpose

This experiment checks whether the causal history result survives comparison with a nonlinear tabular learner. It is a development analysis: both specifications were evaluated on validation only, and the test set was not inspected.

## Fixed model

- Histogram gradient boosting with 150 iterations, learning rate 0.05 and 31 leaves.
- Preprocessing and category vocabularies are fitted on training data only.
- Hyperparameters were fixed before this run; no validation search or best-run selection was performed.
- The temporal specification adds the same one-step causal history used by the locked logistic model.

## Validation results

| Population / model | N | AUROC | Average precision | Brier |
|---|---:|---:|---:|---:|
| All encounters - nonlinear current-encounter baseline | {len(validation):,} | {current_metrics['auroc']:.3f} | {current_metrics['average_precision']:.3f} | {current_metrics['brier_score']:.3f} |
| All encounters - nonlinear + causal history | {len(validation):,} | {temporal_metrics['auroc']:.3f} | {temporal_metrics['average_precision']:.3f} | {temporal_metrics['brier_score']:.3f} |
| Encounters with history - nonlinear current baseline | {int(has_history.sum()):,} | {repeated_current['auroc']:.3f} | {repeated_current['average_precision']:.3f} | {repeated_current['brier_score']:.3f} |
| Encounters with history - nonlinear + causal history | {int(has_history.sum()):,} | {repeated_temporal['auroc']:.3f} | {repeated_temporal['average_precision']:.3f} | {repeated_temporal['brier_score']:.3f} |

Adding one-step history changed validation AUROC by **{100 * auc_change:+.2f} percentage points** and average precision by **{100 * ap_change:+.2f} percentage points** relative to the nonlinear baseline.

## Interpretation boundary

This is a stronger tabular comparator, not a newly selected final model. Its role is to test whether graph-derived history adds information beyond nonlinear interactions. Test evaluation would require a new pre-specified lock and is intentionally deferred.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
