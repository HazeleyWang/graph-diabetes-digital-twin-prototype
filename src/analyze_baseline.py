"""Estimate validation-set permutation importance for the logistic baseline."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_baseline import (
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    metric_bundle,
    prepare_data,
    sigmoid,
    transform,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_importance.md"
LOCAL_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_permutation_importance.csv"


def main(repeats: int = 5) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run `python -m src.train_baseline` first.")
    with MODEL_PATH.open("rb") as model_file:
        artifact = pickle.load(model_file)

    validation = prepare_data()
    validation = validation[validation["split"] == "validation"].reset_index(drop=True)
    y = validation["target_30d"].to_numpy()
    baseline_probability = sigmoid(
        transform(validation, artifact["encoder"]) @ artifact["weights"] + artifact["bias"]
    )
    baseline_auc = metric_bundle(y, baseline_probability)["auroc"]

    rng = np.random.default_rng(20260803)
    rows = []
    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        drops = []
        original = validation[feature].to_numpy(copy=True)
        for _ in range(repeats):
            permuted = validation.copy()
            permuted[feature] = rng.permutation(original)
            probability = sigmoid(
                transform(permuted, artifact["encoder"]) @ artifact["weights"] + artifact["bias"]
            )
            permuted_auc = metric_bundle(y, probability)["auroc"]
            drops.append(baseline_auc - permuted_auc)
        rows.append(
            {
                "feature": feature,
                "mean_auroc_drop": float(np.mean(drops)),
                "sd_auroc_drop": float(np.std(drops, ddof=1)),
            }
        )

    importance = pd.DataFrame(rows).sort_values("mean_auroc_drop", ascending=False)
    LOCAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(LOCAL_RESULTS_PATH, index=False)

    positive = importance[importance["mean_auroc_drop"] > 0].head(15)
    table = "\n".join(
        f"| `{row.feature}` | {100 * row.mean_auroc_drop:.3f} | {100 * row.sd_auroc_drop:.3f} |"
        for row in positive.itertuples()
    )
    negligible = int((importance["mean_auroc_drop"].abs() < 0.0005).sum())

    report = f"""# Baseline permutation importance

## Method

The fixed logistic baseline was evaluated on validation data. One raw feature was shuffled at a time, five times, while all other columns were left unchanged. Importance is the decrease in AUROC relative to the unshuffled validation AUROC of **{baseline_auc:.3f}**.

This measures how much the fitted model relies on a feature for ranking validation patients. It does not establish causality, and correlated features can share or mask importance.

## Most influential feature families

| Feature | Mean AUROC decrease (percentage points) | SD |
|---|---:|---:|
{table}

Features with an absolute mean AUROC change below 0.05 percentage points: **{negligible} of {len(importance)}**. Small or negative values should be treated as negligible rather than evidence that a feature is protective.

## How this guides the project

1. The result identifies which tabular signals the graph model must complement rather than merely rediscover.
2. Prior-utilisation variables are especially relevant to a longitudinal patient graph because they summarize earlier encounters.
3. Weak individual medication fields may be grouped into higher-level medication-change representations to reduce sparsity.
4. Importance is calculated on validation only; the test set remains outside feature-selection decisions.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

