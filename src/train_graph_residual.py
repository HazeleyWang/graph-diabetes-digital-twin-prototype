"""Train a stable causal graph-residual model over previous encounters."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.train_baseline import MODEL_PATH, metric_bundle, prepare_data, sigmoid, transform
from src.train_graphsage_prototype import neighbour_matrix, ordered_split
from src.train_temporal_baseline import LOCAL_MODEL_PATH as TEMPORAL_MODEL_PATH
from src.train_temporal_baseline import add_causal_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "graph_residual.pkl"
REPORT_PATH = PROJECT_ROOT / "reports" / "graph_residual_results.md"


def fit_residual(
    previous_x: np.ndarray,
    base_logit: np.ndarray,
    y: np.ndarray,
    epochs: int = 40,
    batch_size: int = 2048,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(20260803)
    weights = np.zeros(previous_x.shape[1], dtype=np.float32)
    bias = np.float32(0.0)
    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = second_b = np.float32(0.0)
    learning_rate, beta1, beta2, epsilon, l2 = 0.005, 0.9, 0.999, 1e-8, 1e-3
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start : start + batch_size]
            xb, offset, yb = previous_x[batch], base_logit[batch], y[batch]
            residual = sigmoid(offset + xb @ weights + bias) - yb
            grad_w = xb.T @ residual / len(batch) + l2 * weights
            grad_b = residual.mean()
            step += 1
            first_w = beta1 * first_w + (1 - beta1) * grad_w
            second_w = beta2 * second_w + (1 - beta2) * grad_w * grad_w
            first_b = beta1 * first_b + (1 - beta1) * grad_b
            second_b = beta2 * second_b + (1 - beta2) * grad_b * grad_b
            weights -= learning_rate * (first_w / (1 - beta1**step)) / (
                np.sqrt(second_w / (1 - beta2**step)) + epsilon
            )
            bias -= learning_rate * (first_b / (1 - beta1**step)) / (
                np.sqrt(second_b / (1 - beta2**step)) + epsilon
            )
    return weights, float(bias)


def main() -> None:
    data = prepare_data()
    train, train_previous_index = ordered_split(data, "train")
    validation, validation_previous_index = ordered_split(data, "validation")
    y_train = train["target_30d"].to_numpy()
    y_validation = validation["target_30d"].to_numpy()

    with MODEL_PATH.open("rb") as model_file:
        tabular = pickle.load(model_file)
    train_x = transform(train, tabular["encoder"])
    validation_x = transform(validation, tabular["encoder"])
    train_previous = neighbour_matrix(train_x, train_previous_index)
    validation_previous = neighbour_matrix(validation_x, validation_previous_index)
    train_logit = train_x @ tabular["weights"] + tabular["bias"]
    validation_logit = validation_x @ tabular["weights"] + tabular["bias"]

    residual_weights, residual_bias = fit_residual(
        train_previous, train_logit, y_train
    )
    residual_probability = sigmoid(
        validation_logit + validation_previous @ residual_weights + residual_bias
    )
    residual_metrics = metric_bundle(y_validation, residual_probability)
    tabular_probability = sigmoid(validation_logit)
    tabular_metrics = metric_bundle(y_validation, tabular_probability)

    temporal_data = add_causal_history(data)
    temporal_validation = temporal_data[temporal_data["split"] == "validation"]
    temporal_validation = temporal_validation.set_index("encounter_id").loc[validation["encounter_id"]].reset_index()
    with TEMPORAL_MODEL_PATH.open("rb") as model_file:
        temporal = pickle.load(model_file)
    temporal_probability = sigmoid(
        transform(temporal_validation, temporal["encoder"]) @ temporal["weights"]
        + temporal["bias"]
    )
    temporal_metrics = metric_bundle(y_validation, temporal_probability)

    has_history = validation_previous_index >= 0
    repeated_residual = metric_bundle(y_validation[has_history], residual_probability[has_history])
    repeated_tabular = metric_bundle(y_validation[has_history], tabular_probability[has_history])

    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {
                "tabular_model": tabular,
                "residual_weights": residual_weights,
                "residual_bias": residual_bias,
                "direction": "current_to_previous",
            },
            model_file,
        )

    report = f"""# Stable causal graph-residual baseline

## Architecture

The fitted tabular log-odds are frozen as an offset. A directed message from the immediately previous encounter learns only a residual adjustment to the current encounter's log-odds. Previous labels are excluded.

This linear message-passing model is less expressive than GraphSAGE but has a simpler, more stable optimization target. It answers whether the full previous-node representation adds information after the tabular risk score is already known.

## Validation comparison

| Model | AUROC | Average precision | Brier |
|---|---:|---:|---:|
| Tabular logistic | {tabular_metrics['auroc']:.3f} | {tabular_metrics['average_precision']:.3f} | {tabular_metrics['brier_score']:.3f} |
| Hand-selected previous-encounter features | {temporal_metrics['auroc']:.3f} | {temporal_metrics['average_precision']:.3f} | {temporal_metrics['brier_score']:.3f} |
| Causal graph residual | {residual_metrics['auroc']:.3f} | {residual_metrics['average_precision']:.3f} | {residual_metrics['brier_score']:.3f} |

Among the **{int(has_history.sum()):,}** encounters with observed history, tabular AUROC was **{repeated_tabular['auroc']:.3f}** and graph-residual AUROC was **{repeated_residual['auroc']:.3f}**.

## Decision

This model is the stability check for the neural graph prototype. If it matches or exceeds the average neural result, the evidence favours a simpler causal graph adjustment rather than claiming benefit from nonlinear message passing. Test evaluation remains locked until the final temporal specification is selected.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

