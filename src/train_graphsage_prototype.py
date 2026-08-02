"""Train a one-layer causal GraphSAGE-style encounter model in NumPy."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_baseline import MODEL_PATH, metric_bundle, prepare_data, sigmoid, transform
from src.train_temporal_baseline import LOCAL_MODEL_PATH as TEMPORAL_MODEL_PATH
from src.train_temporal_baseline import add_causal_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_GRAPH_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "graphsage_prototype.pkl"
REPORT_PATH = PROJECT_ROOT / "reports" / "graphsage_results.md"


def ordered_split(data: pd.DataFrame, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = data[data["split"] == split].copy()
    frame["encounter_order"] = pd.to_numeric(frame["encounter_id"], errors="raise")
    frame = frame.sort_values(["patient_nbr", "encounter_order"], kind="stable").reset_index(drop=True)
    previous_id = frame.groupby("patient_nbr")["encounter_id"].shift(1)
    index_by_id = pd.Series(frame.index, index=frame["encounter_id"]).to_dict()
    previous_index = previous_id.map(index_by_id).fillna(-1).to_numpy(dtype=np.int32)
    return frame, previous_index


def neighbour_matrix(x: np.ndarray, previous_index: np.ndarray) -> np.ndarray:
    neighbours = np.zeros_like(x)
    available = previous_index >= 0
    neighbours[available] = x[previous_index[available]]
    return neighbours


def fit_graphsage(
    x: np.ndarray,
    x_previous: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 8,
    epochs: int = 22,
    batch_size: int = 2048,
    seed: int = 20260803,
) -> dict:
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2 / x.shape[1])
    params = {
        "w_self": rng.normal(0, scale, (x.shape[1], hidden_dim)).astype(np.float32),
        "w_previous": rng.normal(0, scale, (x.shape[1], hidden_dim)).astype(np.float32),
        "b_hidden": np.zeros(hidden_dim, dtype=np.float32),
        "w_output": rng.normal(0, np.sqrt(2 / hidden_dim), hidden_dim).astype(np.float32),
        "b_output": np.float32(0.0),
    }
    first = {name: np.zeros_like(value) for name, value in params.items()}
    second = {name: np.zeros_like(value) for name, value in params.items()}
    learning_rate, beta1, beta2, epsilon, l2 = 0.003, 0.9, 0.999, 1e-8, 1e-4
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start : start + batch_size]
            xb, pb, yb = x[batch], x_previous[batch], y[batch]
            hidden_linear = (
                xb @ params["w_self"]
                + pb @ params["w_previous"]
                + params["b_hidden"]
            )
            hidden = np.maximum(hidden_linear, 0)
            probability = sigmoid(hidden @ params["w_output"] + params["b_output"])
            delta = (probability - yb) / len(batch)
            grad_output = hidden.T @ delta + l2 * params["w_output"]
            grad_output_bias = delta.sum()
            grad_hidden = delta[:, None] * params["w_output"][None, :]
            grad_hidden[hidden_linear <= 0] = 0
            gradients = {
                "w_self": xb.T @ grad_hidden + l2 * params["w_self"],
                "w_previous": pb.T @ grad_hidden + l2 * params["w_previous"],
                "b_hidden": grad_hidden.sum(axis=0),
                "w_output": grad_output,
                "b_output": grad_output_bias,
            }
            step += 1
            for name in params:
                first[name] = beta1 * first[name] + (1 - beta1) * gradients[name]
                second[name] = beta2 * second[name] + (1 - beta2) * gradients[name] ** 2
                first_hat = first[name] / (1 - beta1**step)
                second_hat = second[name] / (1 - beta2**step)
                params[name] -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
    return params


def predict_graphsage(x: np.ndarray, x_previous: np.ndarray, params: dict) -> np.ndarray:
    hidden = np.maximum(
        x @ params["w_self"]
        + x_previous @ params["w_previous"]
        + params["b_hidden"],
        0,
    )
    return sigmoid(hidden @ params["w_output"] + params["b_output"])


def main() -> None:
    data = prepare_data()
    train, train_previous_index = ordered_split(data, "train")
    validation, validation_previous_index = ordered_split(data, "validation")
    y_train = train["target_30d"].to_numpy()
    y_validation = validation["target_30d"].to_numpy()

    with MODEL_PATH.open("rb") as model_file:
        tabular = pickle.load(model_file)
    x_train = transform(train, tabular["encoder"])
    x_validation = transform(validation, tabular["encoder"])
    train_previous = neighbour_matrix(x_train, train_previous_index)
    validation_previous = neighbour_matrix(x_validation, validation_previous_index)

    seed_results = []
    params = None
    graph_probability = None
    graph_metrics = None
    for seed in [20260803, 20260804, 20260805, 20260806, 20260807]:
        seed_params = fit_graphsage(x_train, train_previous, y_train, seed=seed)
        seed_probability = predict_graphsage(x_validation, validation_previous, seed_params)
        seed_metrics = metric_bundle(y_validation, seed_probability)
        seed_results.append(seed_metrics)
        if params is None:
            params = seed_params
            graph_probability = seed_probability
            graph_metrics = seed_metrics
    mean_auc = float(np.mean([result["auroc"] for result in seed_results]))
    sd_auc = float(np.std([result["auroc"] for result in seed_results], ddof=1))
    mean_ap = float(np.mean([result["average_precision"] for result in seed_results]))
    sd_ap = float(np.std([result["average_precision"] for result in seed_results], ddof=1))
    tabular_probability = sigmoid(
        x_validation @ tabular["weights"] + tabular["bias"]
    )
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
    repeated_metrics = metric_bundle(y_validation[has_history], graph_probability[has_history])
    first_metrics = metric_bundle(y_validation[~has_history], graph_probability[~has_history])

    LOCAL_GRAPH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_GRAPH_MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {"encoder": tabular["encoder"], "parameters": params, "direction": "current_to_previous"},
            model_file,
        )

    report = f"""# One-layer causal GraphSAGE prototype

## Architecture

For each current encounter, a hidden representation combines two separately weighted inputs: its own encoded features and the encoded features of its immediately preceding observed encounter. A ReLU hidden layer with 8 units feeds an unweighted logistic output. Previous-node labels are never inputs.

The model follows only `current -> previous` edges. It is a minimal one-hop message-passing network implemented in NumPy so the core operation remains inspectable.

## Validation comparison

| Model | AUROC | Average precision | Brier |
|---|---:|---:|---:|
| Tabular logistic | {tabular_metrics['auroc']:.3f} | {tabular_metrics['average_precision']:.3f} | {tabular_metrics['brier_score']:.3f} |
| Hand-selected previous-encounter features | {temporal_metrics['auroc']:.3f} | {temporal_metrics['average_precision']:.3f} | {temporal_metrics['brier_score']:.3f} |
| One-layer causal GraphSAGE | {graph_metrics['auroc']:.3f} | {graph_metrics['average_precision']:.3f} | {graph_metrics['brier_score']:.3f} |

Across five pre-specified random seeds, GraphSAGE achieved mean AUROC **{mean_auc:.3f} ± {sd_auc:.3f}** and mean average precision **{mean_ap:.3f} ± {sd_ap:.3f}**. The table reports the fixed primary seed (20260803); no best-seed selection was performed.

## Graph-model subgroup diagnostic

| Observed history | N | AUROC | Average precision | Brier |
|---|---:|---:|---:|---:|
| Has previous encounter | {int(has_history.sum()):,} | {repeated_metrics['auroc']:.3f} | {repeated_metrics['average_precision']:.3f} | {repeated_metrics['brier_score']:.3f} |
| First observed encounter | {int((~has_history).sum()):,} | {first_metrics['auroc']:.3f} | {first_metrics['average_precision']:.3f} | {first_metrics['brier_score']:.3f} |

## Decision rule

The graph prototype must improve validation ranking meaningfully and remain well behaved for both history-available and first-observed encounters before test evaluation. A small or negative gain is a valid result and signals that richer similarity edges, better temporal data, or a simpler non-graph representation may be preferable.

## Limitations

- One-hop aggregation cannot summarize a long trajectory beyond the immediately previous encounter.
- Encounter order is a surrogate derived from identifiers, not verified timestamps.
- The architecture is deliberately small and has not undergone a broad hyperparameter search.
- Validation results guide development; the test set remains uninspected for this model.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
