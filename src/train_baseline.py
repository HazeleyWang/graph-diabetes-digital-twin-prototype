"""Train a leakage-safe logistic baseline with only NumPy and pandas."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "logistic_baseline.pkl"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "logistic_baseline_metrics.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_results.md"

NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]

CATEGORICAL_FEATURES = [
    "race", "gender", "age", "admission_type_id", "admission_source_id", "medical_specialty",
    "diag_1_group", "diag_2_group", "diag_3_group", "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed",
]


def diagnosis_group(value: str) -> str:
    if value in {"", "?"}:
        return "Missing"
    if value.startswith(("V", "E")):
        return "Supplementary"
    try:
        code = float(value)
    except ValueError:
        return "Other"
    if 390 <= code < 460 or code == 785:
        return "Circulatory"
    if 460 <= code < 520 or code == 786:
        return "Respiratory"
    if 520 <= code < 580 or code == 787:
        return "Digestive"
    if 250 <= code < 251:
        return "Diabetes"
    if 800 <= code < 1000:
        return "Injury"
    if 710 <= code < 740:
        return "Musculoskeletal"
    if 580 <= code < 630 or code == 788:
        return "Genitourinary"
    if 140 <= code < 240:
        return "Neoplasms"
    return "Other"


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    splits = pd.read_csv(SPLIT_PATH, dtype={"encounter_id": str, "patient_nbr": str})
    df = df.merge(splits, on=["encounter_id", "patient_nbr"], validate="one_to_one")
    df = df[df["eligible_primary"]].copy()
    df["target_30d"] = (df["readmitted"] == "<30").astype(np.float32)
    for column in ["diag_1", "diag_2", "diag_3"]:
        df[f"{column}_group"] = df[column].map(diagnosis_group)
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].replace({"?": "Missing", "Unknown/Invalid": "Missing"})
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def fit_encoder(
    train: pd.DataFrame,
    min_frequency: int = 20,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict:
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    medians = train[numeric_features].median().to_numpy(dtype=np.float32)
    numeric = train[numeric_features].fillna(pd.Series(dict(zip(numeric_features, medians))))
    means = numeric.mean().to_numpy(dtype=np.float32)
    scales = numeric.std(ddof=0).replace(0, 1).to_numpy(dtype=np.float32)
    categories = {}
    for column in categorical_features:
        counts = train[column].value_counts()
        categories[column] = sorted(counts[counts >= min_frequency].index.astype(str).tolist())
    return {
        "medians": medians,
        "means": means,
        "scales": scales,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categories": categories,
    }


def transform(df: pd.DataFrame, encoder: dict) -> np.ndarray:
    numeric_features = encoder["numeric_features"]
    categorical_features = encoder["categorical_features"]
    numeric = df[numeric_features].copy()
    for index, column in enumerate(numeric_features):
        numeric[column] = numeric[column].fillna(float(encoder["medians"][index]))
    numeric_array = numeric.to_numpy(dtype=np.float32)
    numeric_array = (numeric_array - encoder["means"]) / encoder["scales"]

    width = len(numeric_features) + sum(len(values) + 1 for values in encoder["categories"].values())
    matrix = np.zeros((len(df), width), dtype=np.float32)
    matrix[:, : len(numeric_features)] = numeric_array
    row_indices = np.arange(len(df))
    offset = len(numeric_features)
    for column in categorical_features:
        values = encoder["categories"][column]
        lookup = {value: index for index, value in enumerate(values)}
        other_index = len(values)
        indices = df[column].map(lookup).fillna(other_index).to_numpy(dtype=np.int32)
        matrix[row_indices, offset + indices] = 1.0
        offset += len(values) + 1
    return matrix


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic(x: np.ndarray, y: np.ndarray, epochs: int = 35, batch_size: int = 2048) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(20260803)
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = np.float32(0.0)
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.float32(0.0)
    v_b = np.float32(0.0)
    step = 0
    learning_rate, beta1, beta2, epsilon, l2 = 0.01, 0.9, 0.999, 1e-8, 1e-4
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start : start + batch_size]
            xb, yb = x[batch], y[batch]
            residual = sigmoid(xb @ weights + bias) - yb
            grad_w = xb.T @ residual / len(batch) + l2 * weights
            grad_b = residual.mean()
            step += 1
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            weights -= learning_rate * (m_w / (1 - beta1**step)) / (np.sqrt(v_w / (1 - beta2**step)) + epsilon)
            bias -= learning_rate * (m_b / (1 - beta1**step)) / (np.sqrt(v_b / (1 - beta2**step)) + epsilon)
    return weights, float(bias)


def metric_bundle(y: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_int = y.astype(np.int8)
    prediction = (probability >= threshold).astype(np.int8)
    positive = int(y_int.sum())
    negative = len(y_int) - positive
    ranks = pd.Series(probability).rank(method="average").to_numpy()
    auroc = (ranks[y_int == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
    order = np.argsort(-probability, kind="stable")
    sorted_y = y_int[order]
    precision_curve = np.cumsum(sorted_y) / np.arange(1, len(y_int) + 1)
    average_precision = float(precision_curve[sorted_y == 1].mean())
    tp = int(((prediction == 1) & (y_int == 1)).sum())
    fp = int(((prediction == 1) & (y_int == 0)).sum())
    tn = int(((prediction == 0) & (y_int == 0)).sum())
    fn = int(((prediction == 0) & (y_int == 1)).sum())
    return {
        "auroc": float(auroc),
        "average_precision": average_precision,
        "brier_score": float(np.mean((probability - y) ** 2)),
        "accuracy": float((prediction == y_int).mean()),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall_sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "threshold": threshold,
        "n": len(y_int),
        "positives": positive,
        "predicted_positive_rate": float(prediction.mean()),
    }


def select_youden_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """Choose a demonstrative operating threshold using validation data only."""
    candidates = np.linspace(0.01, 0.50, 300)
    best_threshold, best_score = 0.5, -np.inf
    for threshold in candidates:
        bundle = metric_bundle(y, probability, float(threshold))
        score = bundle["recall_sensitivity"] + bundle["specificity"] - 1
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold


def main() -> None:
    df = prepare_data()
    train = df[df["split"] == "train"].reset_index(drop=True)
    validation = df[df["split"] == "validation"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    encoder = fit_encoder(train)
    x_train = transform(train, encoder)
    weights, bias = fit_logistic(x_train, train["target_30d"].to_numpy())
    del x_train

    validation_probability = sigmoid(transform(validation, encoder) @ weights + bias)
    test_probability = sigmoid(transform(test, encoder) @ weights + bias)
    selected_threshold = select_youden_threshold(
        validation["target_30d"].to_numpy(), validation_probability
    )
    results = {
        "selected_threshold": selected_threshold,
        "validation_at_0_5": metric_bundle(
            validation["target_30d"].to_numpy(), validation_probability, 0.5
        ),
        "test_at_0_5": metric_bundle(test["target_30d"].to_numpy(), test_probability, 0.5),
        "validation_selected": metric_bundle(
            validation["target_30d"].to_numpy(), validation_probability, selected_threshold
        ),
        "test_selected": metric_bundle(
            test["target_30d"].to_numpy(), test_probability, selected_threshold
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump({"encoder": encoder, "weights": weights, "bias": bias}, model_file)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for split, key in [("validation", "validation_selected"), ("test", "test_selected")]:
        result = results[key]
        rows.append(
            f"| {split} | {result['n']:,} | {result['positives']:,} | {result['auroc']:.3f} | "
            f"{result['average_precision']:.3f} | {result['brier_score']:.3f} | "
            f"{result['precision']:.3f} | {result['recall_sensitivity']:.3f} | {result['specificity']:.3f} |"
        )
    report = f"""# Logistic-regression baseline

## Purpose

This is the transparent tabular reference model that future graph models must beat on the same patient-grouped splits. Encoding, rare-category handling, standardization, and model fitting use training data only.

## Model

- Prediction time: discharge; primary cohort excludes death/hospice dispositions.
- Numeric features: {len(NUMERIC_FEATURES)}; categorical feature families: {len(CATEGORICAL_FEATURES)}.
- Diagnosis codes are grouped into broad clinical categories.
- Categories occurring fewer than 20 times in training are grouped as other.
- Optimizer: deterministic mini-batch Adam with L2 regularization, implemented in NumPy.
- No class weighting, preserving a meaningful probability baseline.
- A demonstrative operating threshold is selected on validation data by maximizing Youden's J (sensitivity + specificity - 1): **{selected_threshold:.3f}**.

## Results

| Split | N | Positives | AUROC | Average precision | Brier | Precision | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{"\n".join(rows)}

A random ranking has expected average precision close to prevalence (about 0.11). Accuracy is not a headline metric because an all-negative classifier would already appear highly accurate.

At the untouched 0.50 threshold, test sensitivity was {results['test_at_0_5']['recall_sensitivity']:.3f}. Applying the validation-selected threshold increased test sensitivity to {results['test_selected']['recall_sensitivity']:.3f}, with specificity {results['test_selected']['specificity']:.3f}. This threshold is an analytical demonstration, not a clinically approved operating point.

## Interpretation guardrails

- AUROC measures ranking across thresholds; it does not guarantee calibration.
- Average precision is especially informative for the 11% positive class.
- Low sensitivity at threshold 0.50 is expected for an unweighted rare-outcome model; the selected threshold uses validation data only.
- Future choices must use validation data rather than repeated inspection of test performance.
- This is an association model, not evidence that any variable causes readmission.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
