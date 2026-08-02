"""Audit calibration and demographic subgroup performance for the locked model."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.train_baseline import metric_bundle, prepare_data, select_youden_threshold, sigmoid, transform
from src.train_temporal_baseline import LOCAL_MODEL_PATH, add_causal_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "calibration_subgroups.md"
CALIBRATION_FIGURE = PROJECT_ROOT / "reports" / "figures" / "calibration.png"
SUBGROUP_FIGURE = PROJECT_ROOT / "reports" / "figures" / "subgroup_auroc.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def calibration_slope_intercept(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    design = np.column_stack([np.ones(len(y)), logit])
    coefficients = np.array([0.0, 1.0])
    for _ in range(50):
        fitted = sigmoid(design @ coefficients)
        weight = np.clip(fitted * (1 - fitted), 1e-8, None)
        gradient = design.T @ (y - fitted)
        hessian = design.T @ (weight[:, None] * design)
        update = np.linalg.solve(hessian, gradient)
        coefficients += update
        if np.max(np.abs(update)) < 1e-8:
            break
    return float(coefficients[0]), float(coefficients[1])


def calibration_table(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"outcome": y, "probability": probability})
    frame["bin"] = pd.qcut(frame["probability"], q=10, duplicates="drop")
    return frame.groupby("bin", observed=True).agg(
        n=("outcome", "size"),
        predicted=("probability", "mean"),
        observed=("outcome", "mean"),
    ).reset_index(drop=True)


def subgroup_rows(test: pd.DataFrame, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    working = test.copy()
    working["probability"] = probability
    working["has_history"] = np.where(
        working["observed_prior_encounters"] > 0, "Has history", "First observed"
    )
    rows = []
    for family, column in [
        ("Race", "race"),
        ("Gender", "gender"),
        ("Age", "age"),
        ("History", "has_history"),
    ]:
        for group, subset in working.groupby(column):
            y = subset["target_30d"].to_numpy()
            if len(subset) < 100 or y.sum() < 20 or (len(y) - y.sum()) < 20:
                continue
            result = metric_bundle(y, subset["probability"].to_numpy(), threshold)
            rows.append(
                {
                    "family": family,
                    "group": str(group),
                    "n": len(subset),
                    "positives": int(y.sum()),
                    "prevalence": float(y.mean()),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def draw_calibration(table: pd.DataFrame) -> None:
    width, height = 1000, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 110, 930, 100, 600
    maximum = max(0.25, float(table[["predicted", "observed"]].to_numpy().max()) * 1.15)
    draw.text((left, 28), "Calibration of locked temporal model", fill="#172033", font=font(30, True))
    draw.line((left, bottom, right, bottom), fill="#334155", width=2)
    draw.line((left, bottom, left, top), fill="#334155", width=2)
    for tick in np.linspace(0, maximum, 6):
        x = left + (right - left) * tick / maximum
        y = bottom - (bottom - top) * tick / maximum
        draw.line((x, bottom, x, bottom + 8), fill="#64748b", width=1)
        draw.line((left - 8, y, left, y), fill="#64748b", width=1)
        label = f"{tick:.2f}"
        draw.text((x - 18, bottom + 16), label, fill="#475569", font=font(16))
        draw.text((left - 58, y - 9), label, fill="#475569", font=font(16))
    draw.line((left, bottom, right, top), fill="#94a3b8", width=2)
    points = []
    for row in table.itertuples():
        x = left + (right - left) * row.predicted / maximum
        y = bottom - (bottom - top) * row.observed / maximum
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill="#2563eb", width=4)
    for x, y in points:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#2563eb", outline="white", width=2)
    draw.text((420, 655), "Mean predicted probability", fill="#172033", font=font(20))
    draw.text((18, 330), "Observed rate", fill="#172033", font=font(20))
    draw.text((730, 120), "Perfect calibration", fill="#64748b", font=font(16))
    image.save(CALIBRATION_FIGURE)


def draw_subgroups(subgroups: pd.DataFrame) -> None:
    plotted = subgroups[subgroups["family"].isin(["Race", "Gender", "History"])].copy()
    plotted = plotted.sort_values(["family", "auroc"])
    width, height = 1100, 130 + 58 * len(plotted)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top = 300, 1030, 90
    draw.text((40, 24), "Test AUROC by subgroup", fill="#172033", font=font(30, True))
    for tick in np.arange(0.5, 0.81, 0.05):
        x = left + (right - left) * (tick - 0.5) / 0.3
        draw.line((x, top - 12, x, height - 35), fill="#e2e8f0", width=1)
        draw.text((x - 18, height - 30), f"{tick:.2f}", fill="#475569", font=font(15))
    for index, row in enumerate(plotted.itertuples()):
        y = top + index * 58
        label = f"{row.family}: {row.group} (n={row.n:,})"
        draw.text((35, y - 10), label, fill="#334155", font=font(17))
        x0 = left
        x1 = left + (right - left) * (max(row.auroc, 0.5) - 0.5) / 0.3
        draw.rectangle((x0, y - 11, x1, y + 13), fill="#3b82f6")
        draw.text((x1 + 10, y - 10), f"{row.auroc:.3f}", fill="#172033", font=font(16, True))
    image.save(SUBGROUP_FIGURE)


def main() -> None:
    data = add_causal_history(prepare_data())
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    with LOCAL_MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    validation_probability = sigmoid(
        transform(validation, model["encoder"]) @ model["weights"] + model["bias"]
    )
    test_probability = sigmoid(
        transform(test, model["encoder"]) @ model["weights"] + model["bias"]
    )
    threshold = select_youden_threshold(
        validation["target_30d"].to_numpy(), validation_probability
    )
    y_test = test["target_30d"].to_numpy()
    calibration = calibration_table(y_test, test_probability)
    intercept, slope = calibration_slope_intercept(y_test, test_probability)
    ece = float(
        np.average(
            np.abs(calibration["observed"] - calibration["predicted"]),
            weights=calibration["n"],
        )
    )
    subgroups = subgroup_rows(test, test_probability, threshold)

    CALIBRATION_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    draw_calibration(calibration)
    draw_subgroups(subgroups)

    calibration_rows = "\n".join(
        f"| {index + 1} | {int(row.n):,} | {row.predicted:.3f} | {row.observed:.3f} |"
        for index, row in enumerate(calibration.itertuples())
    )
    subgroup_table = "\n".join(
        f"| {row.family} | {row.group} | {int(row.n):,} | {int(row.positives):,} | "
        f"{row.prevalence:.3f} | {row.auroc:.3f} | {row.average_precision:.3f} | "
        f"{row.recall_sensitivity:.3f} | {row.specificity:.3f} |"
        for row in subgroups.itertuples()
    )
    report = f"""# Calibration and subgroup audit

## Calibration

- Mean predicted risk: **{test_probability.mean():.3f}**.
- Observed 30-day readmission rate: **{y_test.mean():.3f}**.
- Calibration intercept: **{intercept:.3f}** (ideal 0).
- Calibration slope: **{slope:.3f}** (ideal 1).
- Decile expected calibration error: **{ece:.3f}** (lower is better).

| Risk decile | N | Mean predicted | Observed rate |
|---|---:|---:|---:|
{calibration_rows}

![Calibration curve](figures/calibration.png)

## Subgroup performance at the global validation-selected threshold ({threshold:.3f})

Groups with fewer than 100 encounters, fewer than 20 positives, or fewer than 20 negatives are omitted from metric tables. Results are descriptive and do not establish fairness or explain the source of differences.

| Family | Group | N | Positives | Prevalence | AUROC | Avg precision | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{subgroup_table}

![Subgroup AUROC](figures/subgroup_auroc.png)

## Interpretation

1. Calibration assesses probability accuracy, not ranking. A useful risk model needs both.
2. Average precision changes with outcome prevalence, so direct subgroup comparisons require care.
3. A single threshold can produce very different sensitivity and specificity across groups, especially when history availability changes baseline risk.
4. These retrospective subgroup estimates have no external validation and should not be used to claim clinical equity.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

