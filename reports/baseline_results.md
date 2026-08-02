# Logistic-regression baseline

## Purpose

This is the transparent tabular reference model that future graph models must beat on the same patient-grouped splits. Encoding, rare-category handling, standardization, and model fitting use training data only.

## Model

- Prediction time: discharge; primary cohort excludes death/hospice dispositions.
- Numeric features: 8; categorical feature families: 34.
- Diagnosis codes are grouped into broad clinical categories.
- Categories occurring fewer than 20 times in training are grouped as other.
- Optimizer: deterministic mini-batch Adam with L2 regularization, implemented in NumPy.
- No class weighting, preserving a meaningful probability baseline.
- A demonstrative operating threshold is selected on validation data by maximizing Youden's J (sensitivity + specificity - 1): **0.108**.

## Results

| Split | N | Positives | AUROC | Average precision | Brier | Precision | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 14,683 | 1,670 | 0.639 | 0.198 | 0.098 | 0.160 | 0.610 | 0.589 |
| test | 14,939 | 1,729 | 0.628 | 0.206 | 0.099 | 0.152 | 0.582 | 0.575 |

A random ranking has expected average precision close to prevalence (about 0.11). Accuracy is not a headline metric because an all-negative classifier would already appear highly accurate.

At the untouched 0.50 threshold, test sensitivity was 0.031. Applying the validation-selected threshold increased test sensitivity to 0.582, with specificity 0.575. This threshold is an analytical demonstration, not a clinically approved operating point.

## Interpretation guardrails

- AUROC measures ranking across thresholds; it does not guarantee calibration.
- Average precision is especially informative for the 11% positive class.
- Low sensitivity at threshold 0.50 is expected for an unweighted rare-outcome model; the selected threshold uses validation data only.
- Future choices must use validation data rather than repeated inspection of test performance.
- This is an association model, not evidence that any variable causes readmission.
