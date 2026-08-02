# Baseline permutation importance

## Method

The fixed logistic baseline was evaluated on validation data. One raw feature was shuffled at a time, five times, while all other columns were left unchanged. Importance is the decrease in AUROC relative to the unshuffled validation AUROC of **0.639**.

This measures how much the fitted model relies on a feature for ranking validation patients. It does not establish causality, and correlated features can share or mask importance.

## Most influential feature families

| Feature | Mean AUROC decrease (percentage points) | SD |
|---|---:|---:|
| `number_inpatient` | 6.190 | 0.155 |
| `medical_specialty` | 0.608 | 0.110 |
| `diag_1_group` | 0.545 | 0.110 |
| `number_diagnoses` | 0.483 | 0.095 |
| `age` | 0.472 | 0.232 |
| `time_in_hospital` | 0.355 | 0.068 |
| `diabetesMed` | 0.328 | 0.144 |
| `diag_2_group` | 0.307 | 0.090 |
| `diag_3_group` | 0.226 | 0.147 |
| `num_medications` | 0.197 | 0.104 |
| `number_emergency` | 0.187 | 0.049 |
| `admission_source_id` | 0.159 | 0.084 |
| `admission_type_id` | 0.154 | 0.058 |
| `insulin` | 0.129 | 0.101 |
| `A1Cresult` | 0.105 | 0.050 |

Features with an absolute mean AUROC change below 0.05 percentage points: **23 of 42**. Small or negative values should be treated as negligible rather than evidence that a feature is protective.

## How this guides the project

1. The result identifies which tabular signals the graph model must complement rather than merely rediscover.
2. Prior-utilisation variables are especially relevant to a longitudinal patient graph because they summarize earlier encounters.
3. Weak individual medication fields may be grouped into higher-level medication-change representations to reduce sparsity.
4. Importance is calculated on validation only; the test set remains outside feature-selection decisions.
