# First-pass data audit

Generated from the local raw data by `python -m src.audit_data`.

## Dataset integrity

| Check | Result |
|---|---:|
| Encounter rows | 101,766 |
| Columns | 50 |
| Unique encounter IDs | 101,766 |
| Duplicate encounter IDs | 0 |
| Exact duplicate rows | 0 |
| Unique patients | 71,518 |
| Patients with multiple encounters | 16,773 (23.45%) |
| Maximum encounters for one patient | 40 |

## Readmission target

| Original label | Encounters | Share |
|---|---:|---:|
| `NO` | 54,864 | 53.91% |
| `>30` | 35,545 | 34.93% |
| `<30` | 11,357 | 11.16% |

For a binary 30-day-readmission endpoint, `<30` is the positive class: **11,357 encounters (11.16%)**. The class imbalance means accuracy alone is unsuitable; report AUROC, average precision, calibration, sensitivity/specificity, and confidence intervals.

## Explicit missing-value tokens

The source uses strings such as `?` and `Unknown/Invalid` rather than ordinary null values. Columns without one of the configured tokens are omitted below.

| Column | Missing/unknown | Share |
|---|---:|---:|
| `weight` | 98,569 | 96.86% |
| `max_glu_serum` | 96,420 | 94.75% |
| `A1Cresult` | 84,748 | 83.28% |
| `medical_specialty` | 49,949 | 49.08% |
| `payer_code` | 40,256 | 39.56% |
| `race` | 2,273 | 2.23% |
| `diag_3` | 1,423 | 1.40% |
| `diag_2` | 358 | 0.35% |
| `diag_1` | 21 | 0.02% |
| `gender` | 3 | 0.00% |

## Leakage and modelling decisions

1. **Split by `patient_nbr`, never by encounter row.** Repeated encounters from one patient must not cross train, validation, and test partitions.
2. **Do not use identifiers as predictive features.** Keep `encounter_id` only for traceability and `patient_nbr` only for grouping/graph construction.
3. **Define the prediction time explicitly.** Audit discharge-related variables before modelling; any field only known after the prediction point must be excluded.
4. **Fit preprocessing on training data only.** Imputation, category grouping, scaling, feature selection, and graph-neighbour construction must not inspect validation/test outcomes.
5. **Treat demographics as sensitive.** Evaluate race, gender, and age subgroups, but avoid causal or clinical claims from observational associations.
6. **Avoid a transductive evaluation shortcut.** If graph edges use patient similarity, test nodes must not leak labels or outcome-derived features into training.

## Recommended next step

Create a data dictionary that classifies every variable as identifier, demographic, admission-known, during-stay, discharge-known, outcome, or mapping-only. Then lock the prediction time and feature set before building a patient-grouped baseline split.
