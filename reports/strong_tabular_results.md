# Strong nonlinear tabular baseline

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
| All encounters - nonlinear current-encounter baseline | 14,683 | 0.645 | 0.201 | 0.097 |
| All encounters - nonlinear + causal history | 14,683 | 0.651 | 0.205 | 0.097 |
| Encounters with history - nonlinear current baseline | 4,238 | 0.599 | 0.250 | 0.140 |
| Encounters with history - nonlinear + causal history | 4,238 | 0.606 | 0.256 | 0.139 |

Adding one-step history changed validation AUROC by **+0.61 percentage points** and average precision by **+0.36 percentage points** relative to the nonlinear baseline.

## Interpretation boundary

This is a stronger tabular comparator, not a newly selected final model. Its role is to test whether graph-derived history adds information beyond nonlinear interactions. Test evaluation would require a new pre-specified lock and is intentionally deferred.
