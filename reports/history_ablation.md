# Prior-utilisation ablation

## Question

How much predictive information is lost when the model cannot use outpatient, emergency, and inpatient visit counts from the preceding year?

## Validation results

| Model | AUROC | Average precision | Brier score |
|---|---:|---:|---:|
| Full tabular baseline | 0.639 | 0.198 | 0.098 |
| Without prior-utilisation counts | 0.595 | 0.150 | 0.100 |

Removing the three prior-utilisation counts reduced AUROC by **4.38 percentage points** and average precision by **4.82 percentage points**.

## Interpretation

This controlled ablation supports the hypothesis that longitudinal healthcare history contains meaningful information beyond the current admission. It motivates a patient–encounter graph that represents earlier encounters explicitly rather than relying only on three aggregate counts. It does not yet prove that a graph neural network will outperform the baseline; that remains the next empirical question.

Only validation data is used for this model comparison. The test set is not used to decide whether longitudinal graph features should be developed.
