# History-structure ablation

## Question

Does explicit encounter history contribute detailed clinical information, or mainly indicate that a patient has previously used hospital care?

All specifications use the same training and validation patients. Test data are not evaluated. Previous readmission outcomes are excluded from every specification.

## Validation results

| Specification | AUROC | Average precision | Brier |
|---|---:|---:|---:|
| Current encounter, no prior-utilisation counts | 0.595 | 0.150 | 0.100 |
| + history availability only | 0.631 | 0.170 | 0.099 |
| + observed history count | 0.629 | 0.181 | 0.098 |
| + previous-encounter detail | 0.634 | 0.174 | 0.099 |
| + count and previous-encounter detail | 0.638 | 0.185 | 0.098 |
| Full tabular utilisation + causal history | 0.648 | 0.200 | 0.098 |

## Reading the ablation

- **History availability** tests whether edge existence alone carries signal.
- **Observed history count** tests trajectory length without previous-node clinical content.
- **Previous-encounter detail** tests the latest node's measurements without prior-year utilisation counts.
- **Full utilisation + causal history** tests whether explicit history adds information after the existing aggregate history variables are already known.

These comparisons diagnose the source of predictive information; they do not establish a causal effect of healthcare utilisation or treatment.
