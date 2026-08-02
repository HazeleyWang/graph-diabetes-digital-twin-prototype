# Stable causal graph-residual baseline

## Architecture

The fitted tabular log-odds are frozen as an offset. A directed message from the immediately previous encounter learns only a residual adjustment to the current encounter's log-odds. Previous labels are excluded.

This linear message-passing model is less expressive than GraphSAGE but has a simpler, more stable optimization target. It answers whether the full previous-node representation adds information after the tabular risk score is already known.

## Validation comparison

| Model | AUROC | Average precision | Brier |
|---|---:|---:|---:|
| Tabular logistic | 0.639 | 0.198 | 0.098 |
| Hand-selected previous-encounter features | 0.648 | 0.200 | 0.098 |
| Causal graph residual | 0.647 | 0.201 | 0.098 |

Among the **4,238** encounters with observed history, tabular AUROC was **0.587** and graph-residual AUROC was **0.591**.

## Decision

This model is the stability check for the neural graph prototype. If it matches or exceeds the average neural result, the evidence favours a simpler causal graph adjustment rather than claiming benefit from nonlinear message passing. Test evaluation remains locked until the final temporal specification is selected.
