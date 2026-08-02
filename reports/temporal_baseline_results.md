# Causal one-step temporal baseline

## Question

Does adding information from the immediately preceding observed encounter improve ranking beyond the current-encounter tabular baseline?

The added history features are observed prior-encounter count plus the previous encounter's length of stay, laboratory-procedure count, medication count, diagnosis count, primary diagnosis group, HbA1c-test category, insulin status, and diabetes-medication-change status. Previous readmission labels are deliberately excluded.

## Validation results

| Population / model | N | AUROC | Average precision | Brier |
|---|---:|---:|---:|---:|
| All encounters — tabular | 14,683 | 0.639 | 0.198 | 0.098 |
| All encounters — + causal history | 14,683 | 0.648 | 0.200 | 0.098 |
| Repeated encounters — tabular | 4,238 | 0.587 | 0.249 | 0.141 |
| Repeated encounters — + causal history | 4,238 | 0.591 | 0.249 | 0.141 |

Across all validation encounters, causal history changed AUROC by **+0.91 percentage points** and average precision by **+0.17 percentage points**.

## Interpretation

- This is a graph-derived lag-feature baseline, not a graph neural network.
- A gain would show that the previous node carries incremental information; no gain would warn against assuming that more complex message passing will help.
- The repeated-encounter subgroup is the population that can actually receive a temporal message. Its result is therefore more informative than the overall average.
- Model design uses validation data only. Test performance remains uninspected until the temporal architecture is locked.
- Encounter-ID ordering is a chronology surrogate, not a verified timestamp.
