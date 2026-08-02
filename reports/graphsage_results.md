# One-layer causal GraphSAGE prototype

## Architecture

For each current encounter, a hidden representation combines two separately weighted inputs: its own encoded features and the encoded features of its immediately preceding observed encounter. A ReLU hidden layer with 8 units feeds an unweighted logistic output. Previous-node labels are never inputs.

The model follows only `current -> previous` edges. It is a minimal one-hop message-passing network implemented in NumPy so the core operation remains inspectable.

## Validation comparison

| Model | AUROC | Average precision | Brier |
|---|---:|---:|---:|
| Tabular logistic | 0.639 | 0.198 | 0.098 |
| Hand-selected previous-encounter features | 0.648 | 0.200 | 0.098 |
| One-layer causal GraphSAGE | 0.655 | 0.212 | 0.097 |

Across five pre-specified random seeds, GraphSAGE achieved mean AUROC **0.636 ± 0.040** and mean average precision **0.202 ± 0.013**. The table reports the fixed primary seed (20260803); no best-seed selection was performed.

## Graph-model subgroup diagnostic

| Observed history | N | AUROC | Average precision | Brier |
|---|---:|---:|---:|---:|
| Has previous encounter | 4,238 | 0.623 | 0.273 | 0.139 |
| First observed encounter | 10,445 | 0.616 | 0.147 | 0.080 |

## Decision rule

The graph prototype must improve validation ranking meaningfully and remain well behaved for both history-available and first-observed encounters before test evaluation. A small or negative gain is a valid result and signals that richer similarity edges, better temporal data, or a simpler non-graph representation may be preferable.

## Limitations

- One-hop aggregation cannot summarize a long trajectory beyond the immediately previous encounter.
- Encounter order is a surrogate derived from identifiers, not verified timestamps.
- The architecture is deliberately small and has not undergone a broad hyperparameter search.
- Validation results guide development; the test set remains uninspected for this model.
