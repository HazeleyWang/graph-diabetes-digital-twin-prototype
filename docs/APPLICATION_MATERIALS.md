# Application-ready project wording

The wording below is designed for a precision-medicine / graph-based medical digital twin application. Adapt tense and length to the destination, but keep the numerical claims and limitations intact.

## CV project entry

**Graph-based longitudinal diabetes readmission prototype** — Python, NumPy, Pandas, causal temporal graphs, reproducible ML

- Designed a leakage-aware 30-day readmission study using 101,766 hospital encounters from 71,518 patients, with deterministic patient-grouped train/validation/test partitions and training-only preprocessing.
- Constructed 99,343 encounter nodes and 29,353 directed history edges, enforcing current-to-previous message flow to prevent future-encounter leakage.
- Compared logistic, boosted-tree, causal lag-feature, graph-residual, and one-layer GraphSAGE models; rejected an initialization-sensitive neural result and kept post-lock development separate from final test claims.
- Achieved locked test AUROC 0.633 and a paired AUROC gain of 0.0048 over the tabular model (patient-bootstrap 95% CI 0.0004–0.0090); showed that edge existence and trajectory length explained more signal than detailed previous-node content.

## Short CV version

Built a leakage-aware temporal encounter graph for 101,766 diabetes admissions; compared logistic, boosted-tree, causal-history, graph-residual, and GraphSAGE models with patient-level splits, paired bootstrap uncertainty, calibration, and structural ablations. Demonstrated a small, statistically supported ranking gain from explicit history while nonlinear GNN gains were unstable.

## Motivation-letter paragraph

To prepare for research on graph-based medical digital twins, I developed a reproducible prototype using 101,766 diabetes hospital encounters. I formulated 30-day readmission as a patient-grouped longitudinal prediction problem, created strictly backward encounter edges to prevent future-information leakage, and compared logistic, boosted-tree, causal lag-feature, graph-residual, and GraphSAGE models. Rather than selecting the strongest neural run, I evaluated stability across pre-specified seeds and locked the stable temporal specification before test evaluation. Paired patient bootstrap showed a small but consistent AUROC gain from causal history, while structural ablations revealed that history availability and trajectory length carried more signal than detailed previous-node content. This result motivates my next step: learning treatment-conditioned patient-state transitions from heterogeneous longitudinal and molecular data, while distinguishing predictive digital-twin components from causal treatment-effect claims.

## 60-second interview explanation

I built a leakage-aware precursor to a graph-based medical digital twin using repeated diabetes encounters. I split by patient and allowed each encounter to read only earlier encounters. The locked temporal model improved test AUROC from 0.628 to 0.633; paired patient bootstrap placed the gain between 0.0004 and 0.0090, while average-precision and calibration improvements were negligible. A stronger boosted-tree check still showed a small validation gain from causal history. Structural ablations suggested that having an observed trajectory mattered more than detailed content from the previous node, and a GraphSAGE result was unstable across seeds. The project taught me to ask not only whether a graph predicts better, but which aspect of longitudinal structure carries information and what data would be needed to model disease progression and treatment response.

## Claims to avoid

- Do not call the model clinically useful, validated, deployable, or state of the art.
- Do not describe encounter-ID order as verified timestamps.
- Do not claim that GraphSAGE outperformed the stable baseline.
- Do not describe subgroup differences as proof of bias or fairness.
- Do not claim a causal effect of diagnoses, medications, demographics, or utilisation.
