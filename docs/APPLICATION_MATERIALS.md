# Application-ready project wording

The wording below is designed for a precision-medicine / graph-based medical digital twin application. Adapt tense and length to the destination, but keep the numerical claims and limitations intact.

## CV project entry

**Graph-based longitudinal diabetes readmission prototype** — Python, NumPy, Pandas, causal temporal graphs, reproducible ML

- Designed a leakage-aware 30-day readmission study using 101,766 hospital encounters from 71,518 patients, with deterministic patient-grouped train/validation/test partitions and training-only preprocessing.
- Constructed 99,343 encounter nodes and 29,353 directed history edges, enforcing current-to-previous message flow to prevent future-encounter leakage.
- Compared tabular logistic, causal lag-feature, graph-residual, and one-layer GraphSAGE models; rejected an initialization-sensitive neural result and locked the strongest stable specification before test evaluation.
- Achieved test AUROC 0.633 (patient-bootstrap 95% CI 0.615–0.649) and average precision 0.206; audited calibration, history-availability effects, and demographic subgroup performance.

## Short CV version

Built a leakage-aware temporal encounter graph for 101,766 diabetes admissions; compared tabular, causal-history, graph-residual, and GraphSAGE models with patient-level splits, bootstrap uncertainty, calibration, and subgroup audits. Demonstrated that prior utilisation was strongly informative while nonlinear graph gains were small and unstable.

## Motivation-letter paragraph

To prepare for research on graph-based medical digital twins, I developed a reproducible prototype using 101,766 diabetes hospital encounters. I formulated 30-day readmission as a patient-grouped longitudinal prediction problem, created strictly backward encounter edges to prevent future-information leakage, and compared conventional logistic regression with causal lag features, a graph residual, and a one-layer GraphSAGE model. Rather than selecting the strongest single neural run, I evaluated stability across pre-specified random seeds and locked the most stable temporal specification before test evaluation. The project showed that prior healthcare utilisation was highly informative, while detailed graph history added only modest independent value. This experience strengthened my interest in patient-level multimodal representations while also teaching me to distinguish a biologically appealing modelling idea from evidence of genuine out-of-sample improvement.

## 60-second interview explanation

I wanted to test a small but honest version of a graph-based medical digital twin. The data contain repeated diabetes hospital encounters, so I first split by patient to prevent the same person appearing in training and test. I defined prediction at discharge and built directed edges from each current encounter to the previous observed encounter; reversing those edges would leak the future. A tabular model reached test AUROC 0.628. Adding one-step causal history increased it only to 0.633. A GraphSAGE run looked better once, but five random seeds showed that it was unstable, so I did not select it. The main lesson was that healthcare history is strongly predictive, but a graph neural network is not automatically better than a careful tabular representation. I think that combination of graph design, leakage control, uncertainty analysis, and honest negative evidence is the most valuable part of the project.

## Claims to avoid

- Do not call the model clinically useful, validated, deployable, or state of the art.
- Do not describe encounter-ID order as verified timestamps.
- Do not claim that GraphSAGE outperformed the stable baseline.
- Do not describe subgroup differences as proof of bias or fairness.
- Do not claim a causal effect of diagnoses, medications, demographics, or utilisation.

