# Locked causal temporal model: test evaluation

## Pre-test lock

The hand-selected one-step temporal logistic model was selected before this evaluation because it had the strongest stable validation AUROC (0.648), while neural GraphSAGE was initialization-sensitive. No test result was used to choose this specification.

The operating threshold **0.100** was selected on validation data by Youden's J and then applied unchanged to test. Confidence intervals use 500 patient-level bootstrap samples, keeping repeated encounters clustered by patient.

## Test results

| Metric | Estimate | Patient-bootstrap 95% CI |
|---|---:|---:|
| AUROC | 0.633 | 0.615 to 0.649 |
| Average precision | 0.206 | 0.175 to 0.236 |
| Brier score | 0.099 | 0.095 to 0.103 |
| Precision | 0.153 | Not estimated |
| Sensitivity | 0.640 | Not estimated |
| Specificity | 0.536 | Not estimated |

For context, the locked tabular test AUROC was **0.628** and average precision was **0.206**. The temporal model changed test AUROC by **+0.48 percentage points** and average precision by **+0.01 percentage points**.

## Paired incremental value

Each bootstrap draw resamples patients once and evaluates both locked models on exactly the same encounters. This directly quantifies uncertainty in the model difference.

| Candidate minus tabular | Point change | Patient-bootstrap 95% CI | Probability candidate is better |
|---|---:|---:|---:|
| AUROC | +0.0048 | +0.0004 to +0.0090 | 99.0% |
| Average precision | +0.0001 | -0.0032 to +0.0040 | 57.2% |
| Brier score | +0.0001 | -0.0001 to +0.0003 | 21.4% |

## History-availability diagnostic

| Population | N | AUROC | Average precision | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Has previous observed encounter | 4,487 | 0.612 | 0.276 | 0.954 | 0.100 |
| First observed encounter | 10,452 | 0.599 | 0.138 | 0.404 | 0.709 |

## Interpretation boundary

This is a research prototype on retrospective, de-identified data from 1999-2008. It is not clinically validated. Encounter ordering is a surrogate, calibration requires deeper assessment, and external validation is absent. The honest contribution is a leakage-aware comparison showing whether limited observed history adds incremental predictive value.
