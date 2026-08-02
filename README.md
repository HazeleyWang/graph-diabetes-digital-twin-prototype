# Graph Diabetes Digital Twin Prototype

A compact, reproducible research prototype for representing longitudinal diabetes hospital encounters as a patient–encounter graph and evaluating graph-based prediction of 30-day readmission.

## Research question

Can a graph representation that connects repeated encounters from the same patient improve readmission-risk modelling over a strong tabular baseline, while yielding clinically interpretable patient-neighbourhood signals?

## Dataset

The project uses **Diabetes 130-US Hospitals for Years 1999–2008** from the UCI Machine Learning Repository (101,766 encounters, 47 features). The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and should be cited as:

> Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999–2008. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

The raw files are deliberately excluded from Git. See [`data/README.md`](data/README.md) for download and placement instructions. Although the public dataset is de-identified, it contains sensitive demographic and clinical attributes; do not attempt re-identification and avoid publishing row-level extracts.

## Planned workflow

1. Audit schema, missing-value encodings, duplicates, patient overlap, class balance, and temporal consistency.
2. Build leakage-aware patient-level train/validation/test splits.
3. Establish logistic-regression and gradient-boosting baselines.
4. Construct a patient–encounter graph and compare a simple graph model.
5. Report discrimination, calibration, subgroup performance, and limitations.

The current first-pass findings are in [`reports/data_audit.md`](reports/data_audit.md). Regenerate them with:

```powershell
python -m src.audit_data
```

The modelling contract for all 50 variables is in [`reports/data_dictionary.md`](reports/data_dictionary.md). A Chinese, step-by-step learning companion is maintained in [`docs/PROJECT_GUIDE_ZH.md`](docs/PROJECT_GUIDE_ZH.md).

The leakage-safe patient split is summarized in [`reports/split_audit.md`](reports/split_audit.md); its record-level assignment file remains local and Git-ignored.

## Setup

```powershell
conda env create -f environment.yml
conda activate graph-diabetes-twin
```

## Structure

```text
data/       Local data and download instructions
notebooks/  Numbered exploratory and modelling notebooks
src/        Reusable project code
tests/      Automated checks
reports/    Figures and short findings
```

## Reproducibility and ethics

- Raw and derived row-level data remain local.
- Splits must be grouped by `patient_nbr` to prevent the same patient appearing across evaluation partitions.
- Sensitive attributes such as race, gender, and age require subgroup checks and careful interpretation.
- This prototype is research-only and is not a clinical decision-support system.
