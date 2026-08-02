# Patient-grouped split audit

## Definition

- Split ratio: approximately 70% train, 15% validation, 15% test.
- Assignment unit: patient, using a salted deterministic SHA-256 bucket.
- Primary cohort excludes discharge dispositions representing death or hospice: ['11', '13', '14', '19', '20', '21'].
- The row-level assignment file is stored locally at `data/processed/split_assignments.csv` and is ignored by Git.

## Integrity checks

- Patients present in more than one split: **0**.
- Patient allocation: train=50,152 (70.13%), validation=10,679 (14.93%), test=10,687 (14.94%).
- Encounters excluded from the primary cohort: **2,423**.
- Maximum difference in 30-day-positive rate between splits: **0.22 percentage points**.

## Primary-cohort distribution

| Split | Encounters | Patients | 30-day positives | Positive rate |
|---|---:|---:|---:|---:|
| train | 69,721 | 49,093 | 7,915 | 11.35% |
| validation | 14,683 | 10,445 | 1,670 | 11.37% |
| test | 14,939 | 10,452 | 1,729 | 11.57% |

## Interpretation

The test set is now a genuine unseen-patient evaluation. Validation is used for model and threshold choices; test remains untouched until the analysis pipeline is locked. Similar target rates across splits indicate that deterministic grouping did not introduce a material class-balance shift.
