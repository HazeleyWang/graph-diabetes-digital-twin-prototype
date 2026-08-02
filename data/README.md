# Data access and handling

## Source

- Dataset: [Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)
- DOI: https://doi.org/10.24432/C5230J
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Expected files: `diabetic_data.csv` and `IDS_mapping.csv`

Download the archive from UCI and extract the two files into `data/raw/`:

```text
data/raw/diabetic_data.csv
data/raw/IDS_mapping.csv
```

These files are ignored by Git. Do not force-add them. Keep row-level extracts, intermediate tables, trained artifacts containing record-level information, and generated predictions out of the repository.

## Responsible-use notes

The dataset is publicly available and de-identified, but includes sensitive demographic and clinical variables. Use it only for research/education, preserve attribution, do not attempt re-identification, and publish aggregate results only. The source license permits sharing and adaptation with attribution; excluding raw data here also keeps the repository lean and makes provenance explicit.

