# Data Professional Salary Survey Analysis
**Version: v0.1.0**

## Project Overview
This repository contains a comprehensive data analysis and machine learning pipeline based on the "Data_Professional_Salary_Survey_Responses" dataset. The project covers the full lifecycle of data modeling, including exploratory data analysis (EDA), data preparation, feature engineering, and the training and evaluation of various machine learning models. 

The primary goals are to predict the continuous variable `SalaryUSD` using regression models and the categorical variable `CareerPlans This Year` using classification models.

## Directory Structure & Justification

The project is structured to maintain clean code separation, scalability, and reproducibility:

* **`data/`**: Central location for all datasets.
  * **`raw/`**: Contains the original, unmodified dataset exactly as provided. This data is intentionally kept "dirty" to preserve the initial state before cleaning.
  * **`processed/`**: Contains cleaned data and datasets resulting from the feature engineering and data preparation phases.
* **`notebooks/`**: Contains Jupyter Notebooks used for analytical, step-by-step development as required by the project specifications. The work is logically separated:
  * `01_eda.ipynb`: Covers exploratory data analysis, variable distributions, and correlation matrices.
  * `02_data_prep_and_feat_eng.ipynb`: Handles missing values, scaling, and the creation of at least three new engineered features.
  * `03_modeling_regression.ipynb`: Dedicated to building and evaluating at least three regression models for `SalaryUSD`.
  * `04_modeling_classification.ipynb`: Dedicated to building and evaluating at least three classification models for `CareerPlans This Year`.
  * `05_modeling_clustering.ipynb`: Contains the implementation of at least two clustering algorithms.
* **`src/salary_survey/`**: Reusable loading, domain-cleaning and leakage-safe preprocessing components.
* **`models/`**: Serialized fitted pipelines produced by the modeling notebooks.
* **`reports/figures/`**: EDA and model visualizations. `reports/project_report.pdf` collects the tables and figures.
* **`scripts/`**: Reproducible notebook and PDF-report builders.
* **`tests/`**: Unit tests for domain cleaning and the multi-label encoder.

## Setup & Execution

The raw workbook is intentionally not stored in Git. Place it at:

`data/raw/Data_Professional_Salary_Survey_Responses.xlsx`

Then run from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python scripts/build_notebooks.py
python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_data_prep_and_feat_eng.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/03_modeling_regression.ipynb --ExecutePreprocessor.timeout=1800
python -m nbconvert --to notebook --execute --inplace notebooks/04_modeling_classification.ipynb --ExecutePreprocessor.timeout=1800
python -m nbconvert --to notebook --execute --inplace notebooks/05_modeling_clustering.ipynb --ExecutePreprocessor.timeout=1800
python scripts/build_report.py
```

## Methodological notes

- Statistical preprocessing is fitted after train/test split and remains inside `Pipeline`/`ColumnTransformer`.
- `OtherDatabases` is treated as a comma-separated multi-label field.
- `Survey Year` is retained and regression includes a temporal holdout evaluation.
- `CareerPlansThisYear = Not Asked` is excluded from the main classification because the question was not asked in 2017.
- Salaries are nominal USD and are not adjusted for inflation or purchasing power.
