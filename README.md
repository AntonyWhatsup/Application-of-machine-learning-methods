# Data Professional Salary Survey Analysis
**Version: v0.0.1**

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
* **`src/`**: Contains reusable Python modules (`.py` scripts). Moving cross-validation procedures, metric calculations, and custom functions out of notebooks and into dedicated modules maintains clean notebooks and demonstrates professional software engineering practices.
* **`saved_models/`**: A directory for storing trained models (e.g., as `.pkl` or `.joblib` files) after executing Grid Search or Random Search for hyperparameter tuning. This avoids retraining computationally expensive models during every run.
* **`visual_app/`**: Placeholder directory for the future Graphical User Interface (GUI) components.
* **`main.py`**: The single root-level entry point that orchestrates the execution of the final program.

## Setup & Execution
*(Instructions for setting up the environment, installing dependencies, and running `main.py` will be added here as the project evolves).*