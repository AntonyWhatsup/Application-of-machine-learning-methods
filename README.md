*Read this in other languages: [🇵🇱 Polski](README_pl.md)*

---

# Machine Learning Methods Application - Final Project

## 📌 Project Overview
This project is developed as part of the **"Application of Machine Learning Methods"** course. The main objective is to build an end-to-end machine learning pipeline using the "Data Professional Salary Survey" dataset.

The project solves two primary business problems:
1. **Regression:** Predicting the annual salary of IT professionals (`SalaryUSD`).
2. **Classification:** Predicting career plans for the current year (`CareerPlansThisYear`).

## 🎯 Requirements & Goals
According to the assignment guidelines, the project must implement a minimum of **8 different machine learning models** (4 for regression, 4 for classification). 

Key components of the project:
* Exploratory Data Analysis (EDA)
* Data Visualization
* Data Preprocessing (Cleaning, missing values handling)
* Feature Engineering
* Model Building (Min. 8 algorithmic models)
* Hyperparameter Tuning (GridSearchCV / Optuna)
* Model Comparison and Validation
* Model Interpretation (using SHAP library)
* Final Business Report (PDF)

## 📊 Dataset
The data originates from the **Brent Ozar Unlimited Data Professional Salary Survey**. 
Due to the nature of survey responses (manual user input), the dataset is highly noisy and requires advanced cleaning techniques (e.g., outlier detection, categorical feature standardization).

## 🛠 Technologies & Tools
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **ML Models:** Scikit-Learn, XGBoost, LightGBM
* **Explainable AI (XAI):** SHAP
* **Environment:** Jupyter Notebook

## 🚀 Roadmap

### Phase 1: Preparation & EDA
- [x] Initialize GitHub repository and environment.
- [ ] Load and profile data.
- [ ] Clean the `SalaryUSD` target variable (remove extreme outliers).
- [ ] Clean the `CareerPlansThisYear` target variable (group classes).
- [ ] Handle missing data and encode categorical variables (One-Hot / Label Encoding).
- [ ] Advanced visualization (Correlation matrix, salary distributions by tech/country).

### Phase 2: Regression (Predicting `SalaryUSD`)
- [ ] Model 1: Linear Regression (Baseline).
- [ ] Model 2: Decision Tree Regressor.
- [ ] Model 3: Random Forest Regressor.
- [ ] Model 4: XGBoost Regressor.
- [ ] Hyperparameter tuning for top models.
- [ ] Model evaluation (RMSE, MAE, R2).
- [ ] Results interpretation (Feature Importance / SHAP).

### Phase 3: Classification (Predicting `CareerPlansThisYear`)
- [ ] Handle class imbalance (e.g., SMOTE, class weights).
- [ ] Model 1: Logistic Regression (Baseline).
- [ ] Model 2: K-Nearest Neighbors (KNN).
- [ ] Model 3: Random Forest Classifier.
- [ ] Model 4: LightGBM / XGBoost Classifier.
- [ ] Hyperparameter tuning.
- [ ] Model evaluation (Precision, Recall, F1-score, Confusion Matrix).
- [ ] Results interpretation (Feature Importance / SHAP).

### Phase 4: Finalization
- [ ] Formulate final conclusions.
- [ ] Generate a professional PDF report (10-20 pages).
- [ ] Submit project files according to course guidelines.

---
*Author: [Your First and Last Name]*
