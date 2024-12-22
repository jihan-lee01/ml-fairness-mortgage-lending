# Fairness Analysis in Mortgage Lending

This repository contains the source code, data, and documentation for the project **Fairness Analysis in Mortgage Lending**, which investigates biases in U.S. mortgage lending using the 2022 Home Mortgage Disclosure Act (HMDA) dataset. The project employs machine learning techniques and fairness evaluation tools to analyze and address demographic disparities affecting mortgage approvals.

---

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Fairness Evaluation](#fairness-evaluation)
- [How to Run](#how-to-run)
- [Contributions](#contributions)
- [References](#references)

---

## Introduction

Mortgage lending plays a critical role in financial stability and wealth creation. However, biases against certain demographic groups such as racial minorities, women, and older individuals persist despite regulatory frameworks. This project aims to:
- Detect and quantify these biases.
- Build machine learning models for mortgage approval predictions.
- Suggest fairness techniques to mitigate biases and promote equity.

The project leverages machine learning models like Random Forest and XGBoost, alongside IBM's AI Fairness 360 toolkit, to assess and address potential disparities.

---

## Project Structure

The directory structure is organized as follows:
analysis/
|------ models.py
|------ preprocessing.py
|------ random_forest.py
|------ visualization.py
notebooks/
|------ models.ipynb
|------ preprocessing_demo.ipynb
|------ visualization.ipynb
data/
|------ raw/
|       |------ X_test.csv
|       |------ X_train.csv
|       |------ y_test.csv
|       |------ y_train.csv
|------ processed/
|       |------ preprocessed_X_test.csv
|       |------ preprocessed_X_train.csv
|       |------ preprocessed_y_test.csv
|       |------ preprocessed_y_train.csv
|       |------ sampled_preprocessed_data.csv
graphs/
|------ age_distribution.png
|------ approval_rates.png
|------ ethnicity_distribution.png
|------ correlation_heatmap.png
|------ race_distribution.png
|------ sex_distribution.png
references/
|------ Cherian_2014.pdf
|------ Gill_et_al_2020.pdf
|------ Hodges_et_al_2024.pdf
|------ Lee_2017.pdf
|------ Zou_et_al_2022.pdf
docs/
|------ project_proposal.pdf
|------ project_proposal_graded.pdf
|------ report.pdf
|------ project_spotlight.pptx
.gitignore
README.md

---

## Data

- **Source**: 2022 HMDA Public Loan/Application (LAR) Dataset.
- **Attributes**: Includes demographic variables (e.g., race, ethnicity, sex, age) and loan-specific details.
- **Protected Attributes**:
  - **Ethnicity**: Hispanic or Latino (Unprivileged) vs. Not Hispanic or Latino (Privileged).
  - **Race**: Minority races (Unprivileged) vs. White (Privileged).
  - **Sex**: Female (Unprivileged) vs. Male (Privileged).
  - **Age**: <25 or >74 (Unprivileged) vs. 25–74 (Privileged).
- **Preprocessing**: Includes cleaning, feature selection, imputation, and one-hot encoding.

---

## Methods

1. **Machine Learning Models**:
   - Logistic Regression (Baseline)
   - Random Forest
   - XGBoost
2. **Fairness Assessment**:
   - IBM's AI Fairness 360 toolkit.
   - Metrics: Disparate Impact, Statistical Parity Difference, Average Odds Difference, Equal Opportunity Difference.

---

## Results

- **Performance**:
  - XGBoost achieved the highest accuracy (83.28%) with robust feature importance analysis.
  - Random Forest performed similarly with slightly lower accuracy.
- **Fairness Evaluation**:
  - Significant biases against racial minority groups were observed in both the dataset and model predictions.
  - Minimal biases were found in terms of ethnicity, sex, and age.

---

## Fairness Evaluation

Metrics used to evaluate fairness:
- **Dataset Fairness**:
  - Significant bias found for race with Disparate Impact (0.84) and Statistical Parity Difference (-0.12).
- **Model Fairness (XGBoost)**:
  - Race-related bias persisted in model predictions (Disparate Impact: 0.77).

Mitigation strategies proposed:
1. **Preprocessing**: Reweighing and optimized preprocessing.
2. **Inprocessing**: Adversarial debiasing.
3. **Postprocessing**: Equalized odds postprocessing.

---

## References

- **Dataset**: [HMDA Data](https://ffiec.cfpb.gov/data-publication/one-year-national-loan-level-dataset/2022)
- **Fairness Toolkit**: [AI Fairness 360](https://aif360.readthedocs.io/en/stable/)
- **Key references**: 
  - Hodges et al. (2024)
  - Cherian (2014)
  - Gill et al. (2020)
  - Lee (2017)
  - Zou et al. (2022)

___

## Contact
 
If you need any revisions or additional details, feel free to contact me [here](mailto:jihan.lee@alumni.emory.edu)
