# Predicting 30-Day Hospital Readmission in Diabetic Patients

## Description
This project uses the UCI Diabetes 130-US Hospitals dataset to predict whether a diabetic patient will be readmitted within 30 days of discharge. The analysis covers data cleaning, feature engineering, patient-aware splitting, model training, evaluation, and fairness checks.

The main results show that the strongest overall discrimination came from the tree-based models, with Gradient Boosting achieving the best ROC-AUC at 0.6855 and XGBoost close behind at 0.6836. Logistic Regression captured more positive cases, with recall of 0.5798, but at the cost of weaker overall ranking performance. Threshold analysis showed that lowering the decision threshold increased the number of flagged patients, but recall remained modest because the class is highly imbalanced. Subgroup analysis also showed some variation across race, age, and gender.

For the full methodology, detailed results, and interpretation, see the full report in [`paper_or_report/report.md`](paper_or_report/report.md) and the accompanying part 1-3 PDF reports in the project root.

## Data

The data folder is not included in this repository. You can download it from Google Drive here:

[Google Drive data folder](https://drive.google.com/drive/folders/1jmF1DWeETLHrCoJNLR6MlgVYHncbxsqX?usp=sharing)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks in order, starting from `notebooks/00_project_overview.ipynb`.
