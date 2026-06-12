# Predicting 30-Day Hospital Readmission in Diabetic Patients
## A Machine Learning Analysis of the UCI Diabetes 130-US Hospitals Dataset

**Author:** Peter Adepoju  
**Programme:** Decodelabs Data Science Internship - Week 2  
**Date:** June 12, 2026

---

## Abstract

Hospital readmission within 30 days of discharge is a major quality indicator and cost driver in the US healthcare system. This study applies supervised machine learning to the UCI Diabetes 130-US Hospitals dataset to predict early readmission among diabetic patients. The cleaned cohort contains 69,987 first encounters with an 11.2% positive class rate. We compare four models - Logistic Regression, Random Forest, Gradient Boosting, and XGBoost - using a patient-aware train/validation/test split to prevent data leakage. Bootstrap 95% confidence intervals are reported for the primary test metrics, and subgroup performance is examined across race, age, and gender. SHAP is used to explain model predictions and fairness analysis is included for deployment risk review.

**Keywords:** hospital readmission, diabetes, machine learning, clinical informatics, algorithmic fairness, SHAP, XGBoost

---

## 1. Background

Diabetes mellitus affects approximately 537 million adults worldwide and is a leading cause of hospitalisation. In the United States, approximately 20% of hospitalised diabetic patients are readmitted within 30 days of discharge, at an estimated cost of $26 billion annually. Since 2012, the Centers for Medicare and Medicaid Services (CMS) have penalised hospitals with excess readmission rates under the Hospital Readmissions Reduction Program (HRRP).

Predictive models that identify high-risk patients at discharge could enable targeted interventions such as enhanced discharge education, earlier follow-up appointments, or dedicated care coordination to reduce preventable readmissions.

---

## 2. Dataset

- **Source:** UCI Machine Learning Repository - Diabetes 130-US Hospitals
- **URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **Period:** 1999-2008, 130 US hospitals
- **Raw dataset:** 101,766 encounters x 50 features
- **After cleaning:** 69,987 first encounters
- **Target:** Binary - early readmission within 30 days (1) vs. not (0)
- **Class balance:** 88.8% negative, 11.2% positive

### Key features
Clinical and administrative features include patient demographics (age, sex, race), admission and discharge metadata, laboratory result counts, procedure counts, medication counts, 24 individual diabetes medication columns, HbA1c and glucose serum results, and three ICD-9 diagnosis codes grouped into disease categories.

---

## 3. Methods

### 3.1 Data Cleaning
- Replaced `?` with `NaN`
- Dropped `weight`, `payer_code`, and `medical_specialty`
- Removed encounters with hospice or expired discharge disposition
- Removed encounters with invalid or unknown gender
- Retained the first encounter per patient to prevent patient-level leakage

### 3.2 Feature Engineering
- ICD-9 codes mapped to disease category groups
- Age bins encoded ordinally
- 24 medication columns ordinal-encoded
- Derived features: `n_medications_active`, `n_medication_changes`, `total_prior_visits`, `had_prior_inpatient`, `primary_diag_is_diabetes`
- HbA1c and glucose serum results encoded with ordinal and binary tested flags

### 3.3 Data Splitting
Patient-aware `GroupShuffleSplit`: 70% train / 15% validation / 15% test by unique patient ID.
Zero patient overlap was verified across all three splits.

### 3.4 Preprocessing
- `log1p` transform for right-skewed utilisation features
- `StandardScaler` fitted on training data only
- SMOTE applied to the training split to address class imbalance

### 3.5 Models
| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | `C=1.0`, `class_weight='balanced'`, `lbfgs`, `max_iter=2000` |
| Random Forest | 300 trees, `max_depth=10`, `min_samples_leaf=20`, `class_weight='balanced_subsample'` |
| Gradient Boosting | 300 estimators, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8` |
| XGBoost | 300 estimators, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight~7.9` |

### 3.6 Evaluation
- Primary metric: ROC-AUC
- Secondary: Average Precision / PR-AUC
- Clinical metric: Recall for class 1
- Bootstrap 95% CI: 1,000 resamples
- McNemar's test for pairwise comparison

---

## 4. Results

### 4.1 Model Performance

**Table 1: Final test-set performance**

| Model | Accuracy | Balanced Acc. | F1 (class 1) | Recall (class 1) | ROC-AUC | Avg. Precision |
|---|---|---|---|---|---|---|
| Dummy Classifier | 0.9284 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.0716 |
| Logistic Regression | 0.6497 | 0.6174 | 0.1916 | 0.5798 | 0.6617 | 0.1403 |
| Random Forest | 0.8852 | 0.5492 | 0.1638 | 0.1569 | 0.6789 | 0.1385 |
| Gradient Boosting | 0.9282 | 0.5005 | 0.0026 | 0.0013 | 0.6855 | 0.1513 |
| XGBoost | 0.9284 | 0.5000 | 0.0000 | 0.0000 | 0.6836 | 0.1508 |

**Table 2: Bootstrap 95% CI for ROC-AUC**

| Model | ROC-AUC | 95% CI |
|---|---|---|
| Logistic Regression | 0.6617 | [0.6418, 0.6813] |
| Random Forest | 0.6789 | [0.6603, 0.6997] |
| Gradient Boosting | 0.6855 | [0.6662, 0.7049] |
| XGBoost | 0.6836 | [0.6649, 0.7024] |

**Figure callouts:**
- Figure 1: `10_dataset_overview_dashboard.png`
- Figure 2: `09_roc_and_pr_curves.png`
- Figure 3: `10_model_performance_comparison.png`
- Figure 4: `11_shap_summary_beeswarm.png`
- Figure 5: `12_fairness_by_race.png`

### 4.2 Feature Importance

SHAP analysis identifies the following as the most influential features:
1. `number_inpatient` - prior inpatient visits
2. `total_prior_visits` - aggregate prior utilisation
3. `had_prior_inpatient` - binary prior inpatient indicator
4. `time_in_hospital` - length of current stay
5. `num_medications` - number of active medications

### 4.3 Fairness Analysis

Subgroup performance across race, age, and gender shows meaningful variation.

**Race**

| Race | N | Prev. % | F1 | Recall | Precision | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| African American | 2,730 | 7.4 | 0.225 | 0.550 | 0.142 | 0.711 |
| Asian | 41 | 17.1 | 0.476 | 0.714 | 0.357 | 0.710 |
| Caucasian | 7,053 | 7.2 | 0.212 | 0.528 | 0.133 | 0.693 |
| Hispanic | 190 | 7.4 | 0.200 | 0.429 | 0.130 | 0.663 |
| Other | 104 | 4.8 | 0.205 | 0.800 | 0.118 | 0.897 |

**Age**

| Age Group | N | Prev. % | F1 | Recall | Precision | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Young (<40) | 908 | 6.5 | 0.198 | 0.542 | 0.121 | 0.705 |
| Middle (40-60) | 3,235 | 6.6 | 0.199 | 0.533 | 0.122 | 0.702 |
| Older (60-80) | 4,889 | 7.4 | 0.234 | 0.556 | 0.148 | 0.707 |
| Elderly (>=80) | 1,467 | 8.0 | 0.196 | 0.458 | 0.124 | 0.648 |

**Gender**

| Gender | N | Prev. % | F1 | Recall | Precision | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Female | 5,764 | 7.4 | 0.213 | 0.513 | 0.135 | 0.688 |
| Male | 4,735 | 6.9 | 0.215 | 0.560 | 0.133 | 0.707 |

---

## 5. Error Analysis

False negative analysis suggests that missed readmissions tend to occur in patients with fewer prior inpatient admissions, which makes them appear lower risk at discharge. A lower classification threshold can improve sensitivity, but it increases false alarms and should be selected based on clinical workflow constraints.

For the Gradient Boosting model, a threshold of 0.30 flags 158 patients with precision 0.304 and recall 0.064.

---

## 6. Limitations

1. **Temporal gap:** Data from 1999-2008; clinical practice has changed substantially.
2. **Single-system readmission:** Only returns to the same hospital system are captured.
3. **Class imbalance:** About 11.2% positive rate limits absolute recall.
4. **No external validation:** Results reflect performance within the same dataset distribution.
5. **Administrative labels:** Early readmission may include planned readmissions.
6. **Missing features:** Social determinants, discharge destination, and adherence history are absent.

---

## 7. Ethics & Bias

Key concerns:
- Racial and age disparities in healthcare access may confound model predictions
- Deploying predictions without human oversight could reduce care quality for flagged patients
- The model should be used as a screening aid, not a decision-maker

---

## 8. Reproducibility

All code is available in the `notebooks/` directory. The random seed is fixed at `RANDOM_SEED = 42` in `configs/config.py`. To reproduce the full analysis:

```bash
pip install -r requirements.txt
python scripts/build_all_notebooks.py
jupyter notebook
```

---

## 9. References

- Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*.
- UCI Machine Learning Repository: Diabetes 130-US Hospitals Dataset. https://archive.ics.uci.edu/dataset/296
- Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Chawla, N. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*.
