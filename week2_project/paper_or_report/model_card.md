# Model Card — Diabetes 30-Day Readmission Classifier

## Model Details

| Property | Value |
|---|---|
| **Model type** | Random Forest / XGBoost (ensemble classification) |
| **Task** | Binary classification — early readmission (<30 days) prediction |
| **Dataset** | UCI Diabetes 130-US Hospitals (1999–2008) |
| **Training data size** | ~50,000 patient encounters (70% of unique patients) |
| **Primary metric** | ROC-AUC |
| **Version** | 1.0 (research prototype) |
| **Date** | 2025 |
| **Developed by** | Decodelabs Data Science Internship |
| **Status** | Research / portfolio — NOT for clinical deployment |

---

## Intended Use

**Intended users:** Clinical informatics researchers, hospital quality improvement teams (with significant additional validation)

**Intended purpose:** Screening tool to flag diabetic patients at elevated risk of 30-day readmission at the point of discharge, for prioritisation of post-discharge follow-up resources.

**Out-of-scope uses:**
- NOT intended as a standalone clinical decision tool
- NOT validated for non-diabetic patients
- NOT validated for hospitals outside the original dataset
- NOT validated for data from 2009 onwards
- Should NOT be used to deny care or discharge resources

---

## Factors

**Relevant factors:** Race, age group, diabetes medication type, prior hospitalisation history, length of stay

**Evaluation factors:** Model performance is evaluated across race, age group, and gender in Notebook 12 (Fairness Analysis). Performance variations across groups are reported.

---

## Metrics

| Metric | Value |
|---|---|
| ROC-AUC (test set) | *Run Notebook 09* |
| Average Precision | *Run Notebook 09* |
| Recall (class 1) | *Run Notebook 09* |
| F1-Score (class 1) | *Run Notebook 09* |

Bootstrap 95% confidence intervals are reported in Notebook 09.

---

## Training Data

- UCI ML Repository: 101,766 inpatient encounters
- After cleaning (first-encounter-per-patient): ~71,518 patients
- Training split: 70% of unique patients
- Class balance: ~89% no early readmission, ~11% early readmission
- SMOTE applied to balance training data

---

## Evaluation Data

- Test split: 15% of unique patients (zero overlap with training/validation)
- Patient-aware splitting prevents individual-level leakage

---

## Ethical Considerations

1. **Fairness:** Performance gaps across racial subgroups may exist due to historical healthcare inequities embedded in training data. See Notebook 12.
2. **Transparency:** SHAP explanations are provided for individual predictions (Notebook 11).
3. **Human oversight:** All model predictions should be reviewed by qualified clinical staff before any action is taken.
4. **Data privacy:** The dataset is de-identified. No patient re-identification should be attempted.
5. **Feedback loops:** Deploying this model could affect resource allocation, which in turn affects the distribution of future training data.

---

## Caveats and Recommendations

- Dataset is from 1999–2008. Clinical practice, medications, and patient populations have changed.
- External validation on current, locally representative data is required before any deployment.
- A threshold of 0.5 may not be clinically optimal. Threshold selection should involve clinical stakeholders.
- Model should be one input among many in clinical decision-making, not the sole determinant.
