# Limitations & Known Issues

## Dataset Limitations

1. **Temporal validity:** Data from 1999–2008. Drug availability, clinical guidelines, and standard of care have changed significantly.

2. **Single-system readmission only:** If a patient is discharged from Hospital A and readmitted to Hospital B, this is NOT captured as a readmission. This likely underestimates the true readmission rate, particularly for patients with lower geographic stability.

3. **Administrative label noise:** The "readmitted within 30 days" label includes planned readmissions (e.g. scheduled surgery, chemotherapy). A clinical definition would exclude planned returns.

4. **Missing clinical context:** Key predictors of readmission are absent: discharge destination (home vs skilled nursing facility), social support, patient literacy, socioeconomic status, insurance type (partially captured by payer_code, which was dropped due to high missingness).

5. **ICD-9 coding variability:** Diagnosis coding practices vary across hospitals and coders. The same clinical condition may be coded differently in different hospitals.

## Model Limitations

6. **No external validation:** All performance metrics are in-sample — the model has not been evaluated on data from different hospitals, years, or geographic regions.

7. **Class imbalance ceiling:** With only ~11% positive rate, even a recall of 60% means 40% of high-risk patients are missed. The model is a screening tool, not a definitive classifier.

8. **Feature drift:** If the distribution of input features shifts (e.g. newer medications, changed diagnostic coding), model performance will degrade without retraining.

9. **Threshold sensitivity:** Performance metrics reported at the 0.5 default threshold. Clinical deployment would require careful threshold selection by domain experts.

## Ethical Limitations

10. **Embedded historical bias:** If certain patient groups historically received less thorough documentation or fewer diagnostic tests, the model may learn to assign lower risk to those groups — not because they are lower risk, but because their risk factors are less measured.

11. **Feedback loop risk:** If hospitals use this model to allocate follow-up resources, and if the model underserves certain groups, those groups may experience worse outcomes, which in turn validates the model's lower risk predictions — a harmful feedback loop.

12. **Not a replacement for clinical judgment:** No ML model should replace clinical assessment. This model is designed to support, not supplant, clinical decision-making.
