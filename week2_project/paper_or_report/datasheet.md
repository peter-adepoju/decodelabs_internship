# Datasheet for the UCI Diabetes 130-US Hospitals Dataset

*Following the "Datasheets for Datasets" framework (Gebru et al., 2021)*

---

## Motivation

**Q: For what purpose was the dataset created?**  
To analyse the impact of HbA1c testing on hospital readmission rates among diabetic patients.
The original study (Strack et al., 2014) found that measuring HbA1c at admission was associated
with reduced readmission rates.

**Q: Who created the dataset?**  
Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura,
Krzysztof J. Cios, and John N. Clore. Published in *BioMed Research International*, 2014.

**Q: Was there any funding?**  
Not explicitly stated in the original publication.

---

## Composition

**Q: What do the instances represent?**  
Each row represents one **inpatient encounter** for a diabetic patient.
The same patient may have multiple encounters (rows).

**Q: How many instances are there?**  
101,766 encounters from 130 US hospitals across 10 years (1999–2008).

**Q: What data does each instance contain?**  
50 features including:
- Patient demographics: age (decade buckets), race, gender
- Encounter metadata: admission type, discharge disposition, admission source
- Clinical measures: time in hospital, number of lab procedures, number of procedures,
  number of medications, number of diagnoses, blood glucose test, HbA1c test
- Diagnosis codes: three ICD-9 diagnosis fields (primary, secondary, tertiary)
- 24 medication columns indicating whether each diabetes drug was prescribed and whether
  the dosage was changed (No / Steady / Up / Down)
- Administrative: payer code (40% missing), medical specialty (49% missing)

**Q: Is there any missing data?**  
Yes. The `?` character represents missing values in string columns:
- `weight`: ~97% missing
- `medical_specialty`: ~49% missing
- `payer_code`: ~40% missing
- `race`: ~2% missing
- `diag_2`, `diag_3`: small amounts missing

**Q: Are there any known errors or noise?**  
- ICD-9 coding practices vary across hospitals and years.
- Some encounters include E-codes (external cause of injury) and V-codes (supplementary
  classification) in diagnosis fields, which are handled differently than numeric ICD-9 codes.
- The `weight` column is essentially unusable due to near-complete missingness.

---

## Collection Process

**Q: How was the data collected?**  
Extracted from the Health Facts database (Cerner Corporation), a national data warehouse
that collects clinical data from participating US hospitals. Data was de-identified per
HIPAA Safe Harbor provisions.

**Q: Over what time period?**  
1999–2008.

**Q: Were the data subjects notified?**  
Data is de-identified administrative/clinical records. Individual notification was not
applicable under the de-identification approach used.

---

## Preprocessing / Cleaning

**Q: Was any preprocessing done before release?**  
Yes. The UCI release is already a cleaned subset from the original Health Facts database.
Encounters meeting specific criteria (diabetic patients, hospital stays of 1–14 days,
laboratory tests performed, medications administered) were included.

**Q: What preprocessing is done in this project?**  
See Notebooks 02 and 03 and `paper_or_report/report.md` for full details.

---

## Uses

**Q: Has the dataset been used before?**  
Yes. The original Strack et al. (2014) paper used it to study HbA1c testing effects.
It has since been widely used in ML research on clinical readmission prediction.

**Q: What tasks is it suitable for?**  
Binary or multi-class classification (readmission prediction). Also suitable for
fairness research, feature importance studies, and clinical informatics education.

**Q: Are there tasks it should NOT be used for?**  
- Direct patient-level clinical decisions without additional validation
- Research on non-diabetic populations
- Generalisation claims beyond the US hospital context of the late 1990s–2000s

---

## Distribution

**Q: Is the dataset publicly available?**  
Yes. Available from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

**Q: Is it under any licence restrictions?**  
The UCI ML Repository does not place restrictions on academic research use.
Users should cite the original Strack et al. (2014) paper.

---

## Maintenance

**Q: Who maintains the dataset?**  
Hosted by the UCI Machine Learning Repository. The dataset is static (not updated).

**Q: Will the dataset be updated?**  
No — it reflects a fixed historical period (1999–2008) and is not actively maintained.

---

## Citation

```
Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., & Clore, J.N. (2014).
Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database
Patient Records. BioMed Research International, 2014, 781670.
https://doi.org/10.1155/2014/781670
```
