# Datasheet — FedTB-Nigeria Combined CXR Dataset

*Following the Datasheets for Datasets framework (Gebru et al., 2018)*

---

## Motivation

**For what purpose was the dataset created?**
This combined dataset was assembled for the FedTB-Nigeria research project to
train and evaluate federated learning models for TB detection from chest X-rays.
The individual source datasets were created by the US National Library of Medicine
and Guanganmen Hospital for general TB chest X-ray research.

**Who created the dataset and on whose behalf?**
Source datasets created by the US National Library of Medicine (NLM) Open-i project
and Guanganmen Hospital, Beijing. Combined for this project by [Your Name], AIMS.

**Who funded the creation?**
Source datasets: US NLM and collaborating institutions. This project: [Your funding source].

---

## Composition

**What do the instances represent?**
Each instance is a chest X-ray (CXR) image with a binary label:
0 = TB negative, 1 = TB positive.

**How many instances?**
- Montgomery County: 138 images (58 TB+, 80 TB-)
- Shenzhen Hospital: 662 images (336 TB+, 326 TB-)
- Combined: ~800 images (split varies by experiment)

**Does the dataset contain all possible instances or a sample?**
A sample of CXRs collected at specific sites and time periods.

**What data does each instance consist of?**
- A PNG image of a chest X-ray (variable resolution, resized to 224×224 for training)
- A binary label (0 or 1)
- Dataset origin (montgomery / shenzhen)

**Is there a label or target associated with each instance?**
Yes. TB positive (1) vs TB negative (0).
Labels are derived from radiologist report and/or sputum microscopy.

**Is any information missing from individual instances?**
- Patient age, sex, HIV status: NOT AVAILABLE in either source dataset
- Radiologist confidence or inter-rater disagreement: NOT AVAILABLE
- Date of acquisition: NOT AVAILABLE

**Are relationships between individual instances made explicit?**
No. Images are treated as independent. Some patients may have multiple images —
this is not tracked and may constitute a minor source of leakage if the same
patient appears in both train and test sets.

**Are there recommended data splits?**
No official split. We use stratified 70/15/15 train/val/test.

**Does the dataset contain data that might be considered confidential?**
Source datasets are publicly released by the data providers for research use.
No personally identifiable information is included.

**Does the dataset contain data that might be offensive, insulting, or threatening?**
No.

---

## Collection Process

**How was the data associated with each instance acquired?**
- Montgomery: Standard clinical CXR acquired through the Montgomery County TB control program.
- Shenzhen: Clinical CXR from Shenzhen No. 3 People's Hospital TB screening.

**What mechanisms or procedures were used to collect the data?**
Routine clinical chest X-ray imaging; data released by NLM for research.

**Who was involved in the data collection process?**
Radiologists and clinical staff at source institutions.

**Over what timeframe was the data collected?**
Montgomery: 2014 (reported in Jaeger et al., 2014).
Shenzhen: 2012 (reported in Jaeger et al., 2014).

**Were any ethical review processes conducted?**
Source data released by US NLM under public domain licence; institutional ethics
at source hospitals would have covered original collection.

---

## Preprocessing and Cleaning

**Was any preprocessing done?**
For this project:
1. Images resized to 224×224 pixels
2. Grayscale converted to 3-channel RGB (channel replication)
3. Normalised using ImageNet mean and std
4. Training augmentations: random horizontal flip, rotation ±10°, colour jitter

**Was the raw data saved?**
Raw images are stored in `data/raw/` (not committed to git).
Processed manifests are in `data/processed/`.

---

## Uses

**Has the dataset been used for any tasks already?**
Both source datasets are widely used benchmarks for TB CXR classification.
Key references: Jaeger et al. (2014), Lakhani & Sundaram (2017).

**What other tasks could the dataset be used for?**
Binary TB classification, CXR quality assessment, data augmentation research.

**Is there anything about the composition or collection of this dataset
that might impact future uses?**
Yes — see `limitations.md` Section 1 for a full discussion.
Key points: non-Nigerian origin, no demographic metadata, small size.

**Are there tasks for which this dataset should NOT be used?**
- Clinical diagnosis without prospective validation
- Research making generalisation claims to African populations
  without clearly acknowledging the data source limitation

---

## Distribution

**How will the dataset be distributed?**
Source datasets are available from:
- Montgomery: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
- Shenzhen:   https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip

Combined manifests and split files are distributed with this repository.
Raw images are NOT included in the repository.

**Will the dataset be distributed under a copyright or other IP licence?**
Source datasets: public domain.
Combined manifest and split files: MIT licence (same as code).

---

## Maintenance

**Who will be supporting / hosting / maintaining the dataset?**
Source datasets maintained by US NLM.
This project's combined manifest maintained by [Your Name].

**How can the owner be contacted?**
[Your email]

---

## References

- Gebru et al. (2018). Datasheets for Datasets. arXiv:1803.09010.
- Jaeger et al. (2014). Two public chest X-ray datasets for computer-aided screening
  of pulmonary diseases. Quantitative Imaging in Medicine and Surgery.
