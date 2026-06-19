# FedTB-Nigeria: Federated Learning for Tuberculosis Diagnosis from Chest X-Rays Across Nigerian Teaching Hospitals

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Flower](https://img.shields.io/badge/FL-Flower%20(flwr)-green.svg)](https://flower.dev/)

---

## Abstract

This project studies privacy-preserving federated learning for tuberculosis
(TB) screening from chest X-rays using the Montgomery County and Shenzhen
datasets. The study simulates Nigerian teaching hospitals with site-aware data
splits and compares three model settings: a centralised baseline, federated
learning without differential privacy, and federated learning with
Opacus-based differential privacy.

Key results from the final evaluation:
- The test set contains 120 images, with 59 TB-positive and 61 TB-negative cases.
- The centralised model reaches 0.8939 AUC-ROC with 0.8136 sensitivity and
  0.8525 specificity.
- Federated learning without DP reaches 0.9086 AUC-ROC, but with lower
  sensitivity than the centralised model.
- Federated learning with DP reaches 0.7296 AUC-ROC and 0.6928 balanced
  accuracy.
- The centralised vs federated+DP comparison shows a statistically significant
  difference in errors, with McNemar p = 0.0125.
- Non-inferiority is not established for the federated+DP model under the
  0.05 AUC margin used in the report.

All code, figures, tables, and report sources are in this repository.

---

## Hugging Face Space Demo

This repository includes a public-facing Gradio demo for the FedTB-Nigeria research prototype.

- App entrypoint: `app.py`
- Gradio demo module: `app_or_demo/gradio_app.py`
- Model weights: `models/centralised/best_model.pth`

**Important:** This is a research prototype only. It is not clinically validated and must not be used for diagnosis or treatment decisions.

To run locally:

```bash
pip install -r requirements.txt
python app.py
```

Notes:
- Run this from the project root.
- The app expects the trained model at `models/centralised/best_model.pth`.
- If that file is missing, run the centralised training notebook or pipeline
  first.
- The demo is for research only and is not a clinical tool.

---

## 1. Introduction

**FedTB-Nigeria** is a federated learning research project for tuberculosis (TB) detection from chest X-ray images, designed to simulate a multi-hospital deployment across Nigerian teaching hospitals without sharing raw patient data between sites.

Tuberculosis remains a major public health challenge in Nigeria, which carries one of the highest TB burdens globally. Chest X-ray (CXR) screening using AI offers a scalable diagnostic aid — but centralising sensitive hospital imaging data is ethically problematic, legally constrained, and often logistically infeasible. Federated Learning (FL) allows each hospital to train a local model on its own data, sharing only model gradients (not images) with a central aggregator. Differential Privacy (DP) further protects against gradient-based inference attacks.

This project:
- Implements FedAvg-based federated learning using the **Flower (flwr)** framework
- Uses a **ResNet-18** backbone pre-trained on ImageNet, fine-tuned for binary TB classification
- Applies **differentially private training** (Opacus) at the client level
- Benchmarks federated models against a **centralised baseline**
- Evaluates performance rigorously with AUC-ROC, sensitivity, specificity, and bootstrap confidence intervals

---

## Research Question

> Can a differentially-private federated learning system trained across simulated Nigerian teaching hospital sites achieve TB detection performance statistically comparable to a centralised training baseline, while preserving patient privacy and supporting cross-site generalisation?

---

## Hypothesis

A federated ResNet-18 model trained with FedAvg and (ε, δ)-differential privacy will achieve:
- AUC-ROC ≥ 0.85 on the held-out test set
- Sensitivity ≥ 0.80 at clinical operating threshold
- AUC-ROC within 0.05 of the centralised baseline (non-inferiority margin)

*These thresholds are project benchmarks, not fabricated results. Actual values depend on running the full pipeline.*

---

## 2. Data

- Dataset: Montgomery County TB Chest X-ray Set and Shenzhen Hospital Chest X-ray
  Set
- Combined size: 800 images
- Class balance: 394 TB-negative and 406 TB-positive images
- Test split: 120 images
- Image size: 224 x 224 RGB after preprocessing
- Simulation setup: 5 federated sites created with Dirichlet partitioning

### Data Limitations

- The source data are not Nigerian.
- No age, sex, HIV status, or other demographic metadata are available.
- Site partitioning is simulated, not collected from real hospitals.
- The dataset is small by deep-learning standards.

### Data and Report Assets

- Dataset: [Google Drive folder](https://drive.google.com/drive/folders/1ls15Nr6Qk-s24xus3267OMPzMIW8mWIc?usp=sharing)
- Models: [Google Drive folder](https://drive.google.com/drive/folders/1hhn54GzjOyQGQ15wcDp3pELxz5PPi88F?usp=sharing)
  
- [Full report PDF 1](fedtb_nigeria_latex_report1.pdf)
- [Full report PDF 2](fedtb_nigeria_latex_report2.pdf)
- [Full report PDF 3](fedtb_nigeria_latex_report3.pdf)
- [Full report PDF 4](fedtb_nigeria_latex_report4.pdf)
- [Appendix](fedtb_nigeria_latex_report_appendix.pdf)

---

## 3. Methods

### 3.1 Overview

Multi-site CXRs (simulated Nigerian hospital sites)
        │
        ▼
┌───────────────────────────────────────┐
│  Per-client local training            │
│  ResNet-18 + DP-SGD (Opacus)         │
│  Privacy budget: ε ≤ 8, δ = 1e-5     │
└───────────────────────────────────────┘
        │   gradient aggregation (no raw data)
        ▼
┌───────────────────────────────────────┐
│  FedAvg Server (Flower framework)     │
│  Rounds: 50  │  Min clients: 3        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Global model evaluation              │
│  AUC, Sensitivity, Specificity        │
│  Bootstrap CI, McNemar test           │
└───────────────────────────────────────┘
        │
        ▼
   Centralised baseline comparison

### 3.2 Data Preparation

- Cleaned the dataset to 800 usable images.
- Resized all images to 224 x 224.
- Converted grayscale X-rays to 3-channel inputs for ResNet-18.
- Applied standard normalisation and augmentation.
- Split the data into train, validation, and test sets with a held-out test set.

### 3.3 Federated Learning

- Framework: Flower
- Strategy: FedAvg
- Site setup: 5 simulated teaching hospitals
- Backbone: ImageNet-pretrained ResNet-18

### 3.4 Differential Privacy

- Library: Opacus
- Privacy target in the report: (epsilon = 8, delta = 1e-5)
- BatchNorm layers were replaced with GroupNorm for DP compatibility

### 3.5 Evaluation

- AUC-ROC
- AUC-PRC
- Sensitivity
- Specificity
- PPV
- F1
- Balanced accuracy
- Bootstrap confidence intervals
- McNemar test
- Bootstrap AUC difference test

---

## 4. Results

### 4.1 Exploratory Findings

![Class distribution](paper_or_report/figures/class_distribution.png)

![Pixel intensity distributions](paper_or_report/figures/pixel_intensity_distributions.png)

- The dataset is close to balanced, with a TB-positive rate of 49.2%.
- Mean pixel intensity differs slightly by label, which the report flags as a
  possible confound.

### 4.2 Core Performance Comparison

![ROC and PRC comparison](paper_or_report/figures/fig1_roc_prc.png)

![Confusion matrices](paper_or_report/figures/fig3_confusion_matrices.png)

| Model | AUC-ROC | Sensitivity | Specificity | F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Centralised | 0.8939 | 0.8136 | 0.8525 | 0.8276 | 0.8330 |
| Federated (No DP) | 0.9086 | 0.7627 | 0.9180 | 0.8257 | 0.8404 |
| Federated + DP | 0.7296 | 0.7627 | 0.6230 | 0.7087 | 0.6928 |

### 4.3 Statistical Comparison

- McNemar test statistic: 6.2439
- McNemar p-value: 0.0125
- Centralised AUC: 0.8939
- Federated+DP AUC: 0.7296
- Delta AUC: 0.1642
- 95% CI for delta AUC: [0.0755, 0.2538]
- Non-inferiority: not established at the 0.05 margin used in the report

### 4.4 Error Analysis and Interpretability

![Error analysis examples](paper_or_report/figures/error_analysis_examples.png)

![GradCAM examples](paper_or_report/figures/gradcam_examples.png)

- False negatives are the most clinically important errors because they miss
  TB cases.
- The report includes GradCAM visualisations to show which image regions drive
  model predictions.
- The centralised model misses 11 TB-positive cases on the test set, while
  the reported error analysis also shows 9 false positives.

### 4.5 Privacy-Utility Trade-off

![FL + DP training curves](paper_or_report/figures/fl_dp_training_curves.png)

The report's DP analysis shows that stronger privacy generally reduces utility.
The final combined FL+DP model does not match the centralised baseline.

---

## 5. Discussion

The report supports four main conclusions:

1. The data are usable for a TB screening study, but the sample is small and
   non-Nigerian.
2. Federated learning without DP can perform competitively on AUC-ROC.
3. Adding DP introduces a clear performance cost.
4. Error analysis and interpretability remain essential for clinical trust.

Recommended next steps:

1. Validate on real Nigerian chest X-ray data.
2. Revisit the privacy-utility trade-off with additional DP tuning.
3. Add calibration analysis before any decision-support use.

---

## 6. Limitations

- The study uses simulated hospital sites, not real hospitals.
- The data do not include demographic metadata.
- Generalisation to Nigerian patients is not established.
- Clinical deployment is not supported by this work.
- Differential privacy is studied as a research prototype, not a final clinical
  safeguard.

See also:
- [Limitations note](paper_or_report/limitations.md)
- [Model card](paper_or_report/model_card.md)
- [Datasheet](paper_or_report/datasheet.md)

---

## 7. Reproducibility

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_all.py
```

If you prefer conda:

```bash
conda env create -f environment.yml
conda activate fedtb_nigeria
```

---

## 8. Project Structure

```text
week4_project/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── Makefile
├── .gitignore
├── configs/
│   └── config.yaml                  # All hyperparameters and paths
├── data/
│   ├── raw/                         # Downloaded datasets (not committed to git)
│   ├── interim/                     # Cleaned / re-indexed
│   ├── processed/                   # Normalised, split, ready for training
│   ├── external/                    # Supplementary data
│   └── mock/                        # Synthetic mini-dataset for tests
├── notebooks/
│   ├── 02_data_loading_and_first_inspection.ipynb
│   ├── 03_data_cleaning_and_quality_checks.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── 05_preprocessing_and_augmentation.ipynb
│   ├── 06_federated_site_simulation_and_splits.ipynb
│   ├── 07_centralised_baseline_resnet.ipynb
│   ├── 08_federated_learning_with_flower.ipynb
│   ├── 09_differential_privacy_with_opacus.ipynb
│   ├── 10_federated_dp_combined_training.ipynb
│   ├── 11_model_evaluation_and_statistical_comparison.ipynb
│   ├── 12_error_analysis_and_interpretability.ipynb
│   ├── 13_robustness_checks_and_ablations.ipynb
│   ├── 14_fairness_bias_and_ethics_checks.ipynb
│   ├── 15_figures_and_tables.ipynb
│   └── 16_gradio_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                    # Config loader
│   ├── data_utils.py                # Image loading, dataset classes
│   ├── model.py                     # ResNet-18 builder
│   ├── fl_client.py                 # Flower client definition
│   ├── fl_server.py                 # Flower server / strategy
│   ├── dp_utils.py                  # Opacus DP wrappers
│   ├── metrics.py                   # AUC, sensitivity, bootstrap CI
│   ├── visualization.py             # Publication-quality plot helpers
│   └── paths.py                     # Centralised path management
├── scripts/
│   ├── download_data.py             # Automated dataset download
│   ├── run_centralised.py           # Train centralised baseline
│   ├── run_federated.py             # Run FL simulation
│   └── run_all.py                   # End-to-end pipeline
├── tests/
│   ├── test_config.py
│   ├── test_data_utils.py
│   ├── test_model.py
│   ├── test_metrics.py
│   ├── test_fl_client.py
│   └── test_dp_utils.py
├── models/
│   ├── centralised/                 # Saved centralised model weights
│   └── federated/                   # Saved FL model rounds
├── paper_or_report/
│   ├── report.md
│   ├── references.bib
│   ├── figures/
│   ├── tables/
│   ├── model_card.md
│   ├── datasheet.md
│   └── limitations.md
└── app_or_demo/
    └── gradio_app.py
```

---

## 9. Report Figures

The main publication figures used in the report are stored here:

- [Figure 1: ROC and PRC comparison](paper_or_report/figures/fig1_roc_prc.png)
- [Figure 2: Privacy-utility trade-off](paper_or_report/figures/fl_dp_training_curves.png)
- [Figure 3: Confusion matrices](paper_or_report/figures/fig3_confusion_matrices.png)
- [Class distribution](paper_or_report/figures/class_distribution.png)
- [Pixel intensity distributions](paper_or_report/figures/pixel_intensity_distributions.png)
- [Error analysis examples](paper_or_report/figures/error_analysis_examples.png)
- [GradCAM examples](paper_or_report/figures/gradcam_examples.png)

For the full technical write-up, see the PDF reports listed above.

## Citation

If you use this project, please cite:

```
@software{fedtb_nigeria_2026,
  author = {[Your Name]},
  title  = {FedTB-Nigeria: Federated Learning for TB Diagnosis from Chest X-Rays},
  year   = {2026},
  url    = {https://github.com/peter-adepoju/decodelabs_internship/week4_project}
}
```
