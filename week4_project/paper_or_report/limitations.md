# FedTB-Nigeria: Limitations and Caveats

This document provides a thorough, honest account of the limitations of the
FedTB-Nigeria project. Transparent reporting of limitations is required for
Nature Medicine submission and is good scientific practice.

---

## 1. Dataset Limitations

### 1.1 Non-Nigerian Source Data

The primary datasets used — Montgomery County (USA) and Shenzhen Hospital (China) —
do not originate from Nigeria or sub-Saharan Africa. Consequently:

- The TB pathology, disease stage distribution, and comorbidity patterns (notably
  TB/HIV co-infection, which is prevalent in Nigeria) are not represented.
- Imaging equipment, exposure settings, and radiographic technique differ between the
  source sites and typical Nigerian teaching hospital equipment.
- Radiologist labelling conventions may differ across sites and countries.

**Impact**: Model performance reported here cannot be assumed to generalise to real
Nigerian chest X-rays without independent validation.

**Mitigation**: We clearly label this as a simulation study. Deployment would require
retraining or fine-tuning on representative Nigerian CXR data.

### 1.2 Small Dataset Size

The combined dataset (n ≈ 800 images) is small by deep learning standards. This:

- Increases variance in performance estimates.
- Limits the model's ability to learn from heterogeneous presentations.
- Makes bootstrap CI estimates somewhat wide.

**Mitigation**: We use ImageNet-pretrained ResNet-18 (transfer learning), which is
appropriate for small medical imaging datasets. We report bootstrap CIs throughout.

### 1.3 Label Quality

TB labels in both source datasets are derived from radiologist report and/or sputum
smear microscopy — both of which have imperfect sensitivity and specificity. Some
labelled "TB negative" cases may be early-stage TB; some "TB positive" labels may
reflect resolved TB or non-specific consolidation.

**Impact**: Ground-truth label noise may inflate or deflate reported AUC depending on
the direction of systematic mislabelling.

---

## 2. Federated Learning Simulation Limitations

### 2.1 Simulated (Not Real) Hospital Sites

The five "Nigerian teaching hospital" sites are a mathematical simulation:
the training data are partitioned from the same source datasets using Dirichlet
sampling (α = 0.5). This does not capture:

- Real between-hospital variation in imaging equipment
- Patient demographic differences (age, sex, HIV status)
- Radiologist inter-rater variability
- Temporal drift in disease prevalence
- Network infrastructure constraints (bandwidth, latency)

**Impact**: FL performance metrics from this simulation may be optimistic compared
to real multi-site deployment.

### 2.2 Synchronous FedAvg

We use synchronous FedAvg, which assumes all clients are available and respond in
every round. In real deployments:

- Hospitals may have connectivity interruptions.
- Some hospitals may have much larger datasets than others (stragglers).
- Asynchronous FL strategies (e.g., FedAsync, FedBuff) may be more appropriate.

### 2.3 No Cross-Site Generalisation Test

We evaluate the global model on a held-out test set that was sampled uniformly
from the combined dataset before site partitioning. We do NOT test whether the
global model generalises to a completely held-out site (leave-one-site-out
evaluation). This is a more demanding and more clinically relevant evaluation
that we recommend for follow-up work.

---

## 3. Differential Privacy Limitations

### 3.1 Per-Round Sequential Composition

We allocate the total epsilon budget (ε) equally across FL rounds, using
sequential composition: ε_total ≤ N_rounds × ε_per_round. This is a valid
but conservative bound. Tighter accounting via moments accountant or RDP
composition across rounds would yield a smaller effective ε, potentially
improving the privacy-utility trade-off without additional noise.

### 3.2 Epsilon Budget Choice

We use ε = 8 as the primary experimental budget. While this value is
commonly used in FL+DP medical imaging research, some privacy researchers
consider ε > 4 to provide a relatively weak formal guarantee in practice.
The choice of ε involves a value judgement (privacy vs utility) that is
context-dependent and that we cannot make on behalf of deploying hospitals.

### 3.3 Gradient Inversion Attacks

Differential privacy protects against worst-case membership inference attacks
on gradient updates. It does not protect against all potential attacks,
including gradient inversion attacks that attempt to reconstruct training
images from gradients. At ε = 8, the level of protection against gradient
inversion may be insufficient for highly sensitive patient data.

### 3.4 Threat Model Scope

The DP guarantee applies only to gradient updates shared between clients and
the server. It does not address:
- Attacks on locally stored model weights at each hospital
- Data breaches at the aggregating server
- Re-identification from aggregated model outputs

---

## 4. Model Architecture Limitations

### 4.1 BatchNorm Replacement

Opacus DP-SGD requires per-sample gradients, which are incompatible with
standard BatchNorm. We replace BatchNorm with GroupNorm, which is known to
perform slightly differently (often slightly lower accuracy) than BatchNorm
for small batch sizes. This introduces a systematic difference between the
centralised model (BatchNorm) and the DP federated model (GroupNorm), which
should be noted when interpreting the performance gap.

### 4.2 ResNet-18 Vs Larger Models

We use ResNet-18 for computational efficiency. Larger architectures (e.g.,
EfficientNet-B3, DenseNet-121, ViT-Base) may yield higher AUC but are more
expensive under DP-SGD (more parameters → more noise needed for the same
privacy budget). We treat ResNet-18 as a reasonable baseline but acknowledge
this is a limitation for state-of-the-art comparison.

---

## 5. Evaluation Limitations

### 5.1 No Calibration Analysis

We do not report calibration metrics (e.g., Brier score, reliability diagrams,
expected calibration error). A well-calibrated model produces probability
outputs that match empirical risk — important for clinical decision support.
Calibration analysis is recommended before any deployment.

### 5.2 No Temporal Validation

Our evaluation is cross-sectional. We do not test whether model performance
degrades over time as TB epidemiology or imaging practices change (temporal
drift). Prospective evaluation is required for deployment.

### 5.3 Fixed Operating Threshold

We report performance at the Youden-optimal threshold. In clinical practice,
the operating threshold should be chosen based on the relative costs of false
negatives (missed TB) and false positives (unnecessary follow-up), which are
context-dependent. We do not optimise for a specific clinical cost function.

---

## 6. Ethics and Deployment Limitations

- No demographic metadata (age, sex, HIV status) is available for subgroup
  fairness analysis. We cannot assess whether the model underperforms for any
  demographic subgroup.
- Site simulation assignments are random — we cannot assess performance equity
  across real Nigerian ethnic groups, geographic regions, or socioeconomic strata.
- Deployment in a clinical setting would require prospective validation,
  ethics committee approval, regulatory clearance, and ongoing monitoring.
- The model should be deployed as a screening aid requiring radiologist
  confirmation, not as a standalone diagnostic system.

---

## 7. Conclusions on Limitations

The FedTB-Nigeria project demonstrates the technical feasibility of combining
FL and DP for TB CXR classification in a simulated multi-hospital setting.
The results are preliminary and subject to all the limitations above.

Considerable further work is required before the system could be considered for
clinical deployment in Nigeria, including: collection of representative Nigerian
CXR data, independent external validation, demographic fairness analysis, and
regulatory approval.

We report these limitations transparently in accordance with best practices for
responsible AI in healthcare (Topol 2019; Keane & Topol 2018; Obermeyer & Emanuel 2016).
