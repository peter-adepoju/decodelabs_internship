# Model Card — FedTB-Nigeria Global ResNet-18

*Following the Model Card framework (Mitchell et al., 2019)*

---

## Model Details

| Field | Value |
|---|---|
| Name | FedTB-Nigeria Global ResNet-18 |
| Version | 0.1.0 (research prototype) |
| Type | Binary image classifier |
| Architecture | ResNet-18, ImageNet pretrained, custom 2-class head |
| Training method | FedAvg federated learning + Opacus DP-SGD |
| Privacy guarantee | (ε=8, δ=1e-5)-Differential Privacy |
| Input | 224×224 RGB chest X-ray image |
| Output | Probability of TB positive (class 1) in [0, 1] |
| Developer | [Your Name], AIMS |
| Date | 2025 |
| License | MIT (code); model weights for research use only |

---

## Intended Use

### Primary Intended Use
Research demonstration of privacy-preserving federated learning for TB screening.
Portfolio and academic publication supporting material.

### Primary Intended Users
- AI / ML researchers studying federated learning for medical imaging
- Public health researchers evaluating AI-assisted TB screening feasibility

### Out-of-Scope Uses
- **Clinical diagnosis of tuberculosis** (NOT validated for clinical use)
- **Screening decisions** without radiologist confirmation
- **Deployment** in any healthcare setting without ethics approval and prospective validation
- **Commercial use** without written permission

---

## Factors

### Relevant Factors
- **Geography**: Model trained on US (Montgomery) and Chinese (Shenzhen) CXR data.
  Performance on Nigerian CXRs is unknown and likely to differ.
- **Equipment**: Source data acquired on specific X-ray equipment. Different equipment
  (exposure, resolution, processing) may affect performance.
- **Disease stage**: Montgomery and Shenzhen datasets include varied TB severity.
  Performance on subclinical or early-stage TB is not separately characterised.
- **Comorbidities**: TB/HIV co-infection — prevalent in Nigeria — is not specifically
  represented in the training data.

### Evaluation Factors
- **Site**: Performance reported on global held-out test set; per-site analysis is
  simulated (not from real hospital sites).
- **Label source**: Radiologist report ± sputum microscopy.

---

## Metrics

| Metric | Centralised | FL (No DP) | FL + DP (ε=8) |
|---|---|---|---|
| AUC-ROC | [see reports/tables] | [see reports/tables] | [see reports/tables] |
| Sensitivity | [see reports/tables] | [see reports/tables] | [see reports/tables] |
| Specificity | [see reports/tables] | [see reports/tables] | [see reports/tables] |
| F1 | [see reports/tables] | [see reports/tables] | [see reports/tables] |

*All metrics with 95% bootstrap CI (n=1000 resamples). Fill in after running full pipeline.*

---

## Training Data

| Dataset | N | TB+ | Source | License |
|---|---|---|---|---|
| Montgomery County CXR | 138 | 58 | US NLM | Public domain |
| Shenzhen Hospital CXR | 662 | 336 | NLM / Guanganmen Hospital | Public domain |

Training split: 70% of combined data, partitioned into 5 simulated hospital sites
via Dirichlet(α=0.5) distribution.

---

## Evaluation Data

Held-out test split: 15% of combined data (stratified by label).
NOT used during training or hyperparameter selection.
Bootstrap CI computed with n=1000 resamples.

---

## Ethical Considerations

- No real patient data are included in this repository.
- Source datasets are publicly released for research purposes.
- Differential privacy (ε=8) limits gradient-based membership inference attacks.
- The model has not been validated for clinical use.
- Known bias: training data is non-Nigerian and lacks demographic metadata.
- Deployment would require ethics approval, regulatory clearance, and prospective validation.

---

## Caveats and Recommendations

1. **Do not deploy clinically** without prospective validation on Nigerian CXR data.
2. **Retrain** on local data when available, using this model as a starting point.
3. **Calibrate** the probability output before use in decision support.
4. **Monitor** for performance drift if deployed prospectively.
5. **Audit** for demographic fairness once data with patient metadata is available.

---

## References

- Mitchell et al. (2019). Model Cards for Model Reporting. FAccT 2019.
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks. AISTATS.
- Abadi et al. (2016). Deep Learning with Differential Privacy. CCS.
