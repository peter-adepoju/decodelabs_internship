# E-Commerce Customer Intelligence Report
## Segmentation, Delivery Performance, and Review Score Prediction using the Olist Dataset

**Authors:** [Your Name]
**Institution:** [Your Institution]
**Date:** [Run date — filled automatically by Notebook 15]
**Version:** 0.1 (draft)

---

## Abstract

> **[PLACEHOLDER — fill after running all notebooks]**
>
> This report presents a complete data science analysis of the Olist Brazilian
> E-Commerce public dataset (~100,000 orders, 2016–2018).  We address three
> questions: (1) Can customers be segmented into actionable groups using RFM
> analysis?  (2) What operational and product features are most predictive of
> delivery delays?  (3) What drives customer satisfaction (review scores)?
>
> Key results: [INSERT AFTER RUNNING]
> - Champions segment accounts for X% of revenue with Y% of customers.
> - Late delivery is the strongest predictor of low review scores (r = ?).
> - Random Forest achieves accuracy = ? (95% CI: ? – ?) on review score
>   prediction, versus ? for a majority-class baseline.
>
> All code, data download instructions, and reproducibility steps are available
> at: [GITHUB URL]

---

## 1. Introduction

### 1.1 Business Context

E-commerce has grown rapidly in Brazil, and Olist operates as a marketplace
aggregator connecting thousands of small sellers to major retail platforms.
Understanding customer purchasing behaviour, delivery operations, and satisfaction
drivers is critical for seller retention, customer loyalty, and platform growth.

### 1.2 Research Gap

While aggregate e-commerce metrics are widely reported, few public analyses apply
a full pipeline — from raw relational data to interpretable segmentation, predictive
modelling, and actionable recommendations — using transparent, reproducible code.

### 1.3 Research Questions

1. **Segmentation:** Can customers be meaningfully grouped using RFM metrics,
   and do these groups differ in revenue contribution and churn risk?

2. **Operational prediction:** Which order- and product-level features at the
   time of purchase best predict whether a delivery will be late?

3. **Satisfaction prediction:** What features are most associated with low
   review scores, and can we predict them reliably?

### 1.4 Hypotheses

- **H1 (Segmentation):** A small proportion of customers (Champions) will
  account for a disproportionately large share of revenue (Pareto principle).

- **H2 (Delivery):** Seller state, product weight, and order value will be
  among the top predictors of late delivery.

- **H3 (Review):** Delivery delay will be the single strongest predictor of
  review score; a model trained on operational features will significantly
  outperform a majority-class baseline.

---

## 2. Data

### 2.1 Dataset Description

- **Name:** Olist Brazilian E-Commerce Public Dataset
- **Source:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **License:** Creative Commons CC BY-NC-SA 4.0
- **Size:** ~100 MB uncompressed (9 CSV files)
- **Coverage:** September 2016 – October 2018
- **Scope:** Orders, customers, sellers, products, payments, reviews, geolocation
- **Number of orders:** ~100,000 (delivered orders: ~96,000)

### 2.2 Table Relationships

```
customers ──< orders ──< order_items >── products
                │                         │
                │                    (category_translation)
                ├──< order_payments
                └──< order_reviews
                        │
                    sellers
```

### 2.3 Data Limitations

1. **No customer demographics:** Age, income, and household data are not available.
2. **No product descriptions or images:** Limits NLP/vision modelling.
3. **Geographic proxy only:** City/state rather than exact coordinates for distance.
4. **Single-platform dataset:** Results may not generalise to other marketplaces.
5. **Temporal scope:** Only 2 years of data; seasonal effects may be underrepresented.
6. **Right-censoring:** Orders near the dataset end may have incomplete follow-up.

---

## 3. Methods

### 3.1 Data Cleaning and Feature Engineering

> [PLACEHOLDER — describe steps from Notebook 05]

Key steps:
- Filtered to delivered orders only (N = [INSERT]).
- Parsed all timestamp columns; computed `delivery_delay_days` and `is_late`.
- Aggregated items to order level (sum price, freight, count items).
- Translated product category names from Portuguese to English.
- Encoded categorical variables (state, category, payment type) as integer codes.
- Calendar features: purchase month and day of week.

### 3.2 RFM Analysis

> [PLACEHOLDER — describe from Notebook 07]

- Reference date: max(order_purchase_timestamp) + 1 day.
- Recency scored 1–5 (5 = most recent), Frequency and Monetary scored 1–5.
- Quintile breaks; ties resolved by first-rank method to avoid empty bins.
- Segments assigned via standard RFM naming conventions.

### 3.3 Modelling

**Task A: Late delivery prediction (binary classification)**

Features: order_value, freight_value, freight_ratio, n_items, product_weight_g,
          purchase_month, purchase_dayofweek, seller_state_encoded,
          customer_state_encoded, product_category_encoded

Models: Dummy (majority class), Logistic Regression, Random Forest, XGBoost

Split: Time-based split at 2018-01-01; earlier orders = train, later = test.
       This prevents temporal leakage.

**Task B: Review score prediction (5-class classification)**

Features: All Task A features + delivery_delay_days, payment_type_encoded,
          product_photos_qty

Models: Same as Task A.

Split: Same time-based split.

### 3.4 Evaluation

- Primary metric: Accuracy and macro-averaged F1.
- Uncertainty: 95% bootstrap confidence intervals (N=1,000 resamples).
- Statistical test: Permutation test against null distribution (N=500).
- Effect size: Cohen's h for pairwise proportion comparisons.

### 3.5 Interpretability

- Feature importances (Random Forest Gini impurity).
- Permutation importances (model-agnostic).
- SHAP values (TreeSHAP for Random Forest and XGBoost).
- Partial dependence plots for top 3 features.

---

## 4. Results

> **[ALL RESULTS ARE PLACEHOLDERS — insert after running notebooks 09-14]**

### 4.1 Exploratory Findings

![Monthly revenue trend](figures/monthly_revenue.png)
*Figure 1. Monthly revenue trend, [INSERT DATE RANGE].*

![Review distribution](figures/review_distribution.png)
*Figure 2. Distribution of review scores.*

![Delivery delay](figures/delivery_delay_hist.png)
*Figure 3. Delivery delay distribution. [X%] of orders are delivered late.*

### 4.2 RFM Segmentation

![Segment counts](figures/segment_counts.png)
*Figure 4. Customer segment sizes.*

![RFM scatter](figures/rfm_scatter.png)
*Figure 5. RFM scatter plot coloured by frequency score.*

| Segment | N customers | % Revenue | Avg Recency | Avg Frequency | Avg Monetary |
|---------|-------------|-----------|-------------|---------------|--------------|
| Champions | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| Loyal | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| At Risk | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| Lost | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |

*Table 1. RFM segment summary.*

### 4.3 Model Performance

![Model comparison](figures/model_comparison.png)
*Figure 6. Accuracy ± 95% CI for all models on the test set.*

| Model | Accuracy | Lower 95% CI | Upper 95% CI | Macro F1 | Permutation p |
|-------|----------|--------------|--------------|----------|---------------|
| Dummy | [INSERT] | [INSERT] | [INSERT] | [INSERT] | — |
| Logistic Regression | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| Random Forest | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| XGBoost | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |

*Table 2. Review score prediction results (Task B). N_test = [INSERT].*

### 4.4 Interpretability

![Feature importances](figures/feature_importance.png)
*Figure 7. Top 15 feature importances (Random Forest, Gini).*

---

## 5. Discussion

> [PLACEHOLDER — write after reviewing results]

### 5.1 Key Takeaways

1. [INSERT based on H1 finding]
2. [INSERT based on H2 finding]
3. [INSERT based on H3 finding]

### 5.2 Business Recommendations

1. **Re-engage At-Risk customers** with personalised offers; they account for
   X% of historical revenue but have not purchased in Y days.
2. **Prioritise delivery speed for high-weight products** from states with the
   highest late-delivery rates (see geography analysis).
3. **Alert sellers when predicted review score < 3** so they can proactively
   follow up with at-risk customers before a review is submitted.

### 5.3 Alternative Explanations

- Model performance improvements over baseline may reflect feature engineering
  choices rather than model complexity per se.
- RFM segments are descriptive and correlational; causal claims require
  randomised experiments.

---

## 6. Limitations

See also: [limitations.md](limitations.md)

1. Modest prediction accuracy suggests unmeasured confounders (product quality,
   seller service quality, packaging).
2. RFM analysis does not account for customer lifetime stages.
3. Time-based split may underestimate model generalisation to future time periods
   with different seasonal patterns.
4. Feature encoding (integer codes for state/category) does not capture ordinal
   or geographic relationships.

---

## 7. Ethics, Privacy, and Bias

- All data are anonymised by Olist; no individual can be re-identified.
- No demographic variables (age, gender, income) are used.
- The dataset covers a Brazilian market; results may reflect regional patterns
  not applicable to other geographies (including Nigeria, where this project
  was developed for a portfolio).
- Model errors are distributed non-uniformly across product categories and
  states; see error analysis in Notebook 13.

---

## 8. Reproducibility Statement

All code is available at [GITHUB URL].  To reproduce:
```bash
git clone [GITHUB URL]
cd ecommerce_customer_intelligence
pip install -r requirements.txt
make download      # requires Kaggle API key
make notebooks     # runs all notebooks
```
Random seeds are fixed at 42 throughout.  Approximate runtime: 20-30 minutes.

---

## 9. Data Availability

The Olist dataset is publicly available under CC BY-NC-SA 4.0 at:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Raw CSV files are NOT committed to this repository.  Run `make download`.

---

## 10. References

See [references.bib](references.bib)
