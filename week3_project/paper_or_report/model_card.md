# Model Card: Review Score Predictor (Random Forest)

## Model Overview

| Field | Value |
|-------|-------|
| Model type | Random Forest Classifier (scikit-learn) |
| Version | 0.1 |
| Task | 5-class classification (review scores 1–5) |
| Dataset | Olist Brazilian E-Commerce (2016–2018) |
| Training split | Orders before 2018-01-01 |
| Test split | Orders on/after 2018-01-01 |
| Saved at | models/rf_review_model.joblib |

## Intended Use

- **Primary use:** Predict likely review score for a delivered order given
  operational features (delivery timing, order value, product category).
- **Intended users:** Olist platform analysts, seller operations teams.
- **Out-of-scope uses:** Individual customer surveillance, creditworthiness
  assessment, any decision that materially harms sellers or customers.

## Input Features

| Feature | Type | Description |
|---------|------|-------------|
| delivery_delay_days | float | Actual minus estimated delivery date |
| order_value | float | Sum of item prices (BRL) |
| freight_value | float | Sum of freight charges (BRL) |
| freight_ratio | float | freight_value / order_value |
| n_items | int | Number of items in order |
| product_weight_g | float | Weight of first item (g) |
| product_photos_qty | int | Number of product photos |
| purchase_month | int | Month of purchase (1–12) |
| purchase_dayofweek | int | Day of week (0=Mon, 6=Sun) |
| customer_state_encoded | int | Integer code for customer state |
| product_category_encoded | int | Integer code for product category |
| payment_type_encoded | int | Integer code for payment type |

## Output

- Predicted review score: integer 1–5
- Probability vector: [P(1), P(2), P(3), P(4), P(5)]

## Performance

> [INSERT AFTER RUNNING NOTEBOOK 11]

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | [INSERT] | [INSERT] |
| Macro F1 | [INSERT] | [INSERT] |
| Permutation p-value | [INSERT] | — |

Baseline (majority-class dummy): Accuracy = [INSERT]

## Limitations and Caveats

1. Accuracy is modest because review scores reflect product quality, packaging,
   and seller service — factors not in the feature set.
2. Integer encoding does not capture geographic proximity.
3. Class imbalance (many 5-star reviews) means 1-star and 2-star precision is lower.
4. The model was trained on Brazilian e-commerce data (2016–2018) and may not
   transfer to other markets or time periods.

## Fairness Considerations

- Error rates differ by product category and customer state (see Notebook 13).
- No demographic features are used.
- Predictions should be used as operational signals, not as verdicts.

## Training Details

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
```
