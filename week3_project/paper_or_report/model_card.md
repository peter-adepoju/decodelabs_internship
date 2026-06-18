# Model Card: Review Score Predictor (Random Forest)

## Model Overview

| Field | Value |
|-------|-------|
| Model type | Random Forest Classifier |
| Version | 0.1 |
| Task | 5-class classification of review scores |
| Dataset | Olist Brazilian E-Commerce (2016-2018) |
| Training split | Orders before 2018-01-01 |
| Test split | Orders on or after 2018-01-01 |
| Saved at | `models/rf_review_model.joblib` |

## Intended Use

- Predict likely review score for a delivered order using operational features.
- Support seller operations and customer-experience analysis.

## Input Features

- `delivery_delay_days`
- `order_value`
- `freight_value`
- `freight_ratio`
- `n_items`
- `product_weight_g`
- `product_photos_qty`
- `purchase_month`
- `purchase_dayofweek`
- `customer_state_encoded`
- `product_category_encoded`
- `payment_type_encoded`

## Output

- Predicted review score from 1 to 5
- Probability vector for each class

## Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.5493 | 0.5450 - 0.5534 |
| Macro F1 | 0.2924 | - |
| Permutation p-value | Not reported | - |

Baseline accuracy: 0.5933

## Limitations and Caveats

1. Review scores depend on product quality, packaging, and seller service as
   well as delivery timing.
2. Integer encoding does not capture geographic proximity.
3. Class imbalance lowers performance on 1-star and 2-star reviews.
4. The model is specific to Brazilian e-commerce data from 2016-2018.

## Fairness Considerations

- Error rates differ by product category and customer state.
- No demographic features are used.
- Predictions should be treated as operational signals, not verdicts.

