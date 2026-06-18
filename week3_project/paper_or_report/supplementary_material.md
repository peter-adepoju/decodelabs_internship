# Supplementary Material

## S1. Full Preprocessing Decision Log

> [PLACEHOLDER — Notebook 05 prints a decision log; copy it here]

Key decisions and their rationale:
- **Delivered orders only:** Cancelled/unavailable orders have no delivery or review data.
- **min_order_value = R$0.01:** Removes zero-value test orders; retains legitimate low-cost orders.
- **First-item product features:** When an order has multiple items from different categories,
  we use the first item's product. *Assumption:* the primary product drives the customer experience.
- **Most-common payment type:** When an order uses multiple payment methods (split payment),
  we report the most frequent. *Assumption:* the dominant payment type is most behaviorally meaningful.

## S2. Hyperparameter Grids

### Random Forest (review score prediction)
```python
{
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': ['balanced', None],
}
```

### XGBoost (review score prediction)
```python
{
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}
```

## S3. Robustness Check Results

> [PLACEHOLDER — fill from Notebook 12]

| Check | Finding |
|-------|---------|
| Shuffled labels | Accuracy dropped to ~[X%] (chance level), confirming model learns real signal |
| Seed sensitivity (seeds 0,1,2,...,9) | Accuracy std = [X] across seeds |
| 50% data ablation | Accuracy = [X] (vs [X] on full data) |
| Pre-2017 only | Accuracy = [X] (vs [X] on full period) |
| Alternative time split (2017-07-01) | Accuracy = [X] |

## S4. SQL Query Results

> [PLACEHOLDER — paste key query outputs from Notebook 06]

Top 5 states by revenue:
```
[INSERT TABLE]
```

Top 10 categories by late-delivery rate:
```
[INSERT TABLE]
```

## S5. SHAP Summary Plot

> [PLACEHOLDER — see paper_or_report/figures/shap_summary.png after Notebook 13]

Key SHAP insights:
- delivery_delay_days: [INSERT direction and magnitude]
- order_value: [INSERT]
- product_category_encoded: [INSERT]
