# Supplementary Material

## S1. Full Preprocessing Decision Log

- Delivered orders only were used because cancelled and unavailable orders do
  not have complete delivery or review outcomes.
- Zero-value test orders were removed.
- When an order contained multiple items, the first item's product was used for
  product-level features.
- For split payments, the most common payment type was used.

## S2. Hyperparameter Grids

### Random Forest

```python
{
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': ['balanced', None],
}
```

### XGBoost

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

| Check | Finding |
|-------|---------|
| Shuffled labels | Accuracy dropped to chance level in the control run. |
| Seed sensitivity | Accuracy stayed stable across seeds 0 to 9. |
| 50% data ablation | Accuracy remained close to the full-data result. |
| Pre-2017 only | Accuracy was lower than the full-period model. |
| Alternative time split | Results were broadly similar, with some drift. |

## S4. SQL Query Results

Top states by revenue and top categories by late-delivery rate are reported in
the generated tables under `reports/tables/`.

## S5. SHAP Summary Plot

The SHAP summary plot in `figures/shap_summary.png` shows delivery delay as the
dominant contributor to review-score predictions, followed by order value and
category-level effects.

