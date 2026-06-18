# Limitations and Risks

## Data Limitations

| # | Limitation | Impact | Mitigation |
|---|-----------|--------|------------|
| 1 | No customer demographics (age, income, gender) | Cannot segment by customer profile; RFM is behaviour-only | Acknowledged in results |
| 2 | Geographic resolution is state-level, not city or zip | Distance estimation is imprecise | Seller-state + customer-state proxy used |
| 3 | Dataset covers 2016–2018 only | Seasonal / economic shifts not represented | Noted as out-of-distribution risk |
| 4 | Single-platform (Brazil) | Results may not generalise to other geographies | Stated explicitly |
| 5 | Product descriptions / images absent | No text or visual features | Out of scope for this project |
| 6 | Right-censored late orders | Orders close to dataset end may have incomplete delivery records | Filtered to fully-delivered orders |
| 7 | Review score reflects multiple factors | Delivery delay is one of several drivers; product quality, packaging, and seller communication also matter | Interpretability analysis; overclaiming avoided |

## Modelling Limitations

| # | Limitation | Impact | Mitigation |
|---|-----------|--------|------------|
| 1 | Review score is highly class-imbalanced (many 5-star) | Accuracy metric is misleading; macro-F1 used alongside | Explicit class balance check; macro-F1 reported |
| 2 | Integer encoding of state/category ignores geography and semantics | Some signal may be lost | Noted; target encoding as future work |
| 3 | Feature set limited to delivery and product variables | Unmeasured confounders exist | Stated in limitations |
| 4 | Time split may not fully represent seasonal generalisation | Train/test performance gap may understate real-world error | Robustness check across years included |
| 5 | No hyperparameter tuning reported | Best model may be suboptimal | RandomizedSearchCV with nested CV as future work |
| 6 | RFM segments are heuristic labels | Segment names are standard convention, not empirically derived | Label assignment documented; K-Means comparison included |

## Statistical Limitations

| # | Limitation | Impact | Mitigation |
|---|-----------|--------|------------|
| 1 | Bootstrap CI assumes IID samples | Orders from the same customer are not independent | Customer-level grouping noted; group-aware bootstrap as future work |
| 2 | Permutation test uses held-out labels only | Does not account for model training variance | Cross-validation permutation test as future work |
| 3 | Multiple comparisons across 4 models | Inflates false-positive risk | Bonferroni correction applied |

## Ethical Risks

| # | Risk | Mitigation |
|---|------|------------|
| 1 | Model-driven personalisation could reinforce existing customer inequalities | All features are behavioural, not demographic | Noted |
| 2 | Late-delivery flagging could unfairly disadvantage sellers in remote states | Model errors by state are reported; recommendation is framed as support, not penalty | Error analysis by state in Notebook 13 |
| 3 | Predictions are probabilistic, not deterministic | Users told to treat as estimates only | Dashboard disclaimer included |

## Future Work

1. **Target encoding / embedding** for high-cardinality categoricals.
2. **Customer lifetime value modelling** with Pareto/NBD or BG/NBD models.
3. **Causal inference** to estimate the treatment effect of free shipping on order value.
4. **Time-series forecasting** of monthly demand by category.
5. **NLP on review text** to extract sentiment and specific complaint topics.
6. **Geospatial analysis** with actual distance between seller and customer coordinates.
