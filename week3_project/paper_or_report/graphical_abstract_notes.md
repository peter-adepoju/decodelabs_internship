# Graphical Abstract Notes

## Purpose
A graphical abstract is a single-image summary of the entire project.
These notes describe what to include when designing it (e.g. in Canva, Figma, or PowerPoint).

## Recommended Layout (3-column flow)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   DATA               ANALYSIS                  OUTPUTS                      │
│                                                                              │
│  [Olist DB]    →    [RFM Scoring]      →    [5 Segments]                    │
│  9 CSV files        Recency, Frequency,      Champions / Loyal /            │
│  100k orders        Monetary                 At Risk / Hibernating / Lost   │
│                                                                              │
│                →    [Delivery Model]   →    [Late Delivery Predictor]       │
│                     Random Forest            Binary: is_late?               │
│                     10 features              Accuracy = [X]%                │
│                                                                              │
│                →    [Review Model]     →    [Review Score Predictor]        │
│                     XGBoost                  1-5 stars                      │
│                     12 features              Accuracy = [X]%                │
│                                                                              │
│                →    [SQL + EDA]        →    [Dashboard]                     │
│                     DuckDB queries           Streamlit (6 pages)            │
│                     Revenue, delivery,       Live filterable charts         │
│                     category analysis                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Numbers to Feature
- 100,000+ orders analysed
- 5 customer segments identified
- [X]% late-delivery accuracy (RF)
- [X]% review score accuracy (XGBoost)
- Delivery delay = #1 predictor of bad reviews
- Champions = [X]% of customers, [Y]% of revenue

## Colour Palette (consistent with dashboard)
- Primary: #2196F3 (steel blue)
- Accent:  #FF9800 (orange)
- Success: #4CAF50 (green)
- Warning: #F44336 (red)
- Background: #FAFAFA (off-white)

## Font
- Title: Bold, 20pt
- Section headers: SemiBold, 14pt
- Body: Regular, 11pt

## File Formats
- Export as: `graphical_abstract.png` (300 DPI) and `graphical_abstract.pdf`
- Target dimensions: 1200 x 600 px (landscape, 2:1 ratio)
