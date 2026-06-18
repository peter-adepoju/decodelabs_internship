# Streamlit Dashboard — E-Commerce Customer Intelligence

## Overview

An interactive 6-page dashboard for exploring the Olist e-commerce analysis.

## Setup

**Prerequisites:** Run notebooks 00–10 first to generate:
- `data/processed/master.parquet`
- `data/processed/rfm_scores.parquet`
- `models/rf_review_model.joblib`

## Launch

From the project root:
```bash
streamlit run website_or_demo/app.py
```

Then open: http://localhost:8501

## Pages

| Page | Description |
|------|-------------|
| 📊 Overview | KPI cards (orders, revenue, avg review, late %) + monthly revenue trend |
| 🗺  Geography | Revenue and late-delivery rate by Brazilian state |
| 🛍  Products | Top categories by revenue; revenue vs review bubble chart |
| 🚚 Delivery | Delay distribution histogram; avg review by delay bucket |
| 👥 Customers | RFM segment pie chart, scatter, and summary table |
| 🔮 Predict | Sliders → Random Forest review score prediction |

## Sidebar Filters

All pages (except Predict) respond to:
- **Date range picker** — filter to a specific time window
- **State selector** — filter to one or more customer states

## Architecture

```
app.py
├── load_master()     @st.cache_data — reads master.parquet
├── load_rfm()        @st.cache_data — reads rfm_scores.parquet
├── load_model()      @st.cache_resource — loads rf_review_model.joblib
├── sidebar()         navigation + date/state filters
├── apply_filters()   applies sidebar filters to master dataframe
└── page_*()          one function per page
```

## Deployment (optional)

To deploy on Streamlit Community Cloud:
1. Push the project to a public GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo, set `website_or_demo/app.py` as the main file
4. Upload `data/processed/master.parquet` and `rfm_scores.parquet` via LFS or cloud storage

Note: The prediction page requires `models/rf_review_model.joblib`.
