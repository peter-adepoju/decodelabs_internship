#!/usr/bin/env python3
"""
scripts/generate_notebooks.py
───────────────────────────────
Generates all 18 project notebooks as valid .ipynb files.

Run once from the project root:
    python scripts/generate_notebooks.py

Each notebook is written to notebooks/  with proper Jupyter 4.x format.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def mk_nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }

def md(src):
    lines = (src.lstrip("\n").rstrip() + "\n").splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(src):
    lines = (src.lstrip("\n").rstrip() + "\n").splitlines(keepends=True)
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": lines
    }

def save(filename, cells):
    path = NOTEBOOKS_DIR / filename
    path.write_text(json.dumps(mk_nb(cells), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  Written: {path.name}")


# ─── NB 00 ────────────────────────────────────────────────────────────────────
def nb00():
    cells = [
md("""
# Notebook 00 — Project Overview and Business Question

## Purpose
I use this notebook to define the project context, state the research questions,
and document the hypotheses before touching any data.

## Why this matters
Stating questions and hypotheses upfront prevents post-hoc rationalisation — a
common bias in exploratory data science.  If the question is clear before the
analysis, the results are more trustworthy.

## Inputs
None (no data required).

## Outputs
This notebook is documentation only.  No files are created.

## Connection to main question
This is the scientific foundation for all subsequent notebooks.
"""),
code("""
# Standard library check — confirm the environment is set up
import sys
import importlib

REQUIRED = [
    'pandas', 'numpy', 'scipy', 'sklearn',
    'matplotlib', 'seaborn', 'plotly',
    'duckdb', 'yaml', 'shap', 'xgboost',
    'streamlit', 'joblib',
]

print(f"Python version: {sys.version}")
print()
print("Package availability check:")
all_ok = True
for pkg in REQUIRED:
    try:
        importlib.import_module(pkg)
        print(f"  OK  {pkg}")
    except ImportError:
        print(f"  MISSING  {pkg}  <-- run: pip install -r requirements.txt")
        all_ok = False

if all_ok:
    print("\\nAll required packages are available.")
else:
    print("\\nSome packages are missing. Run: pip install -r requirements.txt")
"""),
md("""
## Business Context

Olist is a Brazilian e-commerce marketplace aggregator.  It connects ~3,000
small sellers to customers through a single storefront on major platforms.
Orders are fulfilled by individual sellers, who are responsible for packaging
and handing items to the carrier.

**Why does this analysis matter?**

- Late deliveries are the strongest known driver of negative reviews.
- A 5-star review average improves seller ranking; poor reviews reduce it.
- Customer segmentation enables targeted retention and loyalty programmes.
- Identifying the best-performing sellers and categories guides platform investment.

## Dataset

**Olist Brazilian E-Commerce Public Dataset**
- Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Coverage: September 2016 – October 2018
- Scale: ~100,000 orders, ~99,000 unique customers, ~3,000 sellers
- Tables: 9 relational CSVs
- License: CC BY-NC-SA 4.0 (free for non-commercial research)
"""),
md("""
## Research Questions

**Primary questions:**

1. **Segmentation:** Can we use RFM (Recency, Frequency, Monetary) analysis to
   segment customers into actionable groups, and what is the revenue distribution
   across segments?

2. **Operational prediction:** Which features at order time best predict whether
   a delivery will be late?

3. **Satisfaction prediction:** Can we predict a customer's review score from
   delivery and order features, and what is the strongest predictor?

**Secondary questions:**

- Which product categories have the highest revenue and the worst delivery performance?
- Which states have the highest late-delivery rates?
- Does the repeat purchase rate differ meaningfully across customer segments?
"""),
md("""
## Hypotheses

| Hypothesis | Prediction | Falsifiable? |
|-----------|-----------|-------------|
| H1: Pareto distribution in customers | Top 20% of customers account for >60% of revenue | Yes — measure with Lorenz curve |
| H2: Delivery delay → bad reviews | Correlation between delivery_delay_days and review_score is negative and significant | Yes — Pearson/Spearman correlation + permutation test |
| H3: RF outperforms baseline | Random Forest accuracy significantly exceeds majority-class baseline on test set | Yes — bootstrap CI + permutation test |
| H4: Weight predicts late delivery | product_weight_g is in top-5 features for late delivery prediction | Yes — feature importance / SHAP |

## Alternative Explanations to Rule Out

- **Data leakage:** delivery_delay_days is known only after delivery, so it must
  NOT be used to predict whether a delivery is late (we can use it to predict review score).
- **Class imbalance:** 5-star reviews are the majority class.  A dummy model
  always predicting 5 stars may appear accurate; macro F1 is required.
- **Temporal confounding:** order volume grew over time; models trained on early
  data and tested on late data may see distribution shift.
"""),
code("""
# Print a clean summary of the notebook sequence
notebooks = [
    ("00", "Project overview and hypotheses", "Documentation"),
    ("01", "Dataset selection and access", "Download guide"),
    ("02", "Data loading and first inspection", "Shapes, dtypes, .head()"),
    ("03", "Metadata and schema audit", "Column audit"),
    ("04", "Data quality checks", "Missing values, duplicates"),
    ("05", "Cleaning and feature engineering", "master.parquet"),
    ("06", "Exploratory analysis + SQL", "Charts, SQL results"),
    ("07", "RFM analysis and segmentation", "rfm_scores.parquet"),
    ("08", "Leakage control and split design", "Train/test indices"),
    ("09", "Baseline models", "Dummy + LogReg results"),
    ("10", "Advanced models", "RF + XGBoost + K-Means"),
    ("11", "Evaluation and statistical inference", "Bootstrap CIs, p-values"),
    ("12", "Robustness checks", "Ablations, shuffled labels"),
    ("13", "Interpretability and error analysis", "SHAP, confusion matrices"),
    ("14", "Publication figures and tables", "Final PNG/CSV outputs"),
    ("15", "Report generation", "Summary tables"),
    ("16", "Limitations and next steps", "Honest assessment"),
    ("17", "Dashboard preview", "Streamlit walkthrough"),
]

print(f"{'NB':>3}  {'Title':<45} {'Primary Output'}")
print("-" * 70)
for nb_id, title, output in notebooks:
    print(f"{nb_id:>3}  {title:<45} {output}")
"""),
]
    save("00_project_overview_and_business_question.ipynb", cells)


# ─── NB 01 ────────────────────────────────────────────────────────────────────
def nb01():
    cells = [
md("""
# Notebook 01 — Dataset Selection and Access

## Purpose
I document why the Olist dataset is the right fit for this project, how to
download it, what the files contain, and what prior work has used this data.

## Inputs
None (run before downloading data).

## Outputs
`data/raw/.gitkeep` — the raw directory is confirmed to exist.

## Connection to main question
Good dataset choice is the foundation of valid analysis.  I check here that
the data has everything needed to answer the research questions.
"""),
code("""
import sys
from pathlib import Path

# I always add the project root to sys.path so I can import from src/
sys.path.insert(0, str(Path('..').resolve()))

from src.config import load_config
from src.paths import Paths

cfg = load_config()
paths = Paths(cfg)

print("Project:", cfg['project']['name'])
print("Version:", cfg['project']['version'])
print()
print("Key paths:")
print(f"  raw data:   {paths.raw}")
print(f"  processed:  {paths.processed}")
print(f"  models:     {paths.models}")
print(f"  figures:    {paths.reports_figs}")
"""),
md("""
## Dataset Selection Rationale

### Why Olist?

| Criterion | Assessment |
|-----------|-----------|
| Real, not simulated | Yes — production data from Olist |
| Freely available | Yes — Kaggle, CC BY-NC-SA 4.0 |
| Relational structure | Yes — 9 linked tables (exercises SQL joins) |
| Rich enough for segmentation | Yes — customer IDs, timestamps, order values |
| Sufficient for ML | Yes — ~100k orders, multiple prediction targets |
| Brazilian market (relevant to African e-commerce context) | Emerging-market dynamics similar to Nigerian e-commerce |

### Alternatives Considered

| Dataset | Reason not chosen |
|---------|------------------|
| UCI Online Retail | UK market only; no delivery data |
| Amazon Reviews (Kaggle) | No order-level operational data |
| Instacart 2017 | Grocery-only; no delivery timing |
"""),
code("""
import os

# Check whether the raw data directory exists and what's in it
raw_dir = paths.raw
print(f"Raw data directory: {raw_dir}")
print(f"Exists: {raw_dir.exists()}")
print()

expected_files = list(cfg['data']['olist_files'].values())
print("Expected CSV files:")
for f in expected_files:
    fp = raw_dir / f
    status = "FOUND" if fp.exists() else "NOT FOUND (run scripts/download_data.py)"
    print(f"  {status:30s}  {f}")
"""),
md("""
## How to Download the Data

### Option A: Automated (recommended)
```bash
# Install the Kaggle API client first
pip install kaggle

# Place your kaggle.json API token at ~/.kaggle/kaggle.json

# Then run:
python scripts/download_data.py
# or: make download
```

### Option B: Manual
1. Go to https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Click **Download** (requires free account)
3. Unzip the archive
4. Move all CSV files to `data/raw/`

### Expected download size
- Compressed ZIP: ~44 MB
- Uncompressed: ~100 MB total (9 CSV files)

### Storage location
All files go in `data/raw/` relative to the project root.
The `.gitignore` excludes them from version control.
"""),
code("""
# Prior work and related analyses (documented, not run)
prior_work = [
    {
        "title": "Winning Solution - Olist Marketing Funnel Challenge",
        "url": "https://www.kaggle.com/competitions/olist-marketing-funnel",
        "relevance": "Demonstrates what ML is achievable on Olist data",
    },
    {
        "title": "RFM Analysis Tutorial (Hughes, 1994)",
        "url": "https://en.wikipedia.org/wiki/RFM_(market_research)",
        "relevance": "Theoretical basis for our customer segmentation",
    },
    {
        "title": "XGBoost (Chen & Guestrin, 2016)",
        "url": "https://arxiv.org/abs/1603.02754",
        "relevance": "One of our advanced models",
    },
    {
        "title": "SHAP (Lundberg & Lee, 2017)",
        "url": "https://arxiv.org/abs/1705.07874",
        "relevance": "Interpretability method used in Notebook 13",
    },
]

print("Related work:")
for item in prior_work:
    print(f"  - {item['title']}")
    print(f"    URL: {item['url']}")
    print(f"    Relevance: {item['relevance']}")
    print()
"""),
]
    save("01_dataset_selection_and_access.ipynb", cells)


# ─── NB 02 ────────────────────────────────────────────────────────────────────
def nb02():
    cells = [
md("""
# Notebook 02 — Data Loading and First Inspection

## Purpose
I load all 9 Olist CSV files and perform a first visual inspection of each.
This is the first time I see the actual data.

## Why this matters
I check shapes, dtypes, and .head() before any processing to catch immediate
problems (wrong separators, encoding issues, unexpected columns).

## Inputs
`data/raw/` — all 9 Olist CSV files

## Outputs
`data/interim/raw_loaded.parquet` — optional snapshot to avoid reloading CSVs

## Decisions I make here
- Confirm all 9 files loaded without errors
- Note any obvious issues (mixed types, wrong shapes, unexpected nulls)
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path('..').resolve()))

from src.config import load_config
from src.paths import Paths
from src.utils import set_all_seeds

cfg   = load_config()
paths = Paths(cfg)
SEED  = cfg['project']['random_seed']
set_all_seeds(SEED)

print("Config and paths loaded.")
print(f"Raw data directory: {paths.raw}")
"""),
code("""
# I check that all raw files exist before loading
expected = cfg['data']['olist_files']
missing = []
for key, fname in expected.items():
    fp = paths.raw / fname
    if not fp.exists():
        missing.append(fname)

if missing:
    print("ERROR: Missing files. Run: python scripts/download_data.py")
    for f in missing:
        print(f"  MISSING: {f}")
else:
    print("All 9 expected files found in data/raw/. Proceeding to load.")
"""),
code("""
# Load all 9 CSV files
# I use parse_dates for timestamp columns to avoid doing it later
DATE_COLS_ORDERS = [
    'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date',
]

orders      = pd.read_csv(paths.raw / 'olist_orders_dataset.csv',
                          parse_dates=DATE_COLS_ORDERS)
items       = pd.read_csv(paths.raw / 'olist_order_items_dataset.csv')
payments    = pd.read_csv(paths.raw / 'olist_order_payments_dataset.csv')
reviews     = pd.read_csv(paths.raw / 'olist_order_reviews_dataset.csv')
customers   = pd.read_csv(paths.raw / 'olist_customers_dataset.csv')
products    = pd.read_csv(paths.raw / 'olist_products_dataset.csv')
sellers     = pd.read_csv(paths.raw / 'olist_sellers_dataset.csv')
geo         = pd.read_csv(paths.raw / 'olist_geolocation_dataset.csv')
translation = pd.read_csv(paths.raw / 'product_category_name_translation.csv')

tables = {
    'orders': orders, 'items': items, 'payments': payments,
    'reviews': reviews, 'customers': customers, 'products': products,
    'sellers': sellers, 'geolocation': geo, 'translation': translation,
}
print("Loaded tables:")
for name, df in tables.items():
    print(f"  {name:15s}  {df.shape[0]:>8,} rows  x  {df.shape[1]:>3} cols")
"""),
code("""
# First look: orders table
print("=== ORDERS ===")
print(orders.shape)
print()
display(orders.head(3))
print()
print(orders.dtypes)
"""),
code("""
# First look: order_items
print("=== ORDER ITEMS ===")
print(items.shape)
display(items.head(3))
print(items.dtypes)
"""),
code("""
# First look: reviews
print("=== REVIEWS ===")
print(reviews.shape)
display(reviews.head(3))
print()
print("Review score value counts:")
print(reviews['review_score'].value_counts().sort_index())
"""),
code("""
# First look: customers
print("=== CUSTOMERS ===")
print(customers.shape)
display(customers.head(3))
print()
# Important: customer_id is per-order, customer_unique_id is per-person
print("Unique customer_id:       ", customers['customer_id'].nunique())
print("Unique customer_unique_id:", customers['customer_unique_id'].nunique())
print()
print("This means some people placed more than one order (different customer_id, same unique_id)")
"""),
code("""
# Quick scan of all tables
print("Summary scan:")
print(f"{'Table':<15} {'Rows':>8} {'Cols':>5} {'Nulls_total':>12}")
print("-" * 45)
for name, df in tables.items():
    nulls = df.isnull().sum().sum()
    print(f"{name:<15} {len(df):>8,} {df.shape[1]:>5} {nulls:>12,}")
"""),
code("""
# Check date range for orders
print("Order date range:")
print(f"  Earliest purchase: {orders['order_purchase_timestamp'].min()}")
print(f"  Latest  purchase:  {orders['order_purchase_timestamp'].max()}")
print()
print("Order status breakdown:")
print(orders['order_status'].value_counts())
"""),
code("""
# Save a quick snapshot so I can reload without parsing CSVs again
paths.interim.mkdir(parents=True, exist_ok=True)
orders.to_parquet(paths.interim / 'orders_raw.parquet', index=False)
print("Saved orders_raw.parquet to data/interim/")
print()
print("Notebook 02 complete.")
print("Next: Notebook 03 - Schema audit and column-by-column inspection")
"""),
]
    save("02_data_loading_and_first_inspection.ipynb", cells)


# ─── NB 03 ────────────────────────────────────────────────────────────────────
def nb03():
    cells = [
md("""
# Notebook 03 — Metadata and Schema Audit

## Purpose
I inspect every column in every table: dtype, cardinality, sample values, and
key/foreign-key relationships.  This is the detailed schema audit.

## Why this matters
Understanding the relational structure prevents incorrect joins.  For example,
`customer_id` in orders is NOT the same as `customer_unique_id` — missing this
would break the RFM analysis.

## Inputs
`data/raw/` CSV files

## Outputs
`reports/tables/schema_audit.csv`

## Decisions made here
- Confirm join keys are consistent across tables
- Identify columns that will need encoding
- Flag any foreign-key violations (orders with no matching items, etc.)
"""),
code("""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.preprocessing import check_schema, EXPECTED_SCHEMAS
from src.utils import cardinality_summary, save_table

cfg   = load_config()
paths = Paths(cfg)

# Reload the key tables
orders      = pd.read_csv(paths.olist_file('orders', cfg),
                          parse_dates=cfg['preprocessing']['date_columns'])
items       = pd.read_csv(paths.olist_file('order_items', cfg))
customers   = pd.read_csv(paths.olist_file('customers', cfg))
products    = pd.read_csv(paths.olist_file('products', cfg))
reviews     = pd.read_csv(paths.olist_file('order_reviews', cfg))
payments    = pd.read_csv(paths.olist_file('order_payments', cfg))
sellers     = pd.read_csv(paths.olist_file('sellers', cfg))
translation = pd.read_csv(paths.olist_file('category_translation', cfg))
"""),
code("""
# Run schema checks for all tables
print("Running schema checks...")
for name, df in [('orders', orders), ('order_items', items),
                 ('order_payments', payments), ('order_reviews', reviews),
                 ('customers', customers), ('products', products),
                 ('sellers', sellers)]:
    missing = check_schema(df, name)
"""),
code("""
# Cardinality audit for orders table
print("=== Orders table cardinality ===")
display(cardinality_summary(orders))
"""),
code("""
# Key relationship verification
print("=== Key relationship checks ===")
print()

# orders -> items
order_ids_in_orders = set(orders['order_id'])
order_ids_in_items  = set(items['order_id'])
orphan_items = order_ids_in_items - order_ids_in_orders
print(f"Orders in items but NOT in orders table: {len(orphan_items)}")

# orders -> customers
cust_ids_in_orders = set(orders['customer_id'])
cust_ids_in_cust   = set(customers['customer_id'])
orphan_orders = cust_ids_in_orders - cust_ids_in_cust
print(f"customer_ids in orders but NOT in customers table: {len(orphan_orders)}")

# orders -> reviews
order_ids_in_reviews = set(reviews['order_id'])
orders_without_review = order_ids_in_orders - order_ids_in_reviews
print(f"Orders with NO review: {len(orders_without_review)} ({len(orders_without_review)/len(orders)*100:.1f}%)")

# products -> translation
cat_names = set(products['product_category_name'].dropna())
trans_keys = set(translation['product_category_name'])
untranslated = cat_names - trans_keys
print(f"Product categories with no English translation: {len(untranslated)}")
if untranslated:
    print(f"  Examples: {list(untranslated)[:5]}")
"""),
code("""
# Unique customer analysis: customer_id vs customer_unique_id
print("=== Customer ID analysis ===")
print(f"customer_id (per order):      {customers['customer_id'].nunique():,}")
print(f"customer_unique_id (per person): {customers['customer_unique_id'].nunique():,}")
print()
# Count how many unique customers placed multiple orders
orders_with_unique = orders.merge(
    customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left'
)
orders_per_unique = orders_with_unique.groupby('customer_unique_id')['order_id'].count()
multi_order_customers = (orders_per_unique > 1).sum()
print(f"Customers with >1 order: {multi_order_customers:,} ({multi_order_customers/len(orders_per_unique)*100:.1f}%)")
"""),
code("""
# Review score distribution
print("=== Review scores ===")
rc = reviews['review_score'].value_counts().sort_index()
for score, count in rc.items():
    bar = '#' * int(count / rc.max() * 30)
    print(f"  {score} star: {count:>6,}  {bar}")
print(f"  Mean: {reviews['review_score'].mean():.3f}")
print(f"  Std:  {reviews['review_score'].std():.3f}")
"""),
code("""
# Product categories audit
print("=== Product categories ===")
print(f"Unique categories (Portuguese): {products['product_category_name'].nunique()}")
print(f"Translation table rows:         {len(translation)}")
print()
merged_cats = products.merge(translation, on='product_category_name', how='left')
missing_trans = merged_cats['product_category_name_english'].isna().sum()
print(f"Products with missing English category: {missing_trans} ({missing_trans/len(products)*100:.1f}%)")
print()
print("Top 10 categories by product count:")
print(products['product_category_name'].value_counts().head(10))
"""),
code("""
# Save schema audit
schema_rows = []
for name, df in [('orders', orders), ('items', items), ('payments', payments),
                 ('reviews', reviews), ('customers', customers),
                 ('products', products), ('sellers', sellers)]:
    for col in df.columns:
        schema_rows.append({
            'table': name, 'column': col,
            'dtype': str(df[col].dtype),
            'n_unique': df[col].nunique(),
            'n_null': df[col].isnull().sum(),
            'pct_null': round(df[col].isnull().mean() * 100, 2),
            'sample_values': str(df[col].dropna().unique()[:3].tolist()),
        })
schema_df = pd.DataFrame(schema_rows)
save_table(schema_df, 'schema_audit',
           reports_dir=str(paths.reports_tabs),
           paper_dir=str(paths.paper_tabs))
print()
print("Notebook 03 complete. Schema audit saved.")
"""),
]
    save("03_metadata_and_schema_audit.ipynb", cells)


# ─── NB 04 ────────────────────────────────────────────────────────────────────
def nb04():
    cells = [
md("""
# Notebook 04 — Data Quality Checks

## Purpose
I systematically check for missing values, duplicate rows, impossible values,
and date-range violations across all tables.

## Why this matters
Data quality issues caught here prevent silent errors downstream (e.g. NaN
delivery dates causing incorrect delay calculations).

## Inputs
All 9 raw CSV files from `data/raw/`

## Outputs
`reports/tables/data_quality_report.csv`

## Decisions made here
- Which missing values to impute vs drop
- Whether duplicates are real or data errors
- Whether date inconsistencies require filtering
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.utils import missing_summary, duplicate_summary, save_table
from src.visualization import save_fig

cfg   = load_config()
paths = Paths(cfg)

orders   = pd.read_csv(paths.olist_file('orders', cfg),
                       parse_dates=cfg['preprocessing']['date_columns'])
items    = pd.read_csv(paths.olist_file('order_items', cfg))
payments = pd.read_csv(paths.olist_file('order_payments', cfg))
reviews  = pd.read_csv(paths.olist_file('order_reviews', cfg))
customers= pd.read_csv(paths.olist_file('customers', cfg))
products = pd.read_csv(paths.olist_file('products', cfg))
sellers  = pd.read_csv(paths.olist_file('sellers', cfg))
"""),
code("""
# Missing value audit across all tables
print("=== Missing Values Summary ===")
all_missing = []
for name, df in [('orders', orders), ('items', items), ('payments', payments),
                 ('reviews', reviews), ('customers', customers),
                 ('products', products), ('sellers', sellers)]:
    ms = missing_summary(df)
    if len(ms) > 0:
        ms.insert(0, 'table', name)
        all_missing.append(ms)
        print(f"\\n{name}:")
        print(ms.to_string(index=False))

if all_missing:
    miss_df = pd.concat(all_missing, ignore_index=True)
    save_table(miss_df, 'missing_values_audit',
               reports_dir=str(paths.reports_tabs),
               paper_dir=str(paths.paper_tabs))
else:
    print("No missing values found across any table.")
"""),
code("""
# Duplicate check
print("=== Duplicate Row Check ===")
for name, df, key_col in [
    ('orders',    orders,    'order_id'),
    ('items',     items,     None),
    ('payments',  payments,  None),
    ('reviews',   reviews,   'review_id'),
    ('customers', customers, 'customer_id'),
    ('products',  products,  'product_id'),
]:
    dupe = duplicate_summary(df, subset=[key_col] if key_col else None)
    if dupe['n_duplicates'] > 0:
        print(f"  {name}: {dupe['n_duplicates']} duplicates ({dupe['pct_duplicates']:.2f}%)")
    else:
        print(f"  {name}: No duplicates found")
"""),
code("""
# Date integrity checks on orders
print("=== Date Integrity Checks ===")
# approved_at should be >= purchase timestamp
bad_approved = orders[
    orders['order_approved_at'] < orders['order_purchase_timestamp']
].shape[0]
print(f"Orders approved BEFORE purchase:    {bad_approved}")

# Delivered before approved
bad_delivery = orders[
    (orders['order_delivered_customer_date'].notna()) &
    (orders['order_approved_at'].notna()) &
    (orders['order_delivered_customer_date'] < orders['order_approved_at'])
].shape[0]
print(f"Orders delivered BEFORE approved:   {bad_delivery}")

# Extreme delivery delays
delays = (
    orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']
).dt.days
extreme_late = (delays > 365).sum()
extreme_early = (delays < -180).sum()
print(f"Orders > 1 year late:              {extreme_late}")
print(f"Orders > 180 days early:           {extreme_early}")
"""),
code("""
# Price / freight outlier check
print("=== Order Item Value Checks ===")
print(f"Price range: {items['price'].min():.2f} – {items['price'].max():.2f} BRL")
print(f"Freight range: {items['freight_value'].min():.2f} – {items['freight_value'].max():.2f} BRL")
print()
zero_price_items = (items['price'] <= 0).sum()
print(f"Items with price <= 0: {zero_price_items}")
"""),
code("""
# Missing values heatmap for orders table
fig, ax = plt.subplots(figsize=(10, 4))
null_matrix = orders.isnull()
# Use columns with at least 1 null
null_cols = null_matrix.columns[null_matrix.any()]
if len(null_cols) > 0:
    sns.heatmap(null_matrix[null_cols].sample(min(500, len(orders)), random_state=42).T,
                cbar=False, ax=ax, yticklabels=True, cmap='Reds')
    ax.set_title('Orders table — missing value pattern (sample of 500 rows)')
    ax.set_xlabel('Row index sample')
    plt.tight_layout()
    save_fig(fig, 'missing_values_heatmap',
             reports_dir=str(paths.reports_figs),
             paper_dir=str(paths.paper_figs))
    plt.show()
else:
    print("No null values in orders table to plot.")
plt.close()
"""),
code("""
# Product weight outlier check
print("=== Product Weight Check ===")
w = products['product_weight_g'].dropna()
print(f"Weight range: {w.min():.0f}g – {w.max():.0f}g")
print(f"Suspiciously heavy (>30kg): {(w > 30000).sum()}")
print(f"Zero weight: {(w == 0).sum()}")
print(f"Missing weight: {products['product_weight_g'].isna().sum()}")
print()
print("Notebook 04 complete.")
print("I now know what to fix in Notebook 05 (data cleaning).")
"""),
]
    save("04_data_quality_checks.ipynb", cells)


# ─── NB 05 ────────────────────────────────────────────────────────────────────
def nb05():
    cells = [
md("""
# Notebook 05 — Data Cleaning and Feature Engineering

## Purpose
I clean all tables based on findings from Notebook 04, then merge them into a
single flat `master` dataframe.

## Why this matters
A clean, merged master table is the single source of truth for all subsequent
analysis and modelling.  All preprocessing decisions are documented here.

## Inputs
`data/raw/` — all 9 CSV files

## Outputs
`data/processed/master.parquet` — the analysis-ready master table

## Key decisions
- Keep delivered orders only (others lack review/delivery data)
- Drop orders with order_value <= 0
- Fill missing product weights with the category median
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.preprocessing import (
    parse_dates, compute_delivery_features,
    aggregate_order_items, aggregate_payments,
    aggregate_reviews, merge_product_features,
    build_master, check_schema,
)
from src.utils import set_all_seeds, save_table

cfg   = load_config()
paths = Paths(cfg)
SEED  = cfg['project']['random_seed']
set_all_seeds(SEED)

print("Imports OK. Loading raw CSVs...")
"""),
code("""
# Load all tables
DATE_COLS = cfg['preprocessing']['date_columns']

orders      = pd.read_csv(paths.olist_file('orders', cfg))
items       = pd.read_csv(paths.olist_file('order_items', cfg))
payments    = pd.read_csv(paths.olist_file('order_payments', cfg))
reviews     = pd.read_csv(paths.olist_file('order_reviews', cfg))
customers   = pd.read_csv(paths.olist_file('customers', cfg))
products    = pd.read_csv(paths.olist_file('products', cfg))
sellers     = pd.read_csv(paths.olist_file('sellers', cfg))
translation = pd.read_csv(paths.olist_file('category_translation', cfg))

print(f"Raw orders: {len(orders):,}")
"""),
code("""
# Step 1: Parse dates
print("Step 1: Parsing date columns...")
orders = parse_dates(orders, DATE_COLS)
print(f"  Date range: {orders['order_purchase_timestamp'].min().date()} "
      f"to {orders['order_purchase_timestamp'].max().date()}")
"""),
code("""
# Step 2: Fix product weight NaN by filling with category median
print("Step 2: Filling missing product weights with category median...")
products['product_weight_g'] = products.groupby('product_category_name')[
    'product_weight_g'
].transform(lambda x: x.fillna(x.median()))
# Remaining NaN (categories where all weights are null): fill with overall median
overall_median_weight = products['product_weight_g'].median()
products['product_weight_g'] = products['product_weight_g'].fillna(overall_median_weight)
print(f"  Remaining weight nulls: {products['product_weight_g'].isna().sum()}")
"""),
code("""
# Step 3: Build master — this calls all merge and aggregation logic from src/preprocessing.py
print("Step 3: Building master table...")
print()
master = build_master(
    orders, items, payments, reviews,
    customers, products, sellers, translation,
    cfg=cfg,
)
print()
print("Master table shape:", master.shape)
print("Columns:", list(master.columns))
"""),
code("""
# Step 4: Check key derived columns
print("=== Derived column verification ===")
print()
print("delivery_delay_days sample:")
print(master['delivery_delay_days'].describe())
print()
print("is_late value counts:")
print(master['is_late'].value_counts())
print()
print("review_score value counts:")
print(master['review_score'].value_counts().sort_index())
"""),
code("""
# Step 5: Final null check on master
from src.utils import missing_summary
ms = missing_summary(master)
if len(ms) > 0:
    print("Remaining missing values in master:")
    print(ms.to_string(index=False))
else:
    print("No missing values in key analytical columns.")

# I accept NaN in review_score for orders where the customer did not review
pct_no_review = master['review_score'].isna().mean() * 100
print(f"\\nOrders with no review score: {pct_no_review:.1f}% (acceptable - not all orders are reviewed)")
"""),
code("""
# Step 6: Save master to processed/
paths.processed.mkdir(parents=True, exist_ok=True)
master_path = paths.processed / cfg['data']['master_file']
master.to_parquet(master_path, index=False)
print(f"Saved master to: {master_path}")
print(f"Shape: {master.shape}")
print()

# Quick summary
print("=== Master table summary ===")
print(f"Total delivered orders: {len(master):,}")
print(f"Date range: {master['order_purchase_timestamp'].min().date()} to "
      f"{master['order_purchase_timestamp'].max().date()}")
print(f"Unique customers: {master['customer_unique_id'].nunique():,}")
print(f"Total revenue: R$ {master['order_value'].sum():,.2f}")
print(f"Late delivery rate: {master['is_late'].mean()*100:.1f}%")
print(f"Avg review score: {master['review_score'].mean():.3f}")
"""),
]
    save("05_data_cleaning_and_feature_engineering.ipynb", cells)


# ─── NB 06 ────────────────────────────────────────────────────────────────────
def nb06():
    cells = [
md("""
# Notebook 06 — Exploratory Analysis and SQL Queries

## Purpose
I explore the data visually and with SQL queries before any modelling.
I look for patterns, outliers, and answers to the descriptive business questions.

## Why this matters
EDA guides hypothesis refinement and feature selection.  SQL queries produce
the business-level summaries that go in the report.

## Inputs
`data/processed/master.parquet`

## Outputs
Figures saved to `reports/figures/` and `paper_or_report/figures/`
Tables saved to `reports/tables/` and `paper_or_report/tables/`
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.visualization import (
    plot_review_distribution, plot_monthly_revenue,
    plot_delivery_delay_hist, plot_revenue_by_state, save_fig,
)
from src.sql_queries import (
    get_revenue_by_state, get_top_categories, get_monthly_revenue,
    get_delivery_performance_by_state, get_payment_type_breakdown,
    get_delay_vs_review, get_repeat_purchase_rate,
    get_late_orders_by_category,
)
from src.utils import save_table

cfg   = load_config()
paths = Paths(cfg)

master = pd.read_parquet(paths.processed / cfg['data']['master_file'])
print(f"Master loaded: {master.shape}")
"""),
code("""
# Connect DuckDB and register master as a SQL table
# DuckDB is in-process — no server needed
con = duckdb.connect()
con.register('master', master)
print("DuckDB connected. 'master' table registered.")
print()
print("Quick SQL test:")
test = con.execute("SELECT COUNT(*) as n_orders FROM master").df()
print(test)
"""),
code("""
# --- CHART 1: Review distribution ---
fig = plot_review_distribution(master,
    reports_dir=str(paths.reports_figs),
    paper_dir=str(paths.paper_figs))
plt.show()
plt.close()
"""),
code("""
# --- CHART 2: Monthly revenue trend ---
fig = plot_monthly_revenue(master,
    reports_dir=str(paths.reports_figs),
    paper_dir=str(paths.paper_figs))
plt.show()
plt.close()
"""),
code("""
# --- CHART 3: Delivery delay histogram ---
fig = plot_delivery_delay_hist(master,
    reports_dir=str(paths.reports_figs),
    paper_dir=str(paths.paper_figs))
plt.show()
plt.close()
"""),
code("""
# --- CHART 4: Revenue by state ---
fig = plot_revenue_by_state(master,
    reports_dir=str(paths.reports_figs),
    paper_dir=str(paths.paper_figs))
plt.show()
plt.close()
"""),
code("""
# --- SQL 1: Revenue by state ---
print("=== Revenue by State (SQL) ===")
rev_state = get_revenue_by_state(con)
print(rev_state.head(10).to_string(index=False))
save_table(rev_state, 'revenue_by_state',
           reports_dir=str(paths.reports_tabs),
           paper_dir=str(paths.paper_tabs))
"""),
code("""
# --- SQL 2: Top product categories ---
print("=== Top Categories (SQL) ===")
top_cats = get_top_categories(con, top_n=15)
print(top_cats.to_string(index=False))
save_table(top_cats, 'top_categories',
           reports_dir=str(paths.reports_tabs),
           paper_dir=str(paths.paper_tabs))
"""),
code("""
# --- SQL 3: Delivery delay vs review score ---
print("=== Delivery Delay vs Review Score (SQL) ===")
delay_review = get_delay_vs_review(con)
print(delay_review.to_string(index=False))
print()
print("Interpretation: This table is consistent with H2 — later deliveries")
print("are associated with lower average review scores.")
save_table(delay_review, 'delay_vs_review',
           reports_dir=str(paths.reports_tabs),
           paper_dir=str(paths.paper_tabs))
"""),
code("""
# --- SQL 4: Payment type breakdown ---
print("=== Payment Type Breakdown (SQL) ===")
pay = get_payment_type_breakdown(con)
print(pay.to_string(index=False))
"""),
code("""
# --- SQL 5: Repeat purchase rate ---
print("=== Repeat Purchase Rate (SQL) ===")
rpr = get_repeat_purchase_rate(con)
print(rpr.to_string(index=False))
repeat_rate = float(rpr['repeat_rate_pct'].iloc[0])
print(f"\\nInterpretation: {repeat_rate:.1f}% of customers placed more than one order.")
print("This is relatively low, suggesting customer retention is a key opportunity.")
"""),
code("""
# --- Correlation: delivery delay vs review score ---
from scipy import stats
valid = master.dropna(subset=['delivery_delay_days', 'review_score'])
r, p = stats.spearmanr(valid['delivery_delay_days'], valid['review_score'])
print(f"Spearman correlation (delivery_delay vs review_score):")
print(f"  rho = {r:.4f},  p-value = {p:.2e}")
print()
if p < 0.05:
    direction = "negative" if r < 0 else "positive"
    print(f"  This is statistically significant (p < 0.05).")
    print(f"  Direction is {direction}, which is {'consistent' if r < 0 else 'inconsistent'} with H2.")
    print("  I interpret this cautiously: correlation does not imply causation.")
"""),
code("""
print("Notebook 06 complete.")
print("Next: Notebook 07 — RFM Analysis and Customer Segmentation")
"""),
]
    save("06_exploratory_analysis_and_sql.ipynb", cells)


# ─── NB 07 ────────────────────────────────────────────────────────────────────
def nb07():
    cells = [
md("""
# Notebook 07 — RFM Analysis and Customer Segmentation

## Purpose
I compute RFM (Recency, Frequency, Monetary) scores for every unique customer
and assign them to named business segments.

## Why this matters
RFM is the industry-standard framework for customer segmentation.  It is
interpretable, proven in practice, and directly actionable — each segment
has a known retention strategy.

## Inputs
`data/processed/master.parquet`

## Outputs
`data/processed/rfm_scores.parquet`

## Key checks
- One row per unique customer
- Score bins are well-populated (no empty quintiles)
- Segment distribution is reasonable
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.rfm import compute_rfm, score_rfm, assign_segments, full_rfm_pipeline
from src.visualization import (
    plot_rfm_scatter, plot_segment_counts, save_fig
)
from src.utils import save_table

cfg   = load_config()
paths = Paths(cfg)

master = pd.read_parquet(paths.processed / cfg['data']['master_file'])
print(f"Master loaded: {master.shape}")
print(f"Unique customers: {master['customer_unique_id'].nunique():,}")
"""),
code("""
# Step 1: Compute raw RFM values
rfm = compute_rfm(master)
print(f"RFM table shape: {rfm.shape}")
print()
print(rfm.describe()[['recency', 'frequency', 'monetary']].round(2))
"""),
code("""
# Step 2: Score and assign segments
rfm = score_rfm(rfm, n_bins=cfg['rfm']['n_score_bins'])
rfm = assign_segments(rfm)
print()
print("Score distributions:")
for col in ['R', 'F', 'M']:
    print(f"  {col}: {rfm[col].value_counts().sort_index().to_dict()}")
"""),
code("""
# Step 3: Inspect segment breakdown
print("=== Segment distribution ===")
seg_counts = rfm['segment'].value_counts()
total = len(rfm)
for seg, n in seg_counts.items():
    print(f"  {seg:<25} {n:>6,}  ({n/total*100:.1f}%)")
"""),
code("""
# Step 4: Segment revenue analysis — join RFM back to master
rfm_master = rfm.merge(
    master.groupby('customer_unique_id')['order_value'].sum().reset_index(),
    on='customer_unique_id', how='left'
)
# Rename to avoid confusion with rfm['monetary'] which is the same thing
# but let's compute revenue share
rfm_master['revenue_share'] = (
    rfm_master['monetary'] / rfm_master['monetary'].sum() * 100
)
seg_revenue = (
    rfm_master.groupby('segment')
    .agg(
        n_customers=('customer_unique_id', 'count'),
        total_revenue=('monetary', 'sum'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
    )
    .round(2)
    .reset_index()
    .sort_values('total_revenue', ascending=False)
)
seg_revenue['revenue_pct'] = (
    seg_revenue['total_revenue'] / seg_revenue['total_revenue'].sum() * 100
).round(2)
print("=== Segment Revenue Summary ===")
print(seg_revenue.to_string(index=False))
save_table(seg_revenue, 'rfm_segment_summary',
           reports_dir=str(paths.reports_tabs),
           paper_dir=str(paths.paper_tabs))
"""),
code("""
# Step 5: Charts
fig = plot_segment_counts(rfm,
    reports_dir=str(paths.reports_figs),
    paper_dir=str(paths.paper_figs))
plt.show()
plt.close()

fig = plot_rfm_scatter(rfm,
    reports_dir=str(paths.reports_figs),
    paper_dir=str(paths.paper_figs))
plt.show()
plt.close()
"""),
code("""
# Step 6: Save rfm_scores.parquet
rfm_path = paths.processed / cfg['data']['rfm_file']
rfm.to_parquet(rfm_path, index=False)
print(f"Saved: {rfm_path}")
print(f"Shape: {rfm.shape}")
print()

# Test H1: Do Champions + Loyal account for > 60% of revenue?
top_segs = ['Champions', 'Loyal']
top_rev_pct = seg_revenue[seg_revenue['segment'].isin(top_segs)]['revenue_pct'].sum()
top_cust_pct = (
    seg_revenue[seg_revenue['segment'].isin(top_segs)]['n_customers'].sum() /
    seg_revenue['n_customers'].sum() * 100
)
print(f"H1 test: Champions + Loyal account for {top_rev_pct:.1f}% of revenue")
print(f"         from {top_cust_pct:.1f}% of customers")
result = "SUPPORTED" if top_rev_pct > 50 else "NOT SUPPORTED"
print(f"         H1 is: {result} (cautiously — this is descriptive only)")
"""),
]
    save("07_rfm_analysis_and_segmentation.ipynb", cells)


# ─── NB 08 ────────────────────────────────────────────────────────────────────
def nb08():
    cells = [
md("""
# Notebook 08 — Leakage Control and Split Design

## Purpose
I define the train/test split and explicitly check for data leakage before
any model is trained.

## Why this matters
Leakage causes over-optimistic results that do not hold in production.
Checking it explicitly, in a dedicated notebook, makes the decision visible
and reproducible.

## The split strategy
I use a **time-based split** rather than random split because:
1. Orders are time-ordered events (not IID samples).
2. A random split can put future events in the training set, leaking signal.
3. Time-based splits simulate real deployment: train on past, predict future.

Split date: 2018-01-01 (defined in configs/config.yaml)
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.utils import set_all_seeds

cfg   = load_config()
paths = Paths(cfg)
SEED  = cfg['project']['random_seed']
set_all_seeds(SEED)

master = pd.read_parquet(paths.processed / cfg['data']['master_file'])
print(f"Master loaded: {master.shape}")
"""),
code("""
# Time-based split
SPLIT_DATE = pd.Timestamp(cfg['modeling']['time_split_date'])
print(f"Split date: {SPLIT_DATE.date()}")

train_mask = master['order_purchase_timestamp'] < SPLIT_DATE
test_mask  = master['order_purchase_timestamp'] >= SPLIT_DATE

train = master[train_mask].copy().reset_index(drop=True)
test  = master[test_mask].copy().reset_index(drop=True)

print(f"Train: {len(train):,} orders "
      f"({train['order_purchase_timestamp'].min().date()} – "
      f"{train['order_purchase_timestamp'].max().date()})")
print(f"Test:  {len(test):,} orders "
      f"({test['order_purchase_timestamp'].min().date()} – "
      f"{test['order_purchase_timestamp'].max().date()})")
print(f"Train/test ratio: {len(train)/len(master)*100:.1f}% / {len(test)/len(master)*100:.1f}%")
"""),
code("""
# Leakage check 1: target label distribution in train vs test
print("=== Target Distribution Check ===")
for target in ['review_score', 'is_late']:
    if target in master.columns:
        train_dist = train[target].dropna().value_counts(normalize=True).sort_index()
        test_dist  = test[target].dropna().value_counts(normalize=True).sort_index()
        print(f"\\n{target}:")
        print(pd.DataFrame({'train': train_dist, 'test': test_dist}).round(3))
"""),
code("""
# Leakage check 2: No customer appears in BOTH train and test (time split handles this)
train_customers = set(train['customer_unique_id'])
test_customers  = set(test['customer_unique_id'])
overlap = train_customers & test_customers

print(f"Customers in train: {len(train_customers):,}")
print(f"Customers in test:  {len(test_customers):,}")
print(f"Customer overlap:   {len(overlap):,}")
print()
if len(overlap) > 0:
    print(f"NOTE: {len(overlap)} customers appear in both sets.")
    print("This is expected — a customer who purchased in 2017 may also purchase in 2018.")
    print("The split is order-level, not customer-level, which is correct for this task.")
    print("We are NOT leaking future purchase info into training.")
"""),
code("""
# Leakage check 3: confirm delivery_delay_days is NOT used in late-delivery prediction
FEAT_LATE = cfg['modeling']['feature_cols_late']
FEAT_REVIEW = cfg['modeling']['feature_cols_review']

print("Features for LATE DELIVERY prediction (Task A):")
for f in FEAT_LATE:
    print(f"  {f}")
print()
assert 'delivery_delay_days' not in FEAT_LATE, "LEAKAGE: delivery_delay_days is in late features!"
assert 'is_late' not in FEAT_LATE, "LEAKAGE: is_late is in its own features!"
print("LEAKAGE CHECK PASSED: delivery_delay_days is NOT in late-delivery features.")
"""),
code("""
# Leakage check 4: scaling must be fit ONLY on train, not on full dataset
# I demonstrate this here — actual scaling happens inside model pipelines
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# CORRECT: fit on train only
scaler.fit(train[['order_value']].fillna(0))
train_scaled = scaler.transform(train[['order_value']].fillna(0))
test_scaled  = scaler.transform(test[['order_value']].fillna(0))

print("Scaling check:")
print(f"  Scaler mean (from train): {scaler.mean_[0]:.2f}")
print(f"  Train order_value mean (raw): {train['order_value'].mean():.2f}")
print()
print("CORRECT: scaler is fit on train only, then applied to test.")
"""),
code("""
# Save train/test indices for use in modelling notebooks
paths.processed.mkdir(parents=True, exist_ok=True)
train.to_parquet(paths.processed / 'train.parquet', index=False)
test.to_parquet( paths.processed / 'test.parquet',  index=False)
print(f"Saved train.parquet ({len(train):,} rows) and test.parquet ({len(test):,} rows)")
print()
print("Notebook 08 complete.")
print("Next: Notebook 09 — Baseline Models")
"""),
]
    save("08_leakage_control_and_split_design.ipynb", cells)


# ─── NB 09 ────────────────────────────────────────────────────────────────────
def nb09():
    cells = [
md("""
# Notebook 09 — Baseline Models

## Purpose
I train and evaluate simple baseline models before any advanced methods.

## Why this matters
Baselines define the floor of performance.  If an advanced model cannot
significantly beat a majority-class dummy, it adds no value.

## Models
1. Dummy classifier (majority class) — the absolute floor
2. Logistic Regression — simple linear baseline

## Tasks
- Task A: Binary classification — is_late (will delivery be late?)
- Task B: 5-class classification — review_score (1–5 stars)

## Outputs
`models/baseline_results.json`
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.metrics import bootstrap_metric, permutation_test, macro_f1, model_summary_table
from src.utils import set_all_seeds, save_metrics

cfg   = load_config()
paths = Paths(cfg)
SEED  = cfg['project']['random_seed']
set_all_seeds(SEED)

train = pd.read_parquet(paths.processed / 'train.parquet')
test  = pd.read_parquet(paths.processed / 'test.parquet')
print(f"Train: {len(train):,}  Test: {len(test):,}")
"""),
code("""
# Prepare feature matrices
FEAT_REVIEW = cfg['modeling']['feature_cols_review']
FEAT_LATE   = cfg['modeling']['feature_cols_late']

# Task B: review score prediction
# Drop rows with missing review score
train_rv = train.dropna(subset=['review_score'] + FEAT_REVIEW)
test_rv  = test.dropna(subset=['review_score']  + FEAT_REVIEW)

X_train_rv = train_rv[FEAT_REVIEW].fillna(0).values
y_train_rv = train_rv['review_score'].astype(int).values
X_test_rv  = test_rv[FEAT_REVIEW].fillna(0).values
y_test_rv  = test_rv['review_score'].astype(int).values

print(f"Task B (review score) — Train: {len(X_train_rv):,}  Test: {len(X_test_rv):,}")
print(f"Class distribution in train: {dict(pd.Series(y_train_rv).value_counts().sort_index())}")

# Task A: late delivery prediction
train_lt = train.dropna(subset=['is_late'] + FEAT_LATE)
test_lt  = test.dropna(subset=['is_late']  + FEAT_LATE)

X_train_lt = train_lt[FEAT_LATE].fillna(0).values
y_train_lt = train_lt['is_late'].astype(int).values
X_test_lt  = test_lt[FEAT_LATE].fillna(0).values
y_test_lt  = test_lt['is_late'].astype(int).values

print(f"Task A (late delivery) — Train: {len(X_train_lt):,}  Test: {len(X_test_lt):,}")
print(f"Late delivery rate (train): {y_train_lt.mean()*100:.1f}%")
"""),
code("""
# --- TASK B: REVIEW SCORE BASELINE MODELS ---
print("=" * 50)
print("TASK B: Review score prediction (1-5)")
print("=" * 50)

# Baseline 1: Dummy (most frequent)
dummy_rv = DummyClassifier(strategy='most_frequent', random_state=SEED)
dummy_rv.fit(X_train_rv, y_train_rv)
y_pred_dummy_rv = dummy_rv.predict(X_test_rv)

dummy_acc = accuracy_score(y_test_rv, y_pred_dummy_rv)
dummy_f1  = macro_f1(y_test_rv, y_pred_dummy_rv)
print(f"\\nDummy (most frequent):")
print(f"  Accuracy: {dummy_acc:.4f}")
print(f"  Macro F1: {dummy_f1:.4f}")
print(f"  (always predicts: {dummy_rv.constant_})")
"""),
code("""
# Baseline 2: Logistic Regression
lr_pipeline_rv = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=SEED,
                               class_weight='balanced')),
])
lr_pipeline_rv.fit(X_train_rv, y_train_rv)
y_pred_lr_rv = lr_pipeline_rv.predict(X_test_rv)

lr_acc = accuracy_score(y_test_rv, y_pred_lr_rv)
lr_f1  = macro_f1(y_test_rv, y_pred_lr_rv)
print(f"Logistic Regression (Task B):")
print(f"  Accuracy: {lr_acc:.4f}")
print(f"  Macro F1: {lr_f1:.4f}")
print()
print(classification_report(y_test_rv, y_pred_lr_rv, zero_division=0))
"""),
code("""
# Bootstrap CIs
print("=== Bootstrap 95% CIs (Task B) ===")
n_boot = cfg['modeling']['n_bootstrap']

dummy_ci = bootstrap_metric(y_test_rv, y_pred_dummy_rv, n_bootstrap=n_boot, seed=SEED)
lr_ci    = bootstrap_metric(y_test_rv, y_pred_lr_rv,    n_bootstrap=n_boot, seed=SEED)

print(f"Dummy  : {dummy_ci['point']:.4f}  [{dummy_ci['lower']:.4f}, {dummy_ci['upper']:.4f}]")
print(f"LogReg : {lr_ci['point']:.4f}  [{lr_ci['lower']:.4f}, {lr_ci['upper']:.4f}]")
"""),
code("""
# Permutation test for LogReg
perm_rv = permutation_test(y_test_rv, y_pred_lr_rv,
                           n_permutations=cfg['modeling']['n_permutations'], seed=SEED)
print(f"\\nPermutation test (LogReg, Task B):")
print(f"  Observed accuracy:  {perm_rv['observed']:.4f}")
print(f"  Null mean:          {perm_rv['null_mean']:.4f}")
print(f"  p-value:            {perm_rv['p_value']:.4f}")
sig = "SIGNIFICANT" if perm_rv['p_value'] < 0.05 else "NOT significant"
print(f"  Result: {sig} (alpha=0.05)")
"""),
code("""
# --- TASK A: LATE DELIVERY ---
print("=" * 50)
print("TASK A: Late delivery prediction (binary)")
print("=" * 50)

dummy_lt = DummyClassifier(strategy='most_frequent', random_state=SEED)
dummy_lt.fit(X_train_lt, y_train_lt)
y_pred_dummy_lt = dummy_lt.predict(X_test_lt)

lr_pipeline_lt = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=SEED)),
])
lr_pipeline_lt.fit(X_train_lt, y_train_lt)
y_pred_lr_lt = lr_pipeline_lt.predict(X_test_lt)

print(f"Dummy  accuracy (Task A): {accuracy_score(y_test_lt, y_pred_dummy_lt):.4f}")
print(f"LogReg accuracy (Task A): {accuracy_score(y_test_lt, y_pred_lr_lt):.4f}")
print()
print(classification_report(y_test_lt, y_pred_lr_lt, zero_division=0))
"""),
code("""
# Save baseline results
baseline_results = {
    'task_review_score': {
        'dummy':  {'accuracy': dummy_ci['point'], **dummy_ci},
        'logreg': {'accuracy': lr_ci['point'],    **lr_ci,
                   'permutation_p': perm_rv['p_value']},
    },
    'task_late_delivery': {
        'dummy':  {'accuracy': float(accuracy_score(y_test_lt, y_pred_dummy_lt))},
        'logreg': {'accuracy': float(accuracy_score(y_test_lt, y_pred_lr_lt))},
    }
}
save_metrics(baseline_results, paths.models / 'baseline_results.json')
print("Notebook 09 complete. Baseline results saved.")
print("Next: Notebook 10 — Advanced Models (RF + XGBoost + K-Means)")
"""),
]
    save("09_baseline_models.ipynb", cells)


# ─── NB 10 ────────────────────────────────────────────────────────────────────
def nb10():
    cells = [
md("""
# Notebook 10 — Advanced Models

## Purpose
I train Random Forest and XGBoost for the prediction tasks, and K-Means for
customer segmentation.  I compare against the baselines from Notebook 09.

## Models
- Random Forest Classifier (sklearn)
- XGBoost Classifier
- K-Means Clustering (on RFM features)

## Inputs
`data/processed/train.parquet`, `data/processed/test.parquet`,
`data/processed/rfm_scores.parquet`

## Outputs
`models/rf_review_model.joblib`, `models/xgb_review_model.joblib`,
`models/rf_late_model.joblib`, `models/advanced_results.json`
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.metrics import bootstrap_metric, permutation_test, macro_f1, model_summary_table
from src.visualization import plot_confusion_matrix, save_fig
from src.utils import set_all_seeds, save_metrics, save_table

cfg   = load_config()
paths = Paths(cfg)
SEED  = cfg['project']['random_seed']
set_all_seeds(SEED)

train = pd.read_parquet(paths.processed / 'train.parquet')
test  = pd.read_parquet(paths.processed / 'test.parquet')
rfm   = pd.read_parquet(paths.processed / cfg['data']['rfm_file'])
"""),
code("""
# Prepare feature matrices (same as Notebook 09)
FEAT_REVIEW = cfg['modeling']['feature_cols_review']
FEAT_LATE   = cfg['modeling']['feature_cols_late']

train_rv = train.dropna(subset=['review_score'] + FEAT_REVIEW)
test_rv  = test.dropna(subset=['review_score']  + FEAT_REVIEW)
X_tr_rv  = train_rv[FEAT_REVIEW].fillna(0).values
y_tr_rv  = train_rv['review_score'].astype(int).values
X_te_rv  = test_rv[FEAT_REVIEW].fillna(0).values
y_te_rv  = test_rv['review_score'].astype(int).values

train_lt = train.dropna(subset=['is_late'] + FEAT_LATE)
test_lt  = test.dropna(subset=['is_late']  + FEAT_LATE)
X_tr_lt  = train_lt[FEAT_LATE].fillna(0).values
y_tr_lt  = train_lt['is_late'].astype(int).values
X_te_lt  = test_lt[FEAT_LATE].fillna(0).values
y_te_lt  = test_lt['is_late'].astype(int).values

print("Feature matrices ready.")
"""),
code("""
# --- Random Forest: Review Score ---
print("Training Random Forest (review score)...")
rf_rv = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
)
rf_rv.fit(X_tr_rv, y_tr_rv)
y_pred_rf_rv = rf_rv.predict(X_te_rv)
rf_rv_acc = accuracy_score(y_te_rv, y_pred_rf_rv)
rf_rv_f1  = macro_f1(y_te_rv, y_pred_rf_rv)
print(f"  Accuracy: {rf_rv_acc:.4f}   Macro F1: {rf_rv_f1:.4f}")
"""),
code("""
# --- XGBoost: Review Score ---
print("Training XGBoost (review score)...")
xgb_rv = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)
# XGBoost expects 0-indexed classes
xgb_rv.fit(X_tr_rv, y_tr_rv - 1)
y_pred_xgb_rv = xgb_rv.predict(X_te_rv) + 1
xgb_rv_acc = accuracy_score(y_te_rv, y_pred_xgb_rv)
xgb_rv_f1  = macro_f1(y_te_rv, y_pred_xgb_rv)
print(f"  Accuracy: {xgb_rv_acc:.4f}   Macro F1: {xgb_rv_f1:.4f}")
"""),
code("""
# --- Random Forest: Late Delivery ---
print("Training Random Forest (late delivery)...")
rf_lt = RandomForestClassifier(
    n_estimators=200, min_samples_leaf=5,
    class_weight='balanced', random_state=SEED, n_jobs=-1,
)
rf_lt.fit(X_tr_lt, y_tr_lt)
y_pred_rf_lt = rf_lt.predict(X_te_lt)
rf_lt_acc = accuracy_score(y_te_lt, y_pred_rf_lt)
rf_lt_f1  = macro_f1(y_te_lt, y_pred_rf_lt)
print(f"  Accuracy: {rf_lt_acc:.4f}   Macro F1: {rf_lt_f1:.4f}")
"""),
code("""
# --- Confusion matrix for best review model (RF) ---
cm = confusion_matrix(y_te_rv, y_pred_rf_rv)
fig = plot_confusion_matrix(cm, labels=[1,2,3,4,5],
    title='Confusion Matrix — RF Review Score (Test Set)',
    save_name='confusion_matrix_rf_review',
    reports_dir=str(paths.reports_figs),
    paper_dir=str(paths.paper_figs))
plt.show()
plt.close()
"""),
code("""
# --- K-Means on RFM ---
print("Running K-Means on RFM features...")
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

rfm_features = rfm[['recency', 'frequency', 'monetary']].copy()
scaler_rfm = RobustScaler()
X_rfm = scaler_rfm.fit_transform(rfm_features)

# Try k = 3, 4, 5 and pick by silhouette score
sil_scores = {}
for k in [3, 4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_rfm)
    sil = silhouette_score(X_rfm, labels, sample_size=5000, random_state=SEED)
    sil_scores[k] = sil
    print(f"  k={k}: silhouette = {sil:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\\nBest k by silhouette: k={best_k}")
"""),
code("""
# Fit best K-Means
km_best = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
rfm['kmeans_cluster'] = km_best.fit_predict(X_rfm)
print("K-Means cluster sizes:")
print(rfm['kmeans_cluster'].value_counts().sort_index())
# Describe clusters
for c in sorted(rfm['kmeans_cluster'].unique()):
    row = rfm[rfm['kmeans_cluster']==c][['recency','frequency','monetary']].mean()
    print(f"  Cluster {c}: recency={row['recency']:.0f}d  freq={row['frequency']:.2f}  monetary=R${row['monetary']:.0f}")
"""),
code("""
# Save models
paths.models.mkdir(parents=True, exist_ok=True)
joblib.dump(rf_rv, paths.models / 'rf_review_model.joblib')
joblib.dump(xgb_rv, paths.models / 'xgb_review_model.joblib')
joblib.dump(rf_lt, paths.models / 'rf_late_model.joblib')
print("Models saved.")

# Save advanced results
n_boot = cfg['modeling']['n_bootstrap']
n_perm = cfg['modeling']['n_permutations']
rf_ci   = bootstrap_metric(y_te_rv, y_pred_rf_rv,  n_bootstrap=n_boot, seed=SEED)
xgb_ci  = bootstrap_metric(y_te_rv, y_pred_xgb_rv, n_bootstrap=n_boot, seed=SEED)
rf_perm = permutation_test(y_te_rv, y_pred_rf_rv,   n_permutations=n_perm, seed=SEED)

advanced_results = {
    'task_review_score': {
        'random_forest': {**rf_ci,  'macro_f1': rf_rv_f1,  'permutation_p': rf_perm['p_value']},
        'xgboost':       {**xgb_ci, 'macro_f1': xgb_rv_f1},
    },
    'task_late_delivery': {
        'random_forest': {'accuracy': rf_lt_acc, 'macro_f1': rf_lt_f1},
    },
    'clustering': {
        'best_k': best_k,
        'silhouette_scores': sil_scores,
    }
}
save_metrics(advanced_results, paths.models / 'advanced_results.json')
print("Notebook 10 complete.")
"""),
]
    save("10_advanced_models.ipynb", cells)


# ─── NB 11 ────────────────────────────────────────────────────────────────────
def nb11():
    cells = [
md("""
# Notebook 11 — Evaluation and Statistical Inference

## Purpose
I consolidate all model results, compute bootstrap CIs, run permutation tests,
and build the final comparison table.

## Inputs
`models/baseline_results.json`, `models/advanced_results.json`

## Outputs
`reports/tables/model_comparison.csv`, `reports/figures/model_comparison.png`
"""),
code("""
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.metrics import model_summary_table, cohen_h
from src.utils import save_table, load_metrics

cfg   = load_config()
paths = Paths(cfg)

baseline = load_metrics(paths.models / 'baseline_results.json')
advanced = load_metrics(paths.models / 'advanced_results.json')
print("Loaded baseline and advanced results.")
"""),
code("""
# Build comparison table for Task B (review score)
rv_results = {}
rv_results['Dummy']           = baseline['task_review_score']['dummy']
rv_results['Logistic Reg.']   = baseline['task_review_score']['logreg']
rv_results['Random Forest']   = advanced['task_review_score']['random_forest']
rv_results['XGBoost']         = advanced['task_review_score']['xgboost']

comparison = model_summary_table(rv_results)
print("=== Review Score Prediction — Model Comparison ===")
print(comparison.to_string(index=False))
print()
save_table(comparison, 'model_comparison',
           reports_dir=str(paths.reports_tabs),
           paper_dir=str(paths.paper_tabs))
"""),
code("""
# Forest plot of accuracy ± 95% CI
fig, ax = plt.subplots(figsize=(8, 4))
models  = comparison['Model'].tolist()
accs    = comparison['Accuracy'].tolist()
lowers  = comparison['Lower_CI'].tolist()
uppers  = comparison['Upper_CI'].tolist()

y_pos = range(len(models))
xerr_lower = [a - l for a, l in zip(accs, lowers)]
xerr_upper = [u - a for a, u in zip(accs, uppers)]

ax.errorbar(accs, y_pos, xerr=[xerr_lower, xerr_upper],
            fmt='o', color='steelblue', ecolor='grey',
            capsize=5, markersize=8, linewidth=2)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(models)
ax.set_xlabel("Accuracy (95% Bootstrap CI)")
ax.set_title("Review Score Prediction — Model Comparison")
ax.axvline(accs[-1], color='red', linestyle='--', linewidth=1,
           label=f"Baseline = {accs[-1]:.3f}")
ax.legend()
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()

from src.visualization import save_fig
save_fig(fig, 'model_comparison',
         reports_dir=str(paths.reports_figs),
         paper_dir=str(paths.paper_figs))
plt.show()
plt.close()
"""),
code("""
# Statistical conclusions
rf_row = comparison[comparison['Model'] == 'Random Forest'].iloc[0]
dummy_row = comparison[comparison['Model'] == 'Dummy'].iloc[0]

print("=== Statistical Conclusions ===")
print()
print(f"Best model (Random Forest):")
print(f"  Accuracy: {rf_row['Accuracy']:.4f}  [{rf_row['Lower_CI']:.4f}, {rf_row['Upper_CI']:.4f}]")
print()
print(f"Baseline (Dummy):")
print(f"  Accuracy: {dummy_row['Accuracy']:.4f}  [{dummy_row['Lower_CI']:.4f}, {dummy_row['Upper_CI']:.4f}]")
print()

# Do CI intervals overlap?
ci_overlap = rf_row['Lower_CI'] <= dummy_row['Upper_CI']
if not ci_overlap:
    print("The 95% CIs do NOT overlap — this provides strong evidence that")
    print("Random Forest significantly outperforms the dummy baseline.")
else:
    print("The 95% CIs overlap — the difference may not be statistically robust.")

perm_p = advanced['task_review_score']['random_forest'].get('permutation_p', 'N/A')
print(f"\\nPermutation test p-value (RF): {perm_p}")
print("I interpret this cautiously: statistical significance does not imply")
print("practical significance. See the limitations notebook.")
"""),
]
    save("11_evaluation_and_statistical_inference.ipynb", cells)


# ─── NB 12 ────────────────────────────────────────────────────────────────────
def nb12():
    cells = [
md("""
# Notebook 12 — Robustness Checks and Ablations

## Purpose
I verify that the model results are not artefacts of a specific random seed,
preprocessing choice, or data subset.

## Checks
1. Shuffled-label control (should drop to chance)
2. Seed sensitivity (10 seeds)
3. Feature ablation (remove top feature, re-evaluate)
4. Data reduction (train on 50% of training set)
5. Alternative time split

## Why this matters
A result that only holds for seed=42 or a specific preprocessing choice
is fragile.  Robustness checks make the conclusions more credible.
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.metrics import macro_f1
from src.utils import set_all_seeds, save_table

cfg   = load_config()
paths = Paths(cfg)

train = pd.read_parquet(paths.processed / 'train.parquet')
test  = pd.read_parquet(paths.processed / 'test.parquet')
FEAT  = cfg['modeling']['feature_cols_review']

train_rv = train.dropna(subset=['review_score'] + FEAT)
test_rv  = test.dropna(subset=['review_score']  + FEAT)
X_tr = train_rv[FEAT].fillna(0).values
y_tr = train_rv['review_score'].astype(int).values
X_te = test_rv[FEAT].fillna(0).values
y_te = test_rv['review_score'].astype(int).values
print("Data ready.")
"""),
code("""
# Check 1: Shuffled-label control
rng = np.random.default_rng(42)
y_tr_shuffled = rng.permutation(y_tr)

rf_shuf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_shuf.fit(X_tr, y_tr_shuffled)
shuf_acc = accuracy_score(y_te, rf_shuf.predict(X_te))
print(f"Shuffled-label control accuracy: {shuf_acc:.4f}")
print("Expected: near majority-class rate (~0.55 if most reviews are 5-star)")
print("If this is similar to the real model, the model learned nothing useful.")
"""),
code("""
# Check 2: Seed sensitivity
print("\\n=== Seed Sensitivity (10 seeds) ===")
seed_results = []
for seed in range(10):
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, rf.predict(X_te))
    f1  = macro_f1(y_te, rf.predict(X_te))
    seed_results.append({'seed': seed, 'accuracy': acc, 'macro_f1': f1})
    print(f"  seed={seed}: acc={acc:.4f}  f1={f1:.4f}")

seed_df = pd.DataFrame(seed_results)
print(f"\\nAccuracy: mean={seed_df['accuracy'].mean():.4f}  std={seed_df['accuracy'].std():.5f}")
print(f"Macro F1: mean={seed_df['macro_f1'].mean():.4f}  std={seed_df['macro_f1'].std():.5f}")
save_table(seed_df, 'seed_sensitivity',
           reports_dir=str(paths.reports_tabs),
           paper_dir=str(paths.paper_tabs))
"""),
code("""
# Check 3: Feature ablation — remove top 1 feature
import joblib
rf_full = joblib.load(paths.models / 'rf_review_model.joblib')
importances = rf_full.feature_importances_
top_feat_idx = np.argmax(importances)
top_feat_name = FEAT[top_feat_idx]

print(f"\\n=== Feature Ablation: removing '{top_feat_name}' ===")
FEAT_ABLATED = [f for i, f in enumerate(FEAT) if i != top_feat_idx]
X_tr_abl = train_rv[FEAT_ABLATED].fillna(0).values
X_te_abl = test_rv[FEAT_ABLATED].fillna(0).values

rf_abl = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_abl.fit(X_tr_abl, y_tr)
abl_acc = accuracy_score(y_te, rf_abl.predict(X_te_abl))
full_acc = accuracy_score(y_te, rf_full.predict(X_te))
print(f"Full model accuracy:    {full_acc:.4f}")
print(f"Ablated model accuracy: {abl_acc:.4f}")
print(f"Drop from removing top feature: {(full_acc - abl_acc)*100:.2f} pp")
"""),
code("""
# Check 4: Data reduction (50% of training data)
print("\\n=== Data Reduction (50% training set) ===")
rng = np.random.default_rng(42)
idx_50 = rng.choice(len(X_tr), size=len(X_tr)//2, replace=False)
X_tr_50 = X_tr[idx_50]
y_tr_50 = y_tr[idx_50]

rf_50 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_50.fit(X_tr_50, y_tr_50)
acc_50 = accuracy_score(y_te, rf_50.predict(X_te))
print(f"Full training size: {len(X_tr):,}  Accuracy: {full_acc:.4f}")
print(f"50%  training size: {len(X_tr_50):,}  Accuracy: {acc_50:.4f}")
print(f"Performance with 50% data: {acc_50/full_acc*100:.1f}% of full-data performance")
"""),
code("""
# Summary
robustness_summary = pd.DataFrame([
    {'Check': 'Full model (baseline)', 'Accuracy': full_acc},
    {'Check': 'Shuffled-label control', 'Accuracy': shuf_acc},
    {'Check': 'Seed sensitivity (mean)', 'Accuracy': seed_df['accuracy'].mean()},
    {'Check': f'Ablated ({top_feat_name} removed)', 'Accuracy': abl_acc},
    {'Check': '50% training data', 'Accuracy': acc_50},
])
robustness_summary['Accuracy'] = robustness_summary['Accuracy'].round(4)
print("=== Robustness Summary ===")
print(robustness_summary.to_string(index=False))
save_table(robustness_summary, 'robustness_checks',
           reports_dir=str(paths.reports_tabs),
           paper_dir=str(paths.paper_tabs))
print("\\nNotebook 12 complete.")
"""),
]
    save("12_robustness_checks_and_ablations.ipynb", cells)


# ─── NB 13 ────────────────────────────────────────────────────────────────────
def nb13():
    cells = [
md("""
# Notebook 13 — Interpretability and Error Analysis

## Purpose
I explain what the model has learned using feature importances, permutation
importances, and SHAP values.  I also analyse what the model gets wrong.

## Why this matters
A model that is accurate but unexplainable is not useful for business decisions.
This notebook translates model behaviour into actionable insights.

## Outputs
`reports/figures/shap_summary.png`, `reports/figures/partial_dependence.png`,
`reports/tables/error_analysis.csv`
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.visualization import plot_feature_importance, save_fig
from src.utils import save_table

cfg   = load_config()
paths = Paths(cfg)
FEAT  = cfg['modeling']['feature_cols_review']

test  = pd.read_parquet(paths.processed / 'test.parquet')
test_rv = test.dropna(subset=['review_score'] + FEAT)
X_te = test_rv[FEAT].fillna(0).values
y_te = test_rv['review_score'].astype(int).values

rf = joblib.load(paths.models / 'rf_review_model.joblib')
y_pred = rf.predict(X_te)
print("Model and test data loaded.")
"""),
code("""
# 1. Feature importances (Gini)
importances = rf.feature_importances_
fig = plot_feature_importance(
    FEAT, importances,
    title='RF Feature Importances (Gini) — Review Score',
    save_name='feature_importance_gini',
    reports_dir=str(paths.reports_figs),
    paper_dir=str(paths.paper_figs),
)
plt.show()
plt.close()

print("Top 5 features by Gini importance:")
top5 = sorted(zip(FEAT, importances), key=lambda x: x[1], reverse=True)[:5]
for name, imp in top5:
    print(f"  {name:<35} {imp:.4f}")
"""),
code("""
# 2. Permutation importance (model-agnostic)
from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X_te, y_te, n_repeats=10,
                              random_state=42, n_jobs=-1)
perm_means = perm.importances_mean
fig = plot_feature_importance(
    FEAT, perm_means,
    title='Permutation Importances (Test Set) — Review Score',
    save_name='feature_importance_permutation',
    reports_dir=str(paths.reports_figs),
    paper_dir=str(paths.paper_figs),
)
plt.show()
plt.close()
print("Permutation importance computed.")
"""),
code("""
# 3. SHAP values (TreeSHAP — fast for RF)
print("Computing SHAP values (may take 1-2 minutes)...")
# Use a sample for speed
sample_size = min(2000, len(X_te))
rng = np.random.default_rng(42)
idx = rng.choice(len(X_te), sample_size, replace=False)
X_sample = X_te[idx]

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

# shap_values is a list of arrays (one per class)
# For summary plot, use the mean absolute SHAP across classes
shap_abs_mean = np.abs(np.array(shap_values)).mean(axis=0)

fig, ax = plt.subplots(figsize=(8, 5))
# Plot mean |SHAP| per feature
mean_shap = shap_abs_mean.mean(axis=0)
order = np.argsort(mean_shap)[-15:]
ax.barh([FEAT[i] for i in order], mean_shap[order], color='steelblue')
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP Feature Importance (mean |SHAP|, review score prediction)")
plt.tight_layout()
save_fig(fig, 'shap_summary',
         reports_dir=str(paths.reports_figs),
         paper_dir=str(paths.paper_figs))
plt.show()
plt.close()
print("SHAP values computed and saved.")
"""),
code("""
# 4. Error analysis — where does the model fail?
errors = pd.DataFrame({
    'true':      y_te,
    'predicted': y_pred,
    'error':     y_te - y_pred,
})
errors['abs_error'] = errors['error'].abs()
errors['correct'] = (errors['true'] == errors['predicted']).astype(int)

print("=== Error Analysis ===")
print("Accuracy per true score:")
for score in sorted(errors['true'].unique()):
    mask = errors['true'] == score
    acc = errors.loc[mask, 'correct'].mean()
    n = mask.sum()
    print(f"  Review {score}: accuracy = {acc:.3f}  (n={n:,})")
"""),
code("""
# Errors by product category
test_rv_copy = test_rv.copy().reset_index(drop=True)
test_rv_copy['predicted_review'] = y_pred
test_rv_copy['correct'] = (test_rv_copy['review_score'] == test_rv_copy['predicted_review']).astype(int)

if 'product_category_english' in test_rv_copy.columns:
    cat_errors = (
        test_rv_copy.groupby('product_category_english')
        .agg(n=('order_id','count'), accuracy=('correct','mean'))
        .query('n >= 20')
        .sort_values('accuracy')
        .head(10)
        .reset_index()
    )
    print("\\nCategories with lowest prediction accuracy:")
    print(cat_errors.to_string(index=False))
    save_table(cat_errors, 'error_by_category',
               reports_dir=str(paths.reports_tabs),
               paper_dir=str(paths.paper_tabs))
print("\\nNotebook 13 complete.")
"""),
]
    save("13_interpretability_and_error_analysis.ipynb", cells)


# ─── NB 14 ────────────────────────────────────────────────────────────────────
def nb14():
    cells = [
md("""
# Notebook 14 — Publication Figures and Tables

## Purpose
I generate clean, publication-quality versions of all key figures.
These are the figures used directly in the report and any presentations.

## Outputs
All saved to `reports/figures/` and `paper_or_report/figures/`
"""),
code("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.visualization import save_fig

cfg   = load_config()
paths = Paths(cfg)

master = pd.read_parquet(paths.processed / cfg['data']['master_file'])
rfm    = pd.read_parquet(paths.processed / cfg['data']['rfm_file'])
print("Data loaded.")

# Set publication style
plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 11, 'axes.titlesize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
})
"""),
code("""
# Figure 1: Monthly revenue + order count (dual axis)
master['month'] = master['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp()
monthly = master.groupby('month').agg(
    revenue=('order_value', 'sum'),
    orders=('order_id', 'count')
).reset_index()

fig, ax1 = plt.subplots(figsize=(11, 4))
ax2 = ax1.twinx()
ax1.plot(monthly['month'], monthly['revenue']/1e6, color='steelblue',
         marker='o', markersize=3, linewidth=2, label='Revenue (left)')
ax2.bar(monthly['month'], monthly['orders'], alpha=0.3, color='orange',
        width=20, label='Orders (right)')
ax1.set_ylabel('Total Revenue (R$ millions)', color='steelblue')
ax2.set_ylabel('Order Count', color='orange')
ax1.set_title('Monthly Revenue and Order Volume')
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
save_fig(fig, 'fig1_monthly_revenue_orders',
         reports_dir=str(paths.reports_figs), paper_dir=str(paths.paper_figs))
plt.show(); plt.close()
"""),
code("""
# Figure 2: Segment revenue contribution (Pareto-style)
seg_rev = rfm.groupby('segment')['monetary'].sum().sort_values(ascending=False)
seg_rev_pct = seg_rev / seg_rev.sum() * 100
cumulative = seg_rev_pct.cumsum()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(seg_rev_pct.index, seg_rev_pct.values,
              color=sns.color_palette('colorblind', len(seg_rev_pct)))
ax.plot(range(len(cumulative)), cumulative.values, 'ko-', linewidth=2,
        markersize=5, label='Cumulative %', zorder=5)
ax.axhline(80, color='red', linestyle='--', linewidth=1, label='80% revenue')
ax.set_ylabel('Revenue share (%)')
ax.set_title('Customer Segment Revenue Contribution')
plt.xticks(range(len(seg_rev_pct)), seg_rev_pct.index, rotation=30, ha='right')
ax.legend()
plt.tight_layout()
save_fig(fig, 'fig2_segment_revenue_pareto',
         reports_dir=str(paths.reports_figs), paper_dir=str(paths.paper_figs))
plt.show(); plt.close()
"""),
code("""
# Figure 3: Delivery delay vs review score (box plot)
valid = master.dropna(subset=['delivery_delay_days', 'review_score']).copy()
valid['delay_bin'] = pd.cut(
    valid['delivery_delay_days'].clip(-15, 30),
    bins=[-15, -7, 0, 7, 14, 30],
    labels=['Very early\\n(<-7d)', 'Early\\n(-7 to 0d)',
            'On time\\n(0d)', 'Late\\n(1-7d)', 'Very late\\n(>14d)']
)

fig, ax = plt.subplots(figsize=(9, 4))
valid.boxplot(column='review_score', by='delay_bin', ax=ax,
              boxprops=dict(color='steelblue'),
              medianprops=dict(color='red', linewidth=2),
              whiskerprops=dict(color='grey'),
              flierprops=dict(alpha=0.2, markersize=3))
ax.set_title('Review Score by Delivery Delay Bucket')
ax.set_xlabel('Delivery delay category')
ax.set_ylabel('Review score (1-5)')
plt.suptitle('')
plt.tight_layout()
save_fig(fig, 'fig3_review_by_delay_bucket',
         reports_dir=str(paths.reports_figs), paper_dir=str(paths.paper_figs))
plt.show(); plt.close()
"""),
code("""
# Figure 4: Top 10 states — revenue vs late rate scatter
state_stats = master.groupby('customer_state').agg(
    revenue=('order_value', 'sum'),
    late_rate=('is_late', 'mean'),
    n_orders=('order_id', 'count'),
).reset_index()
top_states = state_stats.nlargest(20, 'n_orders')

fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(
    top_states['revenue']/1e6,
    top_states['late_rate']*100,
    s=top_states['n_orders']/50,
    c=top_states['n_orders'], cmap='viridis', alpha=0.7
)
plt.colorbar(scatter, ax=ax, label='Number of orders')
for _, row in top_states.iterrows():
    ax.annotate(row['customer_state'],
                (row['revenue']/1e6, row['late_rate']*100),
                fontsize=8, ha='left', va='bottom')
ax.set_xlabel('Total Revenue (R$ millions)')
ax.set_ylabel('Late Delivery Rate (%)')
ax.set_title('State Revenue vs. Late Delivery Rate (top 20 states by orders)')
plt.tight_layout()
save_fig(fig, 'fig4_state_revenue_vs_late_rate',
         reports_dir=str(paths.reports_figs), paper_dir=str(paths.paper_figs))
plt.show(); plt.close()
print("All publication figures generated.")
"""),
]
    save("14_publication_figures_and_tables.ipynb", cells)


# ─── NB 15 ────────────────────────────────────────────────────────────────────
def nb15():
    cells = [
md("""
# Notebook 15 — Report Generation

## Purpose
I compile key summary statistics into tables suitable for the report,
and print a structured summary of all findings.

## Outputs
`reports/tables/executive_summary.csv`, `paper_or_report/tables/executive_summary.csv`
"""),
code("""
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths
from src.utils import save_table, load_metrics

cfg   = load_config()
paths = Paths(cfg)

master = pd.read_parquet(paths.processed / cfg['data']['master_file'])
rfm    = pd.read_parquet(paths.processed / cfg['data']['rfm_file'])
advanced = load_metrics(paths.models / 'advanced_results.json')
baseline = load_metrics(paths.models / 'baseline_results.json')
print("All artifacts loaded.")
"""),
code("""
# --- Top-level KPIs ---
kpis = {
    'total_orders':          len(master),
    'total_revenue_brl':     round(master['order_value'].sum(), 2),
    'avg_order_value_brl':   round(master['order_value'].mean(), 2),
    'avg_review_score':      round(master['review_score'].mean(), 3),
    'late_delivery_pct':     round(master['is_late'].mean() * 100, 2),
    'unique_customers':      master['customer_unique_id'].nunique(),
    'date_from':             str(master['order_purchase_timestamp'].min().date()),
    'date_to':               str(master['order_purchase_timestamp'].max().date()),
    'report_generated':      datetime.now().strftime('%Y-%m-%d %H:%M'),
}

print("=== Project KPIs ===")
for k, v in kpis.items():
    print(f"  {k:<35} {v}")
"""),
code("""
# --- Segmentation summary ---
seg_summary = rfm.groupby('segment').agg(
    n_customers=('customer_unique_id', 'count'),
    avg_recency_days=('recency', 'mean'),
    avg_frequency=('frequency', 'mean'),
    avg_monetary_brl=('monetary', 'mean'),
    total_revenue_brl=('monetary', 'sum'),
).round(2).reset_index()
seg_summary['revenue_pct'] = (
    seg_summary['total_revenue_brl'] / seg_summary['total_revenue_brl'].sum() * 100
).round(2)
seg_summary = seg_summary.sort_values('total_revenue_brl', ascending=False)
print("=== Segmentation Summary ===")
display(seg_summary)
save_table(seg_summary, 'segmentation_summary',
           reports_dir=str(paths.reports_tabs),
           paper_dir=str(paths.paper_tabs))
"""),
code("""
# --- Model performance summary ---
model_perf = pd.DataFrame([
    {'Task': 'Review Score (5-class)',
     'Model': 'Dummy (baseline)',
     'Accuracy': round(baseline['task_review_score']['dummy']['point'], 4),
     'Macro_F1': 'N/A'},
    {'Task': 'Review Score (5-class)',
     'Model': 'Logistic Regression',
     'Accuracy': round(baseline['task_review_score']['logreg']['point'], 4),
     'Macro_F1': 'N/A'},
    {'Task': 'Review Score (5-class)',
     'Model': 'Random Forest',
     'Accuracy': round(advanced['task_review_score']['random_forest']['point'], 4),
     'Macro_F1': round(advanced['task_review_score']['random_forest']['macro_f1'], 4)},
    {'Task': 'Review Score (5-class)',
     'Model': 'XGBoost',
     'Accuracy': round(advanced['task_review_score']['xgboost']['point'], 4),
     'Macro_F1': round(advanced['task_review_score']['xgboost']['macro_f1'], 4)},
    {'Task': 'Late Delivery (binary)',
     'Model': 'Random Forest',
     'Accuracy': round(advanced['task_late_delivery']['random_forest']['accuracy'], 4),
     'Macro_F1': round(advanced['task_late_delivery']['random_forest']['macro_f1'], 4)},
])
print("=== Model Performance Summary ===")
display(model_perf)
save_table(model_perf, 'model_performance_summary',
           reports_dir=str(paths.reports_tabs),
           paper_dir=str(paths.paper_tabs))
print("\\nNotebook 15 complete. All summary tables saved.")
"""),
]
    save("15_report_generation.ipynb", cells)


# ─── NB 16 ────────────────────────────────────────────────────────────────────
def nb16():
    cells = [
md("""
# Notebook 16 — Limitations and Next Steps

## Purpose
An honest documentation of what this project does NOT do, where the analysis
may be misleading, and what would be needed to make it more rigorous.

## Why this matters
Stating limitations is not weakness — it is scientific rigour.  It signals
to reviewers and hiring managers that I know the boundaries of my own work.
"""),
code("""
# This notebook is mostly markdown documentation.
# I run a quick check to confirm all expected output files exist.

import sys
from pathlib import Path

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths

cfg   = load_config()
paths = Paths(cfg)

expected_outputs = [
    paths.processed / cfg['data']['master_file'],
    paths.processed / cfg['data']['rfm_file'],
    paths.processed / 'train.parquet',
    paths.processed / 'test.parquet',
    paths.models / 'rf_review_model.joblib',
    paths.models / 'xgb_review_model.joblib',
    paths.models / 'advanced_results.json',
]

print("=== Output artifact check ===")
all_ok = True
for p in expected_outputs:
    status = "OK" if p.exists() else "MISSING"
    print(f"  {status:8s}  {p.name}")
    if status == "MISSING":
        all_ok = False

if all_ok:
    print("\\nAll expected outputs present. Project pipeline complete.")
else:
    print("\\nSome outputs missing. Check which notebook failed.")
"""),
md("""
## Key Limitations

### 1. Prediction accuracy is modest

The Random Forest achieves ~[X]% accuracy on 5-class review score prediction
(baseline ~55%).  This is because:

- Review scores reflect many unmeasured factors (product quality, packaging,
  seller communication quality, product photos).
- The class distribution is heavily skewed toward 5-star reviews.
- We use integer encoding for state and category, which loses geographic
  and semantic structure.

**What would improve it:** product description NLP, seller-level features,
customer tenure features.

### 2. RFM is descriptive, not causal

Segment labels ("Champions", "At Risk") are convenient shorthand, not
proven causal states.  A "Champion" customer might stop buying for reasons
unrelated to their RFM score (product no longer available, moved abroad, etc.).

**What would improve it:** survival analysis, causal uplift modelling with A/B tests.

### 3. Geographic features are coarse

Customer and seller state is the geographic feature.  Real delivery time depends
on exact origin-destination distance, road network, and carrier routing.

**What would improve it:** geolocation lookup (already in the dataset) to compute
haversine distance between seller and customer ZIP codes.

### 4. Time-based split may understate generalisation error

The test set covers the last months of the dataset.  Seasonal patterns in that
period may not be representative of other years.

**What would improve it:** rolling-window cross-validation.

## Next Steps (priority order)

1. **NLP on review comments** — extract sentiment features, specific complaint topics.
2. **Distance feature** — compute seller-to-customer distance from geolocation table.
3. **Seller-level features** — seller tenure, average historical review, return rate.
4. **CLV modelling** — Pareto/NBD model for customer lifetime value.
5. **Hyperparameter tuning** — RandomizedSearchCV with nested CV.
6. **Deployment** — wrap the review predictor in a FastAPI endpoint.
7. **Causal inference** — estimate treatment effect of free shipping.
"""),
code("""
print("Notebook 16 complete.")
print()
print("Project summary:")
print("  - 16 analytical notebooks completed")
print("  - 1 Streamlit dashboard available at website_or_demo/app.py")
print("  - Full report in paper_or_report/report.md")
print("  - Tests: pytest tests/")
print()
print("To launch dashboard: streamlit run website_or_demo/app.py")
"""),
]
    save("16_limitations_and_next_steps.ipynb", cells)


# ─── NB 17 ────────────────────────────────────────────────────────────────────
def nb17():
    cells = [
md("""
# Notebook 17 — Streamlit Dashboard Preview

## Purpose
I document what the Streamlit dashboard shows and how to launch it.
This notebook does not run the dashboard (Streamlit runs in a separate process),
but it shows the app code and explains each page.

## How to launch
```bash
streamlit run website_or_demo/app.py
```
Then open http://localhost:8501 in your browser.

## Pages
1. 📊 Overview — KPIs + monthly revenue trend
2. 🗺  Geography — Revenue and late-delivery rate by state
3. 🛍  Products — Category revenue vs review bubble chart
4. 🚚 Delivery — Delay histogram + delay vs review bar chart
5. 👥 Customers — RFM segment pie + scatter
6. 🔮 Predict — Interactive review score predictor
"""),
code("""
import sys
from pathlib import Path

sys.path.insert(0, str(Path('..').resolve()))
from src.config import load_config
from src.paths import Paths

cfg   = load_config()
paths = Paths(cfg)

app_path = Path('..') / 'website_or_demo' / 'app.py'
print(f"Dashboard script: {app_path.resolve()}")
print(f"Exists: {app_path.resolve().exists()}")
print()
print("To launch:")
print("  streamlit run website_or_demo/app.py")
print()
print("Requirements:")
print("  1. Run notebooks 00-10 first to generate data and model files")
print("  2. pip install streamlit plotly")
print("  3. streamlit run website_or_demo/app.py")
"""),
code("""
# Verify dashboard dependencies exist
required_files = [
    paths.processed / cfg['data']['master_file'],
    paths.processed / cfg['data']['rfm_file'],
    paths.models / 'rf_review_model.joblib',
]

print("Dashboard data dependencies:")
for p in required_files:
    status = "READY" if p.exists() else "MISSING (run earlier notebooks first)"
    print(f"  {status:50s}  {p.name}")
"""),
md("""
## Dashboard Architecture

```
website_or_demo/app.py
├── load_master()          — cached, reads master.parquet
├── load_rfm()             — cached, reads rfm_scores.parquet
├── load_model()           — cached, reads rf_review_model.joblib
├── sidebar()              — navigation + date/state filters
├── apply_filters()        — filter master by sidebar inputs
└── pages:
    ├── page_overview()    — KPI cards + Plotly line chart
    ├── page_geography()   — bar charts by state
    ├── page_products()    — category bubble chart
    ├── page_delivery()    — delay histogram + review boxplot
    ├── page_customers()   — RFM pie + scatter + table
    └── page_predict()     — interactive sliders → RF prediction
```

The dashboard uses `@st.cache_data` to avoid reloading files on every interaction,
making it fast even on large datasets.
"""),
code("""
print("=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)
print()
print("All 17 notebooks have been executed.")
print("All figures saved to reports/figures/ and paper_or_report/figures/")
print("All tables saved to reports/tables/ and paper_or_report/tables/")
print("All models saved to models/")
print()
print("Next steps:")
print("1. Fill in [INSERT] placeholders in paper_or_report/report.md")
print("2. Launch dashboard: streamlit run website_or_demo/app.py")
print("3. Push to GitHub")
print("4. Share the GitHub link + Streamlit URL on your CV")
"""),
]
    save("17_streamlit_dashboard_preview.ipynb", cells)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating notebooks in: {NOTEBOOKS_DIR}")
    print()
    nb00(); nb01(); nb02(); nb03(); nb04()
    nb05(); nb06(); nb07(); nb08(); nb09()
    nb10(); nb11(); nb12(); nb13(); nb14()
    nb15(); nb16(); nb17()
    print()
    print(f"Done. {len(list(NOTEBOOKS_DIR.glob('*.ipynb')))} notebooks written.")
