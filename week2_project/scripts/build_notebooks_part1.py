"""
build_notebooks.py — Generate all 13 notebooks for the Diabetes Readmission project.
Run once: python scripts/build_notebooks.py
"""
import json, os

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"}
        },
        "cells": cells
    }

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

def save_nb(filename, cells):
    path = os.path.join(NB_DIR, filename)
    with open(path, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"  ✓ {filename}  ({len(cells)} cells)")


# ═════════════════════════════════════════════════════════════════════════════
# SHARED SETUP CELL — included at the top of every notebook
# ═════════════════════════════════════════════════════════════════════════════
SETUP_CODE = """\
import sys, os
# Add project root to Python path so we can import configs and src
NOTEBOOK_DIR = os.getcwd()
PROJECT_ROOT = os.path.dirname(NOTEBOOK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import (
    RAW_FILE, IDS_MAP_FILE, INTERIM_FILE, PROCESSED_FILE,
    TRAIN_FILE, VAL_FILE, TEST_FILE,
    FIGURES_DIR, TABLES_DIR, PAPER_FIG_DIR, PAPER_TAB_DIR,
    RANDOM_SEED, TARGET_COL, PATIENT_ID_COL, MEDICATION_COLS,
    AGE_ORDER, icd9_to_category, COLORS, ensure_dirs
)
from src.plot_utils import set_plot_style, save_figure, save_table
ensure_dirs()
set_plot_style()
print("Project config loaded. Random seed:", RANDOM_SEED)
"""


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 00 — Project Overview & Research Question
# ══════════════════════════════════════════════════════════════════════════════
nb00 = [
md("""# Notebook 00 — Project Overview & Research Question
**Decodelabs Data Science Internship | Week 2**

---

## What this notebook does
I use this notebook to frame the research question clearly before touching any data.
This is not just formality — having a precise question prevents me from doing analysis
that looks interesting but doesn't answer anything meaningful.

## Why this step matters
In professional data science, you work backwards from a question, not forwards from
a dataset. Defining the question first shapes every subsequent decision: which features
to engineer, which metrics to prioritise, and what "success" actually means.

## What Python / ML concepts I practise
- Structuring a research question
- Understanding what makes a good ML problem
- Reading real clinical context around a dataset

## Input files
- None (this is a framing notebook)

## Output files
- None (but this thinking informs all later notebooks)
"""),

md("""## Background: Why Hospital Readmission Matters

**Hospital readmission** occurs when a patient is discharged but returns to the hospital
within a short period. Early readmission — returning within **30 days** of discharge — is
a major healthcare quality indicator.

In the United States alone, approximately **20% of Medicare patients** are readmitted within
30 days of discharge, costing the system an estimated **$26 billion per year** in potentially
preventable costs. The Centers for Medicare & Medicaid Services (CMS) introduced financial
penalties for hospitals with high readmission rates in 2012.

For **diabetic patients** specifically, the problem is acute. Diabetes is a complex chronic
condition requiring careful medication management, diet compliance, and follow-up care. Patients
with poorly controlled diabetes are at high risk of readmission due to complications including
hypoglycemia, hyperglycemia, and related comorbidities.

**A predictive model** that flags patients at high risk of early readmission at the point of
discharge could allow hospitals to:
- Prioritise post-discharge phone follow-up for high-risk patients
- Arrange earlier outpatient appointments
- Provide more intensive discharge education
- Reduce preventable readmissions and associated costs
"""),

md("""## The Dataset

**Name:** Diabetes 130-US Hospitals for Years 1999–2008  
**Source:** UCI Machine Learning Repository (publicly available)  
**Citation:** Strack et al. (2014), *BioMed Research International*

| Property | Details |
|---|---|
| **Rows** | ~101,766 inpatient encounters |
| **Columns** | 50 features + 1 target |
| **Patients** | ~71,518 unique patients |
| **Hospitals** | 130 US hospitals |
| **Period** | 1999–2008 |
| **Target** | `readmitted`: NO / >30 days / <30 days |

This dataset is **335× larger** than the Week 1 Heart Disease dataset.
At ~102,000 rows, we have enough data to:
- Train complex models (Random Forest, XGBoost)
- Compute reliable bootstrap confidence intervals
- Perform meaningful subgroup / fairness analysis across demographic groups
- Apply SHAP interpretability at scale

### Important data note
The same patient may appear multiple times (multiple hospital encounters).
This introduces a **patient-level leakage risk**: if the same patient appears
in both train and test, the model may generalise to that patient rather than
to the general population. I address this with a **patient-aware split** in Notebook 06.
"""),

md("""## Primary Research Question

> **Can clinical and demographic features recorded at hospital admission and discharge
> predict whether a diabetic patient will be readmitted within 30 days?**

### Secondary questions
1. Which features are most predictive of early readmission?
2. Do simple models (Logistic Regression) compete with ensemble methods (XGBoost)?
3. Does model performance differ across demographic groups (race, age, gender)?
4. What types of errors does the model make, and which errors are most clinically costly?
5. Is the model reliable enough for real clinical deployment, or only for research?

### Target variable definition
- **Binary target:** `readmitted_binary`
  - `0` = Not readmitted within 30 days (includes "NO" and ">30" in original data)
  - `1` = Readmitted within 30 days (`<30` in original data)

### What counts as success?
Given the class imbalance (~11% early readmission), I will use:
- **Primary metric:** ROC-AUC (insensitive to class imbalance threshold)
- **Secondary metric:** Average Precision / PR-AUC (better for imbalanced classes)
- **Clinical metric:** Recall for class 1 (catching true readmissions is more important than precision)

A model with ROC-AUC > 0.75 and Recall (class 1) > 0.60 would be considered useful for clinical screening.

### What counts as failure?
- A model that simply predicts "no readmission" for every patient.
- A model that performs well overall but poorly for specific demographic subgroups.
- A model that leaks future information or patient-level information.
"""),

md("""## Project Structure & Notebook Workflow

| # | Notebook | Task | Description |
|---|---|---|---|
| 00 | `00_project_overview.ipynb` | Framing | Research question, background, success criteria |
| 01 | `01_data_collection.ipynb` | **Task 1** | Download, extract, load, first inspection |
| 02 | `02_data_cleaning.ipynb` | **Task 2** | Handle missing values, drop bad rows, fix types |
| 03 | `03_feature_engineering.ipynb` | Feature work | ICD-9 grouping, drug encoding, new features |
| 04 | `04_eda_univariate.ipynb` | **Task 3** | Distributions, missing patterns, class balance |
| 05 | `05_eda_bivariate.ipynb` | **Task 3** | Feature vs target, statistical tests |
| 06 | `06_leakage_and_splits.ipynb` | Leakage | Patient-aware train/val/test split, preprocessing |
| 07 | `07_baseline_models.ipynb` | **Task 5** | Dummy, Logistic Regression |
| 08 | `08_advanced_models.ipynb` | **Task 5** | Random Forest, GradientBoosting, XGBoost |
| 09 | `09_model_evaluation.ipynb` | **Task 5** | Full metrics, bootstrap CI, ROC/PR curves |
| 10 | `10_visualizations.ipynb` | **Task 4** | Publication-quality figures, storytelling charts |
| 11 | `11_error_analysis_shap.ipynb` | Interpretability | SHAP values, error analysis, failure cases |
| 12 | `12_fairness_analysis.ipynb` | Ethics | Subgroup performance, demographic disparity |

**All 5 Decodelabs tasks are completed across this pipeline.**
"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 01 — Data Collection & First Inspection (Task 1)
# ══════════════════════════════════════════════════════════════════════════════
nb01 = [
md("""# Notebook 01 — Data Collection & Dataset Understanding
**Decodelabs Internship | Week 2 | Task 1**

---
## What this notebook does
I download the UCI Diabetes 130-US Hospitals dataset from the official source,
extract it from its ZIP archive, load it into pandas, and build a thorough
understanding of every column's meaning, data type, and quality signals.

## Why this step matters
With 50 columns and over 100,000 rows, this dataset is complex enough that
skipping the inspection step will cause serious mistakes in later analysis.
Understanding what each column represents — especially the 24 medication columns
and 3 ICD-9 diagnosis columns — is essential before any cleaning or modelling.

## What Python / ML concepts I practise
- Downloading and extracting ZIP files with `requests` and `zipfile`
- Loading large CSVs efficiently with `pandas`
- Inspecting DataFrames: `.info()`, `.describe()`, `.value_counts()`, `.nunique()`
- Identifying data quality signals from raw inspection

## Input files
- None (dataset downloaded inside this notebook)

## Output files
- `data/raw/diabetic_data.csv` — original data, do not modify
- `data/raw/IDs_mapping.csv` — ID-to-label mapping file
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import zipfile
import io
import os

print("Libraries imported.")
print(f"pandas version : {pd.__version__}")
print(f"numpy  version : {np.__version__}")
"""),

md("## Step 1 — Download the dataset from UCI"),
code("""\
# The dataset is hosted as a ZIP file on the UCI ML Repository.
# I download it only if the extracted CSV doesn't already exist.
# This avoids re-downloading on every notebook run.

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"

raw_dir = os.path.dirname(RAW_FILE)

if os.path.exists(RAW_FILE):
    print(f"Data already exists: {RAW_FILE}")
    print("Skipping download.")
else:
    print("Downloading dataset from UCI ML Repository...")
    print(f"URL: {DATA_URL}")
    
    response = requests.get(DATA_URL, timeout=120)
    
    if response.status_code == 200:
        print(f"Download complete. File size: {len(response.content)/1024/1024:.1f} MB")
        
        # Extract the ZIP in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print(f"ZIP contents: {z.namelist()}")
            z.extractall(raw_dir)
        
        print(f"Extracted to: {raw_dir}")
        print(f"Files in raw/: {os.listdir(raw_dir)}")
    else:
        print(f"Download failed. HTTP status: {response.status_code}")
        print("Please download manually from:")
        print(DATA_URL)
        print("And extract to:", raw_dir)
"""),

md("## Step 2 — Load the main dataset"),
code("""\
# I load the main CSV. With 101,766 rows, this still loads in seconds on most machines.
# I use low_memory=False to avoid mixed-type warnings on columns like diag_1, diag_2, diag_3.

df_raw = pd.read_csv(RAW_FILE, low_memory=False)

print(f"Dataset loaded successfully!")
print(f"Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"Memory usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
"""),

md("## Step 3 — First look"),
code("""\
print("=== First 5 rows ===")
df_raw.head()
"""),
code("""\
print("=== Column names and dtypes ===")
print(df_raw.dtypes.to_string())
"""),
code("""\
# .info() gives the most important first-pass summary:
# column names, non-null counts (spots missing data), and dtypes.
print("=== DataFrame Info ===")
df_raw.info(verbose=True, show_counts=True)
"""),

md("## Step 4 — Column inventory and data dictionary"),
code("""\
# I build a data dictionary that describes every column.
# For a 50-column dataset, this documentation is essential.

col_info = []

for col in df_raw.columns:
    n_unique = df_raw[col].nunique()
    n_missing = (df_raw[col] == "?").sum() if df_raw[col].dtype == object else df_raw[col].isnull().sum()
    sample_values = df_raw[col].dropna().unique()[:4]
    
    col_info.append({
        "Column": col,
        "Dtype": str(df_raw[col].dtype),
        "N_Unique": n_unique,
        "N_Missing_or_?": n_missing,
        "Sample Values": str(list(sample_values))[:60]
    })

col_df = pd.DataFrame(col_info)
print(f"Total columns: {len(col_df)}")
col_df
"""),

md("## Step 5 — Understand the '?' problem"),
code("""\
# The dataset uses '?' to represent missing values in object (string) columns.
# This is different from NaN — pandas does NOT automatically recognise '?' as missing.
# I need to handle this explicitly in the cleaning step (Notebook 02).

print("=== Columns with '?' values ===")
question_counts = {}
for col in df_raw.select_dtypes(include="object").columns:
    n = (df_raw[col] == "?").sum()
    if n > 0:
        question_counts[col] = n

for col, n in sorted(question_counts.items(), key=lambda x: -x[1]):
    pct = n / len(df_raw) * 100
    print(f"  {col:30s}: {n:6,} ({pct:.1f}%)")

print(f"\\nTotal '?' values across all columns: {sum(question_counts.values()):,}")
"""),

md("## Step 6 — Inspect the target variable"),
code("""\
# The target is 'readmitted' with three values: NO, >30, <30
# I check the distribution before any cleaning or binarisation.

print("=== Target: readmitted ===")
target_counts = df_raw["readmitted"].value_counts()
target_pcts   = df_raw["readmitted"].value_counts(normalize=True) * 100

for val in target_counts.index:
    print(f"  {val:5s} : {target_counts[val]:7,} ({target_pcts[val]:5.1f}%)")

print()
print("Binary interpretation:")
early = (df_raw["readmitted"] == "<30").sum()
other = (df_raw["readmitted"] != "<30").sum()
print(f"  Early readmission (<30 days)  : {early:7,} ({early/len(df_raw)*100:.1f}%)")
print(f"  No early readmission (NO/>30) : {other:7,} ({other/len(df_raw)*100:.1f}%)")
print()
print("Note: ~11% early readmission rate — this is a CLASS IMBALANCE problem.")
print("I will need to handle this carefully in modelling (Notebook 07-08).")
"""),

md("## Step 7 — Check unique patient count vs row count"),
code("""\
# The dataset has multiple encounters per patient.
# This means the same patient can appear in both training and test sets
# if I split randomly by row — a form of data leakage.

n_rows     = len(df_raw)
n_patients = df_raw["patient_nbr"].nunique()

print(f"Total encounters (rows)       : {n_rows:,}")
print(f"Unique patients               : {n_patients:,}")
print(f"Average encounters per patient: {n_rows/n_patients:.2f}")
print()

# How many patients have multiple encounters?
encounter_counts = df_raw.groupby("patient_nbr").size()
multi = (encounter_counts > 1).sum()
print(f"Patients with > 1 encounter   : {multi:,} ({multi/n_patients*100:.1f}%)")
print(f"Patients with 1  encounter    : {(encounter_counts == 1).sum():,}")
print()
print("This is important: I must split by patient, not by row.")
print("I will address this in Notebook 06 (Leakage & Splits).")
"""),

md("## Step 8 — Inspect age distribution"),
code("""\
# Age is stored as decade-wide bins: [0-10), [10-20), etc.
# I check the distribution and visualise it.

print("=== Age Distribution ===")
age_counts = df_raw["age"].value_counts().sort_index()
print(age_counts.to_string())

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(age_counts.index, age_counts.values, color=COLORS["no_readmit"],
       alpha=0.8, edgecolor="white")
ax.set_title("Patient Age Distribution", pad=12)
ax.set_xlabel("Age Group")
ax.set_ylabel("Number of Encounters")
plt.xticks(rotation=30)
plt.tight_layout()
save_figure(fig, "01_age_distribution.png", FIGURES_DIR)
plt.show()
"""),

md("## Step 9 — Inspect diagnosis columns"),
code("""\
# The dataset has three ICD-9 diagnosis code columns: diag_1, diag_2, diag_3.
# These contain raw ICD-9 codes like '250', '428', 'E11.9', etc.
# I inspect how many unique codes there are and what the most common ones look like.

for diag_col in ["diag_1", "diag_2", "diag_3"]:
    n_unique = df_raw[diag_col].nunique()
    top5 = df_raw[diag_col].value_counts().head(5)
    n_q  = (df_raw[diag_col] == "?").sum()
    print(f"=== {diag_col} ===")
    print(f"  Unique values: {n_unique:,}  |  '?' count: {n_q:,}")
    print(f"  Top 5 codes:")
    for code, cnt in top5.items():
        print(f"    {code:8s}: {cnt:,}")
    print()
print("ICD-9 codes need to be grouped into disease categories in Notebook 03.")
"""),

md("## Step 10 — Load and inspect the ID mapping file"),
code("""\
# The IDs_mapping.csv file maps integer codes in admission_type_id,
# discharge_disposition_id, and admission_source_id to readable labels.

ids_df = pd.read_csv(IDS_MAP_FILE)
print(f"IDs mapping file: {ids_df.shape[0]} rows × {ids_df.shape[1]} columns")
print()
print(ids_df.head(20).to_string())
"""),

md("""## Summary of Notebook 01

I loaded the **UCI Diabetes 130-US Hospitals** dataset:
- **101,766 encounters**, **50 columns**, **~71,518 unique patients**
- Target: `readmitted` — ~11% early readmission rate (class imbalance)
- Missing values encoded as **`?`** in string columns; need explicit handling
- Key columns with heavy missingness: `weight` (~97%), `payer_code` (~40%), `medical_specialty` (~49%)
- ICD-9 diagnosis codes need grouping (thousands of unique values)
- **Patient-aware splitting is essential** — same patient can appear multiple times

**Next step:** Notebook 02 — Data Cleaning & Preprocessing
"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 02 — Data Cleaning & Preprocessing (Task 2)
# ══════════════════════════════════════════════════════════════════════════════
nb02 = [
md("""# Notebook 02 — Data Cleaning & Preprocessing
**Decodelabs Internship | Week 2 | Task 2**

---
## What this notebook does
I systematically clean the raw dataset. This involves replacing `?` with NaN,
dropping columns with excessive missingness, removing clinically invalid rows,
handling duplicate patient encounters, and saving a clean interim dataset.

## Why this step matters
The raw dataset has several quality issues that would corrupt any downstream analysis.
Most importantly, `?` is not automatically a missing value in pandas — so every analysis
would silently treat it as a real category until I fix it here.

## What Python / ML concepts I practise
- Replacing placeholder missing values (`?` → NaN)
- Deciding when to drop columns vs impute vs encode as category
- Removing rows based on clinical exclusion criteria
- Deduplication at the patient level
- Documenting every cleaning decision with a rationale

## Input files
- `data/raw/diabetic_data.csv`

## Output files
- `data/interim/diabetic_interim.csv` — cleaned, not yet feature-engineered
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(RAW_FILE, low_memory=False)
original_shape = df.shape
print(f"Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
"""),

md("## Step 1 — Replace '?' with NaN everywhere"),
code("""\
# '?' appears in object columns and represents missing data.
# I replace it with np.nan so pandas treats it as a proper missing value.
# I also record how many '?' values existed per column for documentation.

before_counts = {col: (df[col] == "?").sum() 
                 for col in df.select_dtypes(include="object").columns
                 if (df[col] == "?").sum() > 0}

# Replace '?' with NaN across all columns
df.replace("?", np.nan, inplace=True)

print("Replaced '?' with NaN in all columns.")
print()
print("=== Columns affected ===")
for col, cnt in sorted(before_counts.items(), key=lambda x: -x[1]):
    print(f"  {col:35s}: {cnt:,} values replaced")

total_q = sum(before_counts.values())
print(f"\\nTotal replacements: {total_q:,}")
"""),

md("## Step 2 — Assess missing values after replacement"),
code("""\
# Now I get a complete picture of missingness using standard NaN detection.

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
missing_df = missing_df[missing_df["count"] > 0].sort_values("percent", ascending=False)

print("=== Missing Values Summary ===")
print(missing_df.to_string())
print(f"\\nTotal missing values: {missing.sum():,}")
"""),

md("""## Step 3 — Drop high-missingness columns

I drop columns where the missingness rate is too high to be useful or where
the column is not relevant to the readmission prediction task.

**Decisions:**
- `weight` (~97% missing): Drop. Almost no usable data.
- `payer_code` (~40% missing): Drop. Not a clinical predictor; administrative.
- `medical_specialty` (~49% missing): Drop. Too many missing to impute reliably.
- `encounter_id`: Drop. This is just a row ID, not a clinical feature.

**Retain:**
- `patient_nbr`: Keep temporarily for patient-aware splitting; remove before modelling.
"""),
code("""\
cols_to_drop = ["weight", "payer_code", "medical_specialty", "encounter_id"]

for col in cols_to_drop:
    pct = df[col].isnull().mean() * 100 if col != "encounter_id" else 0
    print(f"  Dropping: {col:25s}  (missing: {pct:.1f}%)")

df.drop(columns=cols_to_drop, inplace=True)

print(f"\\nShape after dropping columns: {df.shape[0]:,} rows × {df.shape[1]} columns")
"""),

md("## Step 4 — Remove clinically invalid rows"),
code("""\
# Some discharge_disposition_id values indicate the patient died or was sent to
# hospice. These patients cannot be readmitted, so including them would distort
# the model. I remove them following standard clinical exclusion criteria.
#
# IDs for: Expired (11), Hospice/medical (13), Hospice/home (14),
#          Expired in medical facility (19), Expired (20), Expired (21)

EXCLUDE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

n_before = len(df)
df = df[~df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_IDS)]
n_removed = n_before - len(df)

print(f"Removed {n_removed:,} rows with hospice/expired discharge disposition.")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
"""),

md("## Step 5 — Handle gender column"),
code("""\
# Gender has values: Male, Female, Unknown/Invalid
# 'Unknown/Invalid' is a tiny fraction — I drop those rows.

print("=== Gender value counts ===")
print(df["gender"].value_counts())

n_before = len(df)
df = df[df["gender"].isin(["Male", "Female"])]
print(f"\\nRemoved {n_before - len(df)} rows with Unknown/Invalid gender.")
print(f"Remaining: {len(df):,}")
"""),

md("## Step 6 — Handle remaining missing values"),
code("""\
# After the column drops and row removals, I check what's left.
# For the diagnosis columns (diag_1, diag_2, diag_3), a small number of
# rows have missing values. I fill these with 'Unknown' since they will
# be categorised in the feature engineering step.

# Fill missing diagnosis codes with 'Unknown'
for col in ["diag_1", "diag_2", "diag_3"]:
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        df[col].fillna("Unknown", inplace=True)
        print(f"  {col}: filled {n_missing} missing values with 'Unknown'")

# Check remaining missing values
remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

if len(remaining_missing) == 0:
    print("\\nNo remaining missing values. Dataset is complete.")
else:
    print("\\nRemaining missing values:")
    print(remaining_missing.to_string())
"""),

md("## Step 7 — Create binary target variable"),
code("""\
# I binarise the readmitted column:
#   '<30' (early readmission) → 1
#   'NO' and '>30'            → 0

df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

print("=== Original readmitted distribution ===")
print(df["readmitted"].value_counts())
print()
print("=== Binary target distribution ===")
counts = df["readmitted_binary"].value_counts().sort_index()
pcts   = df["readmitted_binary"].value_counts(normalize=True).sort_index() * 100
for v in [0, 1]:
    label = "No early readmission" if v == 0 else "Early readmission"
    print(f"  {v} ({label:25s}): {counts[v]:7,} ({pcts[v]:5.1f}%)")

# Drop the original readmitted column now that we have the binary version
df.drop(columns=["readmitted"], inplace=True)
"""),

md("## Step 8 — Handle duplicate patient encounters"),
code("""\
# Multiple encounters per patient can distort training if not handled.
# Strategy: keep only the FIRST encounter for each patient.
# This mimics a real-world scenario where we predict readmission risk at first admission.
# Note: I keep patient_nbr as a column for now; it will be used for splitting in NB06.

n_before    = len(df)
n_patients  = df[PATIENT_ID_COL].nunique()

# Sort by patient, then keep the first row per patient
df = df.sort_values([PATIENT_ID_COL]).drop_duplicates(subset=PATIENT_ID_COL, keep="first")
n_after     = len(df)

print(f"Rows before deduplication: {n_before:,}")
print(f"Rows after  deduplication: {n_after:,}")
print(f"Rows removed             : {n_before - n_after:,}")
print(f"Unique patients remaining: {df[PATIENT_ID_COL].nunique():,}")
print()
print("I keep only the first encounter per patient to avoid data leakage")
print("and to match the realistic clinical use case.")
"""),

md("## Step 9 — Final cleaning summary"),
code("""\
# I summarise every change made in this notebook.

print("=" * 60)
print("  Data Cleaning Summary")
print("=" * 60)
print(f"  Original shape : {original_shape[0]:,} rows × {original_shape[1]} cols")
print(f"  Final shape    : {df.shape[0]:,} rows × {df.shape[1]} cols")
print()
print(f"  Rows removed:")
print(f"    Hospice/expired discharge  : ~{n_removed:,}")
print(f"    Unknown/Invalid gender     : small")
print(f"    Duplicate patients         : {n_before - n_after:,}")
print()
print(f"  Columns removed: weight, payer_code, medical_specialty, encounter_id")
print(f"  Missing '?'    : replaced with NaN; diag missing filled with 'Unknown'")
print(f"  Target column  : readmitted → readmitted_binary (0 / 1)")
print()
print(f"  Missing values remaining: {df.isnull().sum().sum()}")
print(f"  Duplicate rows           : {df.duplicated().sum()}")
"""),

md("## Step 10 — Save interim dataset"),
code("""\
# I save the cleaned (but not yet feature-engineered) dataset to data/interim/.
# This preserves the cleaning step as a separate checkpoint.

os.makedirs(os.path.dirname(INTERIM_FILE), exist_ok=True)
df.to_csv(INTERIM_FILE, index=False)
print(f"Interim dataset saved: {INTERIM_FILE}")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Quick check: reload and verify
df_check = pd.read_csv(INTERIM_FILE)
assert df_check.shape == df.shape, "Shape mismatch on reload!"
print("Reload check passed.")
"""),

md("""## Summary of Notebook 02

| Step | Action | Rows/Cols Affected |
|---|---|---|
| Replace `?` | → NaN | ~98,000 cells across 5 columns |
| Drop columns | `weight`, `payer_code`, `medical_specialty`, `encounter_id` | -4 columns |
| Remove expired | hospice/expired discharge IDs | ~2,200 rows |
| Remove invalid gender | Unknown/Invalid | ~3 rows |
| Keep first encounter | per patient | ~30,000+ rows |
| Binarise target | `readmitted` → `readmitted_binary` | — |

**Output:** `data/interim/diabetic_interim.csv`

**Next step:** Notebook 03 — Feature Engineering
"""),
]

# ── SAVE NB 00, 01, 02 ──────────────────────────────────────────────────────
print("Building notebooks 00–02...")
save_nb("00_project_overview.ipynb",         nb00)
save_nb("01_data_collection.ipynb",          nb01)
save_nb("02_data_cleaning.ipynb",            nb02)
print("Done with 00-02.")
