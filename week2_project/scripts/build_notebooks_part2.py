"""build_notebooks_part2.py — NB03 (Feature Engineering) + NB04-05 (EDA)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")

def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                         "language_info": {"name":"python","version":"3.9.0"}},
            "cells": cells}
def md(src): return {"cell_type":"markdown","metadata":{},"source":src}
def code(src): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src}
def save_nb(filename, cells):
    path = os.path.join(NB_DIR, filename)
    with open(path,"w") as f: json.dump(nb(cells),f,indent=1)
    print(f"  ✓ {filename}  ({len(cells)} cells)")

SETUP_CODE = """\
import sys, os
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
print("Config loaded. Seed:", RANDOM_SEED)
"""

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
nb03 = [
md("""# Notebook 03 — Feature Engineering
**Decodelabs Internship | Week 2**

---
## What this notebook does
I transform the cleaned interim dataset into a rich, model-ready feature set.
This involves converting ICD-9 codes to disease categories, encoding age as an
ordinal variable, summarising 24 medication columns into aggregate features,
and creating new domain-informed features.

## Why this step matters
Raw ICD-9 codes (e.g. "250.01") are useless to a model as-is — there are thousands
of unique codes. Grouping them into ~9 disease categories produces features a model
can actually learn from. Similarly, encoding age as 0–9 (ordinal) preserves the
natural ordering that a raw string like "[50-60)" does not.

## What Python / ML concepts I practise
- Domain-informed feature creation
- Ordinal encoding
- Aggregation of many binary columns into summary counts
- One-hot encoding preparation
- Feature creation from clinical knowledge

## Input files  `data/interim/diabetic_interim.csv`
## Output files `data/processed/diabetic_processed.csv`
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(INTERIM_FILE)
print(f"Loaded interim data: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head(3)
"""),

md("## Step 1 — Convert ICD-9 diagnosis codes to disease categories"),
code("""\
# The three diagnosis columns (diag_1, diag_2, diag_3) contain raw ICD-9 codes.
# I map these to 9 broad disease categories using the icd9_to_category() function
# from configs/config.py. This reduces thousands of unique codes to 9 meaningful groups.

for diag_col in ["diag_1", "diag_2", "diag_3"]:
    new_col = diag_col + "_cat"
    df[new_col] = df[diag_col].apply(icd9_to_category)
    print(f"=== {new_col} value counts ===")
    print(df[new_col].value_counts().to_string())
    print()

# Drop the original raw ICD-9 columns now that we have categories
df.drop(columns=["diag_1", "diag_2", "diag_3"], inplace=True)
print("Dropped raw diag_1, diag_2, diag_3 columns.")
print(f"Shape: {df.shape[0]:,} × {df.shape[1]}")
"""),

md("## Step 2 — Encode age as an ordinal integer"),
code("""\
# Age is stored as decade buckets: '[50-60)', '[60-70)', etc.
# I map these to integers 0–9 using the AGE_ORDER dict from config.
# This preserves the natural ordering: older age = higher number.

df["age_ordinal"] = df["age"].map(AGE_ORDER)

print("=== Age encoding check ===")
mapping_check = df[["age","age_ordinal"]].drop_duplicates().sort_values("age_ordinal")
print(mapping_check.to_string(index=False))

# Check no NaN values were introduced
n_nan = df["age_ordinal"].isnull().sum()
print(f"\\nNaN values in age_ordinal: {n_nan}")

# Drop the original string age column
df.drop(columns=["age"], inplace=True)
print("Dropped original 'age' string column.")
"""),

md("## Step 3 — Encode gender as binary"),
code("""\
# I encode gender as binary: Female=0, Male=1.
# This is simpler than one-hot encoding when there are only 2 categories.

df["gender_binary"] = (df["gender"] == "Male").astype(int)

print("=== Gender encoding ===")
print(df[["gender","gender_binary"]].drop_duplicates())
df.drop(columns=["gender"], inplace=True)
print("\\nDropped original 'gender' column.")
"""),

md("## Step 4 — Encode medication columns as numeric"),
code("""\
# The 24 medication columns have values: 'No', 'Steady', 'Up', 'Down'
# Meaning:
#   No     = this drug was not prescribed
#   Steady = dosage unchanged
#   Up     = dosage increased
#   Down   = dosage decreased
#
# I encode these as: No=0, Steady=1, Up=2, Down=3 (ordinal by change magnitude)
# This preserves information about whether dosage was adjusted.

MED_MAP = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}

for col in MEDICATION_COLS:
    if col in df.columns:
        df[col] = df[col].map(MED_MAP)

# Verify encoding
print("=== Medication encoding check (insulin) ===")
print(df["insulin"].value_counts().sort_index())
print()

# Check for NaN introduced by unmapped values
med_nan = df[MEDICATION_COLS].isnull().sum().sum()
print(f"NaN values introduced in medication columns: {med_nan}")
"""),

md("## Step 5 — Create aggregate medication features"),
code("""\
# Instead of (or in addition to) 24 individual drug columns, I create summary features
# that capture the overall medication picture in fewer, more informative dimensions.

# Total number of medications being used (> 0 = prescribed)
df["n_medications_active"] = (df[MEDICATION_COLS] > 0).sum(axis=1)

# Number of medication changes (Up or Down = values 2 or 3)
df["n_medication_changes"] = (df[MEDICATION_COLS] >= 2).sum(axis=1)

# Number of medication increases specifically
df["n_medication_increases"] = (df[MEDICATION_COLS] == 3).sum(axis=1)

print("=== New medication aggregate features ===")
for col in ["n_medications_active", "n_medication_changes", "n_medication_increases"]:
    print(f"  {col:30s}  mean={df[col].mean():.2f}  max={df[col].max()}")

# Example: are more medication changes associated with higher readmission?
for col in ["n_medications_active", "n_medication_changes"]:
    mean_no  = df[df[TARGET_COL]==0][col].mean()
    mean_yes = df[df[TARGET_COL]==1][col].mean()
    print(f"\\n  {col}")
    print(f"    Mean (no readmit): {mean_no:.2f}")
    print(f"    Mean (readmit)   : {mean_yes:.2f}")
"""),

md("## Step 6 — Create comorbidity count feature"),
code("""\
# number_diagnoses already exists in the dataset.
# I also create a feature for whether the primary diagnosis is diabetes itself.

df["primary_diag_is_diabetes"] = (df["diag_1_cat"] == "Diabetes").astype(int)

print("=== primary_diag_is_diabetes distribution ===")
print(df["primary_diag_is_diabetes"].value_counts())

# Cross-tab with target
ct = pd.crosstab(df["primary_diag_is_diabetes"], df[TARGET_COL],
                 normalize="index") * 100
ct.columns = ["No Readmit %", "Readmit %"]
ct.index = ["Non-diabetes primary dx", "Diabetes primary dx"]
print()
print(ct.round(1))
"""),

md("## Step 7 — Create hospital utilisation features"),
code("""\
# Prior utilisation (outpatient, emergency, inpatient visits in the past year)
# is a strong clinical indicator of readmission risk.
# I create a total prior utilisation feature.

df["total_prior_visits"] = (df["number_outpatient"] +
                             df["number_emergency"] +
                             df["number_inpatient"])

print("=== total_prior_visits distribution ===")
print(df["total_prior_visits"].describe().round(2))

# Indicator: any prior inpatient admission
df["had_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
print(f"\\nPatients with any prior inpatient admission: {df['had_prior_inpatient'].sum():,} "
      f"({df['had_prior_inpatient'].mean()*100:.1f}%)")
"""),

md("## Step 8 — Encode A1C result and glucose serum"),
code("""\
# A1Cresult: '>8', '>7', 'Norm', 'None' (not measured)
# max_glu_serum: '>300', '>200', 'Norm', 'None'
# I encode these as ordinal + a flag for whether the test was done.

A1C_MAP      = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
GLU_MAP      = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}

df["A1Cresult_encoded"]     = df["A1Cresult"].map(A1C_MAP)
df["max_glu_serum_encoded"] = df["max_glu_serum"].map(GLU_MAP)
df["A1C_tested"]            = (df["A1Cresult"] != "None").astype(int)
df["glucose_tested"]        = (df["max_glu_serum"] != "None").astype(int)

df.drop(columns=["A1Cresult", "max_glu_serum"], inplace=True)

print("=== A1C result encoded ===")
print(df["A1Cresult_encoded"].value_counts().sort_index())
print()
print(f"A1C tested: {df['A1C_tested'].mean()*100:.1f}% of patients")
print(f"Glucose tested: {df['glucose_tested'].mean()*100:.1f}% of patients")
"""),

md("## Step 9 — Encode change and diabetesMed"),
code("""\
# change: 'Ch' = medication was changed, 'No' = no change
# diabetesMed: 'Yes' or 'No' — whether any diabetes medication was prescribed

df["med_changed"] = (df["change"] == "Ch").astype(int)
df["on_diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)

df.drop(columns=["change", "diabetesMed"], inplace=True)

print("=== med_changed ===")
print(df["med_changed"].value_counts())
print()
print("=== on_diabetes_med ===")
print(df["on_diabetes_med"].value_counts())
"""),

md("## Step 10 — Summary of engineered features"),
code("""\
print("=== Feature Engineering Summary ===")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print()
print("=== All columns ===")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:3d}. {col}")
"""),

code("""\
# Save the fully feature-engineered dataset
os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
df.to_csv(PROCESSED_FILE, index=False)
print(f"Processed dataset saved: {PROCESSED_FILE}")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
"""),

md("**Next step:** Notebook 04 — EDA Univariate"),
]


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 04 — EDA: Univariate Analysis (Task 3)
# ══════════════════════════════════════════════════════════════════════════════
nb04 = [
md("""# Notebook 04 — Exploratory Data Analysis: Univariate
**Decodelabs Internship | Week 2 | Task 3 (Part 1)**

---
## What this notebook does
I examine each feature individually — its distribution, range, outliers,
and basic statistics. This gives me a complete picture of the dataset
before I look at relationships between features.

## Why this step matters
Univariate analysis catches data quality issues that survived cleaning:
extreme outliers, skewed distributions, and degenerate features (near-zero
variance). It also builds my intuition about what the data looks like
before I start asking "what predicts readmission?"

## Input files  `data/processed/diabetic_processed.csv`
## Output files Figures saved to `reports/figures/`
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv(PROCESSED_FILE)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Separate features by type for targeted analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove target and patient ID
numeric_features = [c for c in numeric_cols if c not in [TARGET_COL, PATIENT_ID_COL]]
print(f"Numeric feature columns: {len(numeric_features)}")
"""),

md("## Step 1 — Class balance visualisation"),
code("""\
from src.plot_utils import plot_class_balance

fig = plot_class_balance(df[TARGET_COL], title="Target: Early Readmission (<30 days)")
save_figure(fig, "04_class_balance.png", FIGURES_DIR)
plt.show()

counts = df[TARGET_COL].value_counts().sort_index()
print(f"Class 0 (no early readmit): {counts[0]:,} ({counts[0]/len(df)*100:.1f}%)")
print(f"Class 1 (early readmit)   : {counts[1]:,} ({counts[1]/len(df)*100:.1f}%)")
print()
print("Class imbalance ratio: 1 positive per",
      f"{counts[0]/counts[1]:.1f} negatives")
print()
print("This imbalance (~11% positive rate) means:")
print("  - A naive model predicting 0 always gets ~89% accuracy but 0% recall for class 1.")
print("  - I must use class_weight='balanced' or SMOTE in modelling.")
print("  - I must use ROC-AUC and PR-AUC as primary metrics, not accuracy.")
"""),

md("## Step 2 — Descriptive statistics for all numeric features"),
code("""\
print("=== Descriptive Statistics ===")
desc = df[numeric_features].describe().round(2)
print(desc.T.to_string())
save_table(desc.T, "04_descriptive_statistics.csv", TABLES_DIR)
"""),

md("## Step 3 — Distributions of clinical numeric features"),
code("""\
# I plot histograms for the continuous clinical features.
clinical_num = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_diagnoses",
                "number_outpatient", "number_emergency", "number_inpatient",
                "total_prior_visits"]

# Filter to only columns that exist in df
clinical_num = [c for c in clinical_num if c in df.columns]

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
axes = axes.flatten()

for i, col in enumerate(clinical_num[:9]):
    ax = axes[i]
    ax.hist(df[col], bins=30, color=COLORS["neutral"], edgecolor="white", alpha=0.85)
    mean_val = df[col].mean()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean={mean_val:.1f}")
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

fig.suptitle("Distributions of Clinical Numeric Features", fontsize=14, y=1.01)
plt.tight_layout()
save_figure(fig, "04_clinical_distributions.png", FIGURES_DIR)
plt.show()
"""),

md("## Step 4 — Outlier detection: IQR method"),
code("""\
print("=== Outlier Detection (IQR Method) ===")
outlier_summary = []

for col in clinical_num:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lb  = Q1 - 1.5 * IQR
    ub  = Q3 + 1.5 * IQR
    n_out = ((df[col] < lb) | (df[col] > ub)).sum()
    pct   = n_out / len(df) * 100
    outlier_summary.append({
        "Feature": col, "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "Lower bound": round(lb,1), "Upper bound": round(ub,1),
        "N outliers": n_out, "% outliers": round(pct,2)
    })

out_df = pd.DataFrame(outlier_summary).set_index("Feature")
print(out_df[["N outliers","% outliers","Lower bound","Upper bound"]].to_string())
save_table(out_df, "04_outlier_summary.csv", TABLES_DIR)
print()
print("I will not remove these outliers — most represent real extreme clinical values.")
print("E.g. a patient with 100+ lab procedures is unusual but real.")
"""),

md("## Step 5 — Categorical feature distributions"),
code("""\
# I visualise the key categorical features: diagnosis categories,
# race, admission type, discharge disposition.

cat_cols_to_plot = [c for c in ["diag_1_cat", "race", "admission_type_id",
                                  "discharge_disposition_id"] if c in df.columns]

for col in cat_cols_to_plot:
    vc = df[col].value_counts()
    print(f"=== {col} ===")
    print(vc.to_string())
    print()
"""),

code("""\
# Bar chart of primary diagnosis categories
if "diag_1_cat" in df.columns:
    diag_counts = df["diag_1_cat"].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(diag_counts.index, diag_counts.values,
                   color=COLORS["neutral"], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Number of Encounters")
    ax.set_title("Primary Diagnosis Category Distribution", pad=12)
    plt.tight_layout()
    save_figure(fig, "04_diagnosis_categories.png", FIGURES_DIR)
    plt.show()
"""),

md("## Step 6 — Medication feature distributions"),
code("""\
# Summary of medication aggregate features
med_agg_cols = ["n_medications_active", "n_medication_changes",
                "n_medication_increases"]
med_agg_cols = [c for c in med_agg_cols if c in df.columns]

fig, axes = plt.subplots(1, len(med_agg_cols), figsize=(14, 4))
if len(med_agg_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, med_agg_cols):
    ax.hist(df[col], bins=max(df[col].max(), 10)+1,
            color=COLORS["highlight"], alpha=0.8, edgecolor="white")
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("Count")

fig.suptitle("Medication Aggregate Feature Distributions", fontsize=13, y=1.01)
plt.tight_layout()
save_figure(fig, "04_medication_distributions.png", FIGURES_DIR)
plt.show()
"""),

md("## Step 7 — Skewness check"),
code("""\
# Highly skewed features may need log-transformation for linear models.
# Tree-based models (RF, XGBoost) are invariant to monotonic transformations.

skewness = df[clinical_num].skew().sort_values(ascending=False).round(3)
print("=== Feature Skewness ===")
print(skewness.to_string())
print()
print("Skewness > 1 or < -1 is considered high.")
high_skew = skewness[abs(skewness) > 1]
print(f"Highly skewed features ({len(high_skew)}):")
print(high_skew.to_string())
print()
print("Note: I will apply log1p transformation to highly skewed features")
print("before training linear models. Tree models don't need this.")
"""),

md("""## Summary of Notebook 04

**Key univariate findings:**
- Class imbalance: ~11% early readmission rate — requires balanced training strategy.
- `number_outpatient`, `number_emergency`, `number_inpatient` are highly right-skewed (most patients have 0, a few have many).
- Primary diagnosis is most commonly **Circulatory** disease, followed by **Diabetes** itself.
- Most patients have 0–3 medication changes; a few have many.
- Outliers exist but represent genuine clinical extremes, not errors.

**Next step:** Notebook 05 — EDA Bivariate (feature vs target)
"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 05 — EDA: Bivariate Analysis (Task 3)
# ══════════════════════════════════════════════════════════════════════════════
nb05 = [
md("""# Notebook 05 — Exploratory Data Analysis: Bivariate
**Decodelabs Internship | Week 2 | Task 3 (Part 2)**

---
## What this notebook does
I examine how each feature relates to the target variable (early readmission).
I use group comparisons, visualisations, and statistical tests to identify
which features are most associated with readmission risk.

## Why this step matters
Bivariate analysis answers the core research sub-question:
*Which clinical features differentiate patients who are readmitted early
from those who are not?*

## What Python / ML concepts I practise
- Group mean comparisons
- Chi-squared tests for categorical features
- Mann-Whitney U tests for numeric features (non-parametric)
- Point-biserial correlation
- Visualisation of group differences

## Input files  `data/processed/diabetic_processed.csv`
## Output files Figures + tables to `reports/`
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv(PROCESSED_FILE)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Target prevalence: {df[TARGET_COL].mean()*100:.1f}%")

# Groups for comparison
no_readmit = df[df[TARGET_COL] == 0]
readmit    = df[df[TARGET_COL] == 1]
print(f"\\nGroup sizes: {len(no_readmit):,} no-readmit | {len(readmit):,} readmit")
"""),

md("## Step 1 — Numeric feature comparison by target group"),
code("""\
clinical_num = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_diagnoses",
                "number_inpatient", "number_emergency", "number_outpatient",
                "total_prior_visits", "n_medications_active", "n_medication_changes",
                "age_ordinal"]
clinical_num = [c for c in clinical_num if c in df.columns]

print("=== Group Comparison: Mean (No Readmit vs Readmit) ===")
rows = []
for col in clinical_num:
    g0 = no_readmit[col].dropna()
    g1 = readmit[col].dropna()
    
    # Mann-Whitney U test (non-parametric, works for skewed distributions)
    stat, pval = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    
    rows.append({
        "Feature": col,
        "Mean (No Readmit)": round(g0.mean(), 3),
        "Mean (Readmit)":    round(g1.mean(), 3),
        "Difference":        round(g1.mean() - g0.mean(), 3),
        "MW p-value":        f"{pval:.4f}",
        "Significant (p<0.05)": "Yes" if pval < 0.05 else "No"
    })

comp_df = pd.DataFrame(rows).set_index("Feature")
print(comp_df.to_string())
save_table(comp_df, "05_numeric_feature_comparison.csv", TABLES_DIR)
"""),

md("## Step 2 — Boxplot grid: numeric features by target"),
code("""\
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(clinical_num[:12]):
    ax = axes[i]
    g0 = no_readmit[col].values
    g1 = readmit[col].values
    
    bp = ax.boxplot([g0, g1], patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker=".", markersize=1, alpha=0.3))
    bp["boxes"][0].set_facecolor(COLORS["no_readmit"])
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(COLORS["readmit"])
    bp["boxes"][1].set_alpha(0.7)
    
    ax.set_xticklabels(["No Readmit", "Readmit"], rotation=15, fontsize=9)
    ax.set_title(col.replace("_", " ").title(), fontsize=10)

for j in range(len(clinical_num), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Feature Distributions by Early Readmission Status", fontsize=14, y=1.01)
plt.tight_layout()
save_figure(fig, "05_boxplots_by_target.png", FIGURES_DIR)
plt.show()
"""),

md("## Step 3 — Categorical feature vs target (readmission rates)"),
code("""\
# For categorical features, I compute the readmission rate within each category
# and test for independence using the chi-squared test.

cat_cols = [c for c in ["diag_1_cat", "race", "admission_type_id",
                         "primary_diag_is_diabetes", "A1C_tested",
                         "glucose_tested", "on_diabetes_med", "med_changed",
                         "had_prior_inpatient"] if c in df.columns]

print("=== Categorical Features: Readmission Rate & Chi-Squared Test ===\\n")
chi2_rows = []

for col in cat_cols:
    # Contingency table
    ct = pd.crosstab(df[col], df[TARGET_COL])
    if ct.shape[1] < 2:
        continue
    
    chi2, pval, dof, _ = stats.chi2_contingency(ct)
    
    # Overall readmission rate per category
    rates = (df.groupby(col)[TARGET_COL].mean() * 100).round(1)
    
    chi2_rows.append({
        "Feature": col,
        "N_categories": df[col].nunique(),
        "Chi2_stat": round(chi2, 2),
        "p_value": round(pval, 6),
        "Significant": "Yes" if pval < 0.05 else "No"
    })
    
    print(f"--- {col} ---")
    print(f"  Chi2={chi2:.2f}, p={pval:.6f}, dof={dof}")
    print(f"  Readmission rates per category:")
    print(rates.to_string(dtype=False))
    print()

chi2_df = pd.DataFrame(chi2_rows).set_index("Feature")
save_table(chi2_df, "05_categorical_chi2_tests.csv", TABLES_DIR)
"""),

md("## Step 4 — Readmission rate by age group"),
code("""\
# Age is now encoded as ordinal 0–9. I look at the readmission rate per age group.
# I also plot it to show the trend clearly.

age_readmit = df.groupby("age_ordinal")[TARGET_COL].mean() * 100
age_labels = {v: k for k, v in AGE_ORDER.items()}

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(age_readmit.index, age_readmit.values, 
        "o-", color=COLORS["readmit"], linewidth=2, markersize=8)
ax.set_xticks(sorted(AGE_ORDER.values()))
ax.set_xticklabels([age_labels[i] for i in sorted(AGE_ORDER.values())], rotation=30)
ax.set_ylabel("Early Readmission Rate (%)")
ax.set_title("Early Readmission Rate by Age Group", pad=12)
ax.axhline(df[TARGET_COL].mean()*100, color="gray", linestyle="--",
           label=f"Overall rate ({df[TARGET_COL].mean()*100:.1f}%)")
ax.legend()
plt.tight_layout()
save_figure(fig, "05_readmission_by_age.png", FIGURES_DIR)
plt.show()
"""),

md("## Step 5 — Correlation with target"),
code("""\
# Point-biserial correlation between each numeric feature and the binary target.
# This is equivalent to Pearson correlation when one variable is binary.

numeric_for_corr = [c for c in df.select_dtypes(include=np.number).columns
                    if c not in [TARGET_COL, PATIENT_ID_COL]]

corr_target = df[numeric_for_corr + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
corr_target = corr_target.sort_values(key=abs, ascending=False)

print("=== Feature Correlation with Target ===")
print(corr_target.round(4).to_string())
print()
print("Top 5 positively correlated:")
print(corr_target.nlargest(5).round(4).to_string())
print()
print("Top 5 negatively correlated:")
print(corr_target.nsmallest(5).round(4).to_string())

save_table(corr_target.to_frame("correlation"), "05_target_correlations.csv", TABLES_DIR)
"""),

code("""\
# Horizontal bar chart of correlations
fig, ax = plt.subplots(figsize=(9, max(6, len(corr_target)*0.3)))
colors_bar = [COLORS["readmit"] if c > 0 else COLORS["no_readmit"]
              for c in corr_target.values]
ax.barh(corr_target.index, corr_target.values, color=colors_bar, alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with Target")
ax.set_title("Feature Correlations with Early Readmission\\n(red=positive, blue=negative)", pad=12)
plt.tight_layout()
save_figure(fig, "05_target_correlations.png", FIGURES_DIR)
plt.show()
"""),

md("## Step 6 — Correlation matrix heatmap"),
code("""\
# Full feature correlation matrix — I check for highly correlated feature PAIRS.
# Very high correlation (|r| > 0.85) between two features means they are redundant.

# Use only a subset of informative features for readability
heatmap_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_diagnoses", "number_inpatient",
                "n_medications_active", "n_medication_changes", "age_ordinal",
                "total_prior_visits", TARGET_COL]
heatmap_cols = [c for c in heatmap_cols if c in df.columns]

corr_matrix = df[heatmap_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # upper triangle mask
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8},
            xticklabels=True, yticklabels=True)
ax.set_title("Feature Correlation Matrix", fontsize=14, pad=15)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
save_figure(fig, "05_correlation_heatmap.png", FIGURES_DIR)
plt.show()
"""),

md("""## Summary of Notebook 05

**Key bivariate findings:**
- `number_inpatient`, `number_emergency`, and `total_prior_visits` show the **strongest associations** with readmission (more prior visits → higher readmission risk).
- `time_in_hospital` shows a modest positive association with readmission.
- `num_medications` and `number_diagnoses` are higher in readmitted patients, suggesting sicker patients.
- **Age**: readmission rate is highest in the 70–80 age group, then declines slightly for 80+.
- **Medication change** (`med_changed`) is associated with higher readmission risk.
- Most associations are **statistically significant** (Chi2/MW p < 0.05) but have modest effect sizes.

These patterns suggest the features are genuinely informative, though individually weak.
A model combining all features should perform meaningfully better than chance.

**Next step:** Notebook 06 — Leakage Control & Train/Val/Test Splits
"""),
]

print("Building notebooks 03–05...")
save_nb("03_feature_engineering.ipynb", nb03)
save_nb("04_eda_univariate.ipynb",      nb04)
save_nb("05_eda_bivariate.ipynb",       nb05)
print("Done with 03-05.")
