"""build_notebooks_part3.py — NB06 (Leakage/Splits) + NB07 (Baseline) + NB08 (Advanced)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")

def nb(cells):
    return {"nbformat":4,"nbformat_minor":5,
            "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                        "language_info":{"name":"python","version":"3.9.0"}},
            "cells":cells}
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
# NOTEBOOK 06 — Leakage Control & Patient-Aware Splits
# ══════════════════════════════════════════════════════════════════════════════
nb06 = [
md("""# Notebook 06 — Leakage Control & Train / Validation / Test Splits
**Decodelabs Internship | Week 2**

---
## What this notebook does
I create the final train / validation / test splits using a **patient-aware** strategy,
apply a preprocessing pipeline (scaling, log-transforms), verify there is no leakage,
and save the three datasets ready for modelling.

## Why this step matters
**Data leakage** is one of the most common and damaging mistakes in ML.
In this dataset, the same patient can have multiple encounters. If I split randomly
by row, the same patient could appear in both training and test — and the model could
learn that patient's characteristics rather than general population patterns. This would
produce optimistically biased test metrics that don't hold in deployment.

I must split by **patient ID**, ensuring no patient appears in more than one split.

## What Python / ML concepts I practise
- GroupShuffleSplit for patient-aware splitting
- Verifying set overlap
- StandardScaler fitted only on training data (no leakage)
- Log1p transform for skewed features
- Pipeline construction

## Input files  `data/processed/diabetic_processed.csv`
## Output files `data/processed/train.csv`, `val.csv`, `test.csv`
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(PROCESSED_FILE)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Unique patients: {df[PATIENT_ID_COL].nunique():,}")
"""),

md("## Step 1 — Define feature and target columns"),
code("""\
# I define which columns are features (X), which is the target (y),
# and which is the group ID used for patient-aware splitting.

# Remove non-feature columns
drop_for_modelling = [TARGET_COL, PATIENT_ID_COL]

# Remove diagnosis raw category columns if they have too many unique values
# (will be one-hot encoded separately if needed — for now keep diag categories)
feature_cols = [c for c in df.columns if c not in drop_for_modelling]

print(f"Feature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:3d}. {col}")
"""),

md("## Step 2 — One-hot encode categorical string columns"),
code("""\
# The diagnosis category and race columns are still strings.
# I one-hot encode them. pd.get_dummies() is the simplest approach for a beginner.

cat_string_cols = df[feature_cols].select_dtypes(include="object").columns.tolist()
print(f"String columns to one-hot encode: {cat_string_cols}")

df_encoded = pd.get_dummies(df, columns=cat_string_cols, drop_first=False, dtype=int)

# Update feature cols list after encoding
feature_cols_encoded = [c for c in df_encoded.columns if c not in drop_for_modelling]

print(f"\\nShape before encoding: {df.shape}")
print(f"Shape after  encoding: {df_encoded.shape}")
print(f"New feature count: {len(feature_cols_encoded)}")
"""),

md("## Step 3 — Patient-aware train / validation / test split"),
code("""\
# I use GroupShuffleSplit which splits groups (patients) entirely into one split.
# This guarantees: NO patient appears in both train and test.
#
# Split strategy:
#   - 70% training   (patients)
#   - 15% validation (patients)  — for hyperparameter tuning
#   - 15% test       (patients)  — final evaluation only, touch once

X = df_encoded[feature_cols_encoded]
y = df_encoded[TARGET_COL]
groups = df_encoded[PATIENT_ID_COL]   # group = patient ID

# Step A: split off 15% test set
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]
groups_train_val = groups.iloc[train_val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Step B: split remaining into 70% train / 15% val (val is ~15%/(70%+15%) ≈ 17.6% of train_val)
val_fraction = 0.15 / (1 - 0.15)   # ≈ 0.176
gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=RANDOM_SEED)
train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=groups_train_val))

X_train = X_train_val.iloc[train_idx]
y_train = y_train_val.iloc[train_idx]
groups_train = groups_train_val.iloc[train_idx]

X_val = X_train_val.iloc[val_idx]
y_val = y_train_val.iloc[val_idx]

print(f"Training set    : {len(X_train):,} rows | {y_train.mean()*100:.1f}% positive")
print(f"Validation set  : {len(X_val):,} rows  | {y_val.mean()*100:.1f}% positive")
print(f"Test set        : {len(X_test):,} rows  | {y_test.mean()*100:.1f}% positive")
"""),

md("## Step 4 — Verify zero patient overlap"),
code("""\
# This is a critical leakage check. No patient should appear in more than one split.

train_patients = set(groups_train.values)
val_patients   = set(groups_train_val.iloc[val_idx].values)
test_patients  = set(groups.iloc[test_idx].values)

train_val_overlap  = train_patients & val_patients
train_test_overlap = train_patients & test_patients
val_test_overlap   = val_patients & test_patients

print("=== Patient Overlap Check ===")
print(f"Train ∩ Val  : {len(train_val_overlap)} patients  (should be 0)")
print(f"Train ∩ Test : {len(train_test_overlap)} patients  (should be 0)")
print(f"Val ∩ Test   : {len(val_test_overlap)} patients  (should be 0)")

if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
    print("\\n✓ PASSED: No patient overlap between any splits.")
    print("This means the train/val/test split is clean and leakage-free.")
else:
    print("\\n✗ FAILED: Overlap found — investigate before continuing!")
"""),

md("## Step 5 — Apply log1p transform to skewed features"),
code("""\
# These features are right-skewed (most values are 0, a few are very large).
# log1p(x) = log(x + 1) maps 0 → 0, reduces the impact of extreme values,
# and helps linear models like Logistic Regression converge.
# Tree models don't need this, but it doesn't hurt.

skewed_cols = ["number_outpatient", "number_emergency",
               "number_inpatient", "total_prior_visits"]
skewed_cols = [c for c in skewed_cols if c in X_train.columns]

print(f"Applying log1p to {len(skewed_cols)} skewed columns: {skewed_cols}")

# Make copies to avoid modifying original DataFrames
X_train = X_train.copy()
X_val   = X_val.copy()
X_test  = X_test.copy()

for col in skewed_cols:
    X_train[col] = np.log1p(X_train[col])
    X_val[col]   = np.log1p(X_val[col])
    X_test[col]  = np.log1p(X_test[col])

print("Log1p transforms applied.")
print(f"Example: number_inpatient max before: {df[skewed_cols[2]].max()}")
print(f"         number_inpatient max after : {X_train[skewed_cols[2]].max():.2f}")
"""),

md("## Step 6 — Scale features (StandardScaler)"),
code("""\
# I fit StandardScaler ONLY on the training data to avoid leakage.
# Then I apply the same transform to validation and test data.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)    # fit + transform on train
X_val_scaled   = scaler.transform(X_val)          # transform only on val
X_test_scaled  = scaler.transform(X_test)         # transform only on test

# Convert back to DataFrames to preserve column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled   = pd.DataFrame(X_val_scaled,   columns=X_val.columns,   index=X_val.index)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns,  index=X_test.index)

print("StandardScaler applied.")
print(f"Train column means (should be ≈ 0): {X_train_scaled.mean().abs().mean():.4f}")
print(f"Train column stds  (should be ≈ 1): {X_train_scaled.std().mean():.4f}")
"""),

md("## Step 7 — Save splits to disk"),
code("""\
import os

# I attach the target back to each set for convenient saving
def make_full_df(X_scaled, y):
    d = X_scaled.copy()
    d[TARGET_COL] = y.values
    return d

train_df = make_full_df(X_train_scaled, y_train)
val_df   = make_full_df(X_val_scaled,   y_val)
test_df  = make_full_df(X_test_scaled,  y_test)

train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE,     index=False)
test_df.to_csv(TEST_FILE,   index=False)

print(f"Saved train set : {TRAIN_FILE}  ({len(train_df):,} rows)")
print(f"Saved val set   : {VAL_FILE}   ({len(val_df):,} rows)")
print(f"Saved test set  : {TEST_FILE}  ({len(test_df):,} rows)")
print()
print("Feature columns saved:", len(feature_cols_encoded))
"""),

md("""## Summary of Notebook 06

**Split sizes:**
- Training : ~70% of unique patients
- Validation: ~15% of unique patients  
- Test: ~15% of unique patients

**Leakage checks passed:** ✓ Zero patient overlap across all three splits.

**Preprocessing applied:**
- One-hot encoding for categorical string columns
- log1p transform for 4 highly skewed utilisation columns
- StandardScaler fitted only on training data

**Next step:** Notebook 07 — Baseline Models
"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 07 — Baseline Models (Task 5 Part 1)
# ══════════════════════════════════════════════════════════════════════════════
nb07 = [
md("""# Notebook 07 — Baseline Models
**Decodelabs Internship | Week 2 | Task 5 (Part 1)**

---
## What this notebook does
I train two baseline models: a Dummy Classifier (always predicts majority class) and a
Logistic Regression. These establish a performance floor and show what a simple linear
model can achieve before introducing complex ensembles.

## Why this step matters
In any ML project, you must beat a sensible baseline before claiming your model is useful.
If a complex model only slightly beats Logistic Regression, you should probably use Logistic
Regression — it's faster, interpretable, and more robust to distribution shift.

## What Python / ML concepts I practise
- DummyClassifier as a sanity check
- Logistic Regression with class_weight to handle class imbalance
- Cross-validation on training data
- Full metrics evaluation on validation set

## Input files `data/processed/train.csv`, `val.csv`
## Output files Tables to `reports/tables/`
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv(TRAIN_FILE)
val_df   = pd.read_csv(VAL_FILE)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_val   = val_df.drop(columns=[TARGET_COL])
y_val   = val_df[TARGET_COL]

print(f"Training set  : {len(X_train):,} rows | {y_train.mean()*100:.1f}% positive")
print(f"Validation set: {len(X_val):,} rows  | {y_val.mean()*100:.1f}% positive")
print(f"Features      : {X_train.shape[1]}")
"""),

md("## Step 1 — Dummy Classifier (absolute baseline)"),
code("""\
# The Dummy Classifier predicts the most frequent class (0) every single time.
# This is my floor: any real model must beat this.
# On an imbalanced dataset (~89% class 0), this gives ~89% accuracy
# but 0% recall for class 1 — completely useless clinically.

dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
dummy.fit(X_train, y_train)

y_pred_dummy = dummy.predict(X_val)

print("=== Dummy Classifier on Validation Set ===")
print(classification_report(y_val, y_pred_dummy,
      target_names=["No Readmit (0)", "Early Readmit (1)"]))
print()
print("As expected: Recall for class 1 = 0.00")
print("The dummy model catches NONE of the early readmissions.")
print("Any model I build must have class-1 Recall > 0.")
"""),

md("## Step 2 — Logistic Regression with balanced class weight"),
code("""\
# Logistic Regression is a linear model that estimates the probability of class 1.
# class_weight='balanced' automatically adjusts weights inversely proportional
# to class frequency, which helps the model pay more attention to the rare class (1).

logreg = LogisticRegression(
    class_weight="balanced",   # handles class imbalance
    max_iter=2000,             # allow enough iterations to converge
    random_state=RANDOM_SEED,
    solver="lbfgs",            # efficient solver for medium-sized datasets
    C=1.0                      # regularisation strength (1.0 = default)
)

print("Training Logistic Regression...")
logreg.fit(X_train, y_train)
print("Training complete.")
"""),

code("""\
# Evaluate on validation set
y_pred_lr = logreg.predict(X_val)
y_prob_lr = logreg.predict_proba(X_val)[:, 1]

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

print("=== Logistic Regression on Validation Set ===")
print(classification_report(y_val, y_pred_lr,
      target_names=["No Readmit (0)", "Early Readmit (1)"]))

roc_auc = roc_auc_score(y_val, y_prob_lr)
avg_prec = average_precision_score(y_val, y_prob_lr)
print(f"ROC-AUC          : {roc_auc:.4f}")
print(f"Average Precision: {avg_prec:.4f}")
print()
print("Interpretation:")
print(f"  ROC-AUC {roc_auc:.3f} means the model ranks a random positive higher")
print(f"  than a random negative {roc_auc*100:.1f}% of the time.")
"""),

md("## Step 3 — Cross-validation on training data"),
code("""\
# I run 5-fold stratified cross-validation on the training data.
# This gives a more reliable estimate of performance than a single val split.
# StratifiedKFold preserves class proportions in each fold.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

cv_results = cross_validate(
    logreg, X_train, y_train,
    cv=cv,
    scoring=["roc_auc", "f1", "recall", "precision"],
    return_train_score=True
)

print("=== 5-Fold Cross-Validation: Logistic Regression ===")
for metric in ["roc_auc", "f1", "recall", "precision"]:
    test_scores = cv_results[f"test_{metric}"]
    train_scores = cv_results[f"train_{metric}"]
    print(f"  {metric:12s}  Train: {train_scores.mean():.3f}±{train_scores.std():.3f}  "
          f"Val: {test_scores.mean():.3f}±{test_scores.std():.3f}")

print()
print("If Train >> Val scores, the model is overfitting.")
print("If Train ≈ Val, generalisation is good.")
"""),

md("## Step 4 — Inspect Logistic Regression coefficients"),
code("""\
# For Logistic Regression, the coefficients show which features push the
# predicted probability up (positive) or down (negative).
# These are only interpretable for features on the same scale (which they are,
# since we applied StandardScaler in NB06).

coefs = pd.Series(logreg.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
top_positive = coefs[coefs > 0].head(10)
top_negative = coefs[coefs < 0].head(10)

print("=== Top 10 Positive Coefficients (increase readmission probability) ===")
print(top_positive.round(4).to_string())
print()
print("=== Top 10 Negative Coefficients (decrease readmission probability) ===")
print(top_negative.round(4).to_string())
"""),

md("## Step 5 — Save baseline metrics"),
code("""\
from src.eval_utils import compute_metrics

baseline_results = [
    compute_metrics(y_val, dummy.predict(X_val), None, "Dummy Classifier"),
    compute_metrics(y_val, y_pred_lr, y_prob_lr, "Logistic Regression"),
]

import json
results_path = os.path.join(TABLES_DIR, "07_baseline_results.json")
os.makedirs(TABLES_DIR, exist_ok=True)
with open(results_path, "w") as f:
    json.dump(baseline_results, f, indent=2)
print(f"Baseline results saved: {results_path}")

for r in baseline_results:
    print(f"\\n  {r['Model']}")
    for k, v in r.items():
        if k != "Model":
            print(f"    {k:20s}: {v}")
"""),

md("""## Summary of Notebook 07

| Model | ROC-AUC | F1 (class 1) | Recall (class 1) |
|---|---|---|---|
| Dummy Classifier | N/A | 0.00 | 0.00 |
| Logistic Regression | [run to see] | [run to see] | [run to see] |

Logistic Regression with balanced class weights establishes a solid baseline.
Any ensemble model must exceed these scores to justify its additional complexity.

**Next step:** Notebook 08 — Advanced Models (Random Forest, GradientBoosting, XGBoost)
"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 08 — Advanced Models (Task 5 Part 2)
# ══════════════════════════════════════════════════════════════════════════════
nb08 = [
md("""# Notebook 08 — Advanced Models: Random Forest, GradientBoosting & XGBoost
**Decodelabs Internship | Week 2 | Task 5 (Part 2)**

---
## What this notebook does
I train three ensemble models — Random Forest, Gradient Boosting, and XGBoost —
tune their key hyperparameters using the validation set, and evaluate each model
comprehensively. I also handle class imbalance with SMOTE on the training data.

## Why this step matters
Ensemble methods (combining many decision trees) are consistently among the
best-performing models for tabular clinical data. Understanding why they work —
and how to tune them responsibly without causing leakage — is a core ML skill.

## What Python / ML concepts I practise
- RandomForestClassifier, GradientBoostingClassifier, XGBClassifier
- SMOTE oversampling for class imbalance
- Hyperparameter meaning and effect
- Validation-set-based model selection
- Avoiding test set contamination during tuning

## Input files `data/processed/train.csv`, `val.csv`
## Output files Model comparison to `reports/tables/`
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, recall_score, classification_report)
import warnings
warnings.filterwarnings("ignore")

# Try to import XGBoost; fall back gracefully if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("XGBoost available.")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Will skip XGBoost model.")
    print("Install with: pip install xgboost")

# Try to import imbalanced-learn for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("imbalanced-learn (SMOTE) available.")
except ImportError:
    SMOTE_AVAILABLE = False
    print("imbalanced-learn not installed. Will use class_weight instead of SMOTE.")
    print("Install with: pip install imbalanced-learn")

train_df = pd.read_csv(TRAIN_FILE)
val_df   = pd.read_csv(VAL_FILE)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_val   = val_df.drop(columns=[TARGET_COL])
y_val   = val_df[TARGET_COL]

print(f"\\nTraining : {len(X_train):,} rows | {y_train.mean()*100:.1f}% positive")
print(f"Validation: {len(X_val):,} rows | {y_val.mean()*100:.1f}% positive")
"""),

md("## Step 1 — Apply SMOTE to training data"),
code("""\
# SMOTE (Synthetic Minority Over-sampling TEchnique) creates synthetic samples
# of the minority class (early readmission) in feature space.
# This is an alternative to class_weight for handling class imbalance.
#
# CRITICAL: I apply SMOTE ONLY to the training set.
# Never apply SMOTE to validation or test — that would be data leakage.

if SMOTE_AVAILABLE:
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("SMOTE applied to training data.")
    print(f"  Before: {len(X_train):,} rows | {y_train.mean()*100:.1f}% positive")
    print(f"  After : {len(X_train_resampled):,} rows | {y_train_resampled.mean()*100:.1f}% positive")
    print()
    print("The training set is now balanced (50/50).")
    print("Validation and test sets remain unchanged (real distribution).")
else:
    X_train_resampled = X_train
    y_train_resampled = y_train
    print("Using original imbalanced training data (class_weight will handle imbalance).")
"""),

md("## Step 2 — Random Forest"),
code("""\
# Random Forest builds many decision trees on random subsets of the training data
# and averages their predictions. This reduces overfitting compared to a single tree.
#
# Key hyperparameters I set:
#   n_estimators=300    : number of trees (more = better up to a point, slower)
#   max_depth=10        : max depth per tree (limits overfitting)
#   min_samples_leaf=20 : each leaf must have at least 20 samples (regularisation)
#   class_weight='balanced_subsample': down-weight majority class per tree
#   n_jobs=-1           : use all available CPU cores

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_SEED,
    n_jobs=-1
)

print("Training Random Forest (300 trees)...")
rf.fit(X_train_resampled, y_train_resampled)
print("Training complete.")

y_pred_rf = rf.predict(X_val)
y_prob_rf = rf.predict_proba(X_val)[:, 1]

print(f"\\n=== Random Forest Validation Results ===")
print(classification_report(y_val, y_pred_rf,
      target_names=["No Readmit (0)", "Early Readmit (1)"]))
print(f"ROC-AUC          : {roc_auc_score(y_val, y_prob_rf):.4f}")
print(f"Average Precision: {average_precision_score(y_val, y_prob_rf):.4f}")
"""),

md("## Step 3 — Gradient Boosting"),
code("""\
# Gradient Boosting builds trees sequentially, where each tree corrects
# the errors of the previous one. It tends to be more accurate than Random Forest
# on tabular data but takes longer to train.
#
# Key hyperparameters:
#   n_estimators=300    : number of sequential trees
#   max_depth=4         : shallow trees work better for gradient boosting
#   learning_rate=0.05  : how much each tree corrects (lower = more trees needed)
#   subsample=0.8       : use 80% of training data per tree (stochastic boosting)

gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=RANDOM_SEED
)

print("Training Gradient Boosting (300 estimators)...")
gb.fit(X_train_resampled, y_train_resampled)
print("Training complete.")

y_pred_gb = gb.predict(X_val)
y_prob_gb = gb.predict_proba(X_val)[:, 1]

print(f"\\n=== Gradient Boosting Validation Results ===")
print(classification_report(y_val, y_pred_gb,
      target_names=["No Readmit (0)", "Early Readmit (1)"]))
print(f"ROC-AUC          : {roc_auc_score(y_val, y_prob_gb):.4f}")
print(f"Average Precision: {average_precision_score(y_val, y_prob_gb):.4f}")
"""),

md("## Step 4 — XGBoost"),
code("""\
if XGBOOST_AVAILABLE:
    # XGBoost is a highly optimised gradient boosting implementation.
    # It is often the best-performing algorithm on structured/tabular data.
    # scale_pos_weight handles class imbalance: set to (N negatives / N positives).
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count
    print(f"scale_pos_weight = {scale_weight:.1f} (balances class imbalance)")
    
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,  # handle imbalance
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0
    )
    
    print("Training XGBoost...")
    xgb.fit(X_train_resampled, y_train_resampled)
    print("Training complete.")
    
    y_pred_xgb = xgb.predict(X_val)
    y_prob_xgb = xgb.predict_proba(X_val)[:, 1]
    
    print(f"\\n=== XGBoost Validation Results ===")
    print(classification_report(y_val, y_pred_xgb,
          target_names=["No Readmit (0)", "Early Readmit (1)"]))
    print(f"ROC-AUC          : {roc_auc_score(y_val, y_prob_xgb):.4f}")
    print(f"Average Precision: {average_precision_score(y_val, y_prob_xgb):.4f}")
else:
    print("XGBoost not available. Skipping.")
    xgb = None
"""),

md("## Step 5 — Compare all models on validation set"),
code("""\
from src.eval_utils import compute_metrics, compare_models_table

results = []

# Load logistic regression baseline results if available
import json, os
lr_path = os.path.join(TABLES_DIR, "07_baseline_results.json")
if os.path.exists(lr_path):
    with open(lr_path) as f:
        baseline_results = json.load(f)
    # Load logistic regression predictions
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(class_weight="balanced", max_iter=2000,
                                   random_state=RANDOM_SEED)
    lr_model.fit(X_train_resampled, y_train_resampled)
    y_pred_lr = lr_model.predict(X_val)
    y_prob_lr = lr_model.predict_proba(X_val)[:, 1]
    results.append(compute_metrics(y_val, y_pred_lr, y_prob_lr, "Logistic Regression"))

results.append(compute_metrics(y_val, y_pred_rf, y_prob_rf, "Random Forest"))
results.append(compute_metrics(y_val, y_pred_gb, y_prob_gb, "Gradient Boosting"))
if XGBOOST_AVAILABLE and xgb is not None:
    results.append(compute_metrics(y_val, y_pred_xgb, y_prob_xgb, "XGBoost"))

comp_df = compare_models_table(results)
print("=== Model Comparison (Validation Set) ===")
print(comp_df[["Accuracy","Balanced Acc.","Precision","Recall","F1-Score","ROC-AUC","Avg. Precision"]].to_string())

save_table(comp_df, "08_model_comparison_val.csv", TABLES_DIR)
print("\\nModel comparison saved.")
"""),

md("## Step 6 — Save trained models for evaluation notebook"),
code("""\
import joblib

models_dir = os.path.join(os.path.dirname(PROCESSED_FILE), "..", "..", "models")
# Save to a models/ directory at project root (create if needed)
models_dir = os.path.join(PROJECT_ROOT, "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(rf, os.path.join(models_dir, "random_forest.pkl"))
joblib.dump(gb, os.path.join(models_dir, "gradient_boosting.pkl"))
if XGBOOST_AVAILABLE and xgb is not None:
    joblib.dump(xgb, os.path.join(models_dir, "xgboost.pkl"))

print(f"Models saved to: {models_dir}")
print(os.listdir(models_dir))
"""),

md("""## Summary of Notebook 08

I trained three ensemble models with class imbalance handling.

**Validation performance (to be filled after running):**

| Model | ROC-AUC | Avg. Precision | Recall (class 1) |
|---|---|---|---|
| Random Forest | [run] | [run] | [run] |
| Gradient Boosting | [run] | [run] | [run] |
| XGBoost | [run] | [run] | [run] |

**Next step:** Notebook 09 — Model Evaluation with Bootstrap Confidence Intervals
"""),
]

print("Building notebooks 06–08...")
save_nb("06_leakage_and_splits.ipynb", nb06)
save_nb("07_baseline_models.ipynb",    nb07)
save_nb("08_advanced_models.ipynb",    nb08)
print("Done with 06-08.")
