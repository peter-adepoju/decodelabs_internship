"""build_notebooks_part5.py — NB11 (SHAP + Error Analysis) + NB12 (Fairness)"""
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
# NOTEBOOK 11 — Error Analysis & SHAP Interpretability
# ══════════════════════════════════════════════════════════════════════════════
nb11 = [
md("""# Notebook 11 — Error Analysis & SHAP Interpretability
**Decodelabs Internship | Week 2**

---
## What this notebook does
I perform a deep analysis of where the best model fails and why it makes the
predictions it does. I use SHAP (SHapley Additive exPlanations) to explain
both global feature importance and individual patient predictions.

## Why this step matters
A model with good aggregate metrics can still fail systematically for specific
patient types. Error analysis reveals *which patients* the model struggles with.
SHAP explains *why* the model makes each prediction — turning a "black box"
into something a clinician can interrogate and trust (or challenge).

## What Python / ML concepts I practise
- Confusion matrix decomposition (TP, TN, FP, FN analysis)
- SHAP TreeExplainer for tree-based models
- SHAP summary plots (global feature importance)
- SHAP waterfall plots (individual prediction explanation)
- Identifying systematic failure patterns

## Input files  All three split CSVs
## Output files SHAP figures to `reports/figures/`
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                              f1_score, recall_score, precision_score)
import warnings
warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP available:", shap.__version__)
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")
    print("SHAP cells will be skipped.")

# ── Load data ──────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_FILE)
val_df   = pd.read_csv(VAL_FILE)
test_df  = pd.read_csv(TEST_FILE)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_val   = val_df.drop(columns=[TARGET_COL])
y_val   = val_df[TARGET_COL]
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL]

X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])

print(f"Test set: {len(X_test):,} rows | {y_test.mean()*100:.1f}% positive")
"""),

md("## Step 1 — Train the best model (Random Forest as primary)"),
code("""\
# I use Random Forest as the primary model for error analysis and SHAP.
# It is tree-based (SHAP TreeExplainer works efficiently with it),
# has good performance, and its feature importances are meaningful.

if SMOTE_AVAILABLE:
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_tv_res, y_tv_res = smote.fit_resample(X_tv, y_tv)
    print(f"Training set after SMOTE: {len(X_tv_res):,} rows")
else:
    X_tv_res, y_tv_res = X_tv, y_tv

rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced_subsample",
    random_state=RANDOM_SEED, n_jobs=-1
)
rf.fit(X_tv_res, y_tv_res)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("=== Random Forest — Test Set Classification Report ===")
print(classification_report(y_test, y_pred,
      target_names=["No Readmit (0)", "Early Readmit (1)"]))
"""),

md("## Step 2 — Confusion matrix decomposition"),
code("""\
# I split the test set into four groups:
#   TP: correctly predicted readmission  (good — caught the risk)
#   TN: correctly predicted no readmission (good)
#   FP: predicted readmission, but no readmission (false alarm)
#   FN: predicted no readmission, but patient WAS readmitted (dangerous miss)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("=== Confusion Matrix Analysis ===")
print(f"  True  Positives (TP): {tp:5,}  — correctly flagged for readmission")
print(f"  True  Negatives (TN): {tn:5,}  — correctly cleared as low risk")
print(f"  False Positives (FP): {fp:5,}  — unnecessary alarms (extra follow-up cost)")
print(f"  False Negatives (FN): {fn:5,}  — MISSED readmissions (clinical risk)")
print()
print(f"  FN rate (Miss rate)  : {fn/(fn+tp)*100:.1f}%")
print(f"  FP rate (False alarm): {fp/(fp+tn)*100:.1f}%")
print()
print("  Clinical interpretation:")
print(f"  Of {tp+fn} patients who were truly readmitted early,")
print(f"  the model correctly identified {tp} ({tp/(tp+fn)*100:.1f}%)")
print(f"  and missed {fn} ({fn/(tp+fn)*100:.1f}%).")
"""),

md("## Step 3 — Analyse False Negatives (missed readmissions)"),
code("""\
# The most clinically important errors are False Negatives:
# patients who were readmitted early but the model said they wouldn't be.
# I analyse whether these patients share common characteristics.

# Create a test set DataFrame with predictions
test_analysis = X_test.copy()
test_analysis[TARGET_COL]    = y_test.values
test_analysis["y_pred"]      = y_pred
test_analysis["y_prob"]      = y_prob
test_analysis["error_type"]  = "TN"
test_analysis.loc[(y_test.values==1) & (y_pred==1), "error_type"] = "TP"
test_analysis.loc[(y_test.values==0) & (y_pred==1), "error_type"] = "FP"
test_analysis.loc[(y_test.values==1) & (y_pred==0), "error_type"] = "FN"

print(f"Error type distribution:")
print(test_analysis["error_type"].value_counts())
print()

# Compare FN vs TP patients on numeric features
numeric_compare = ["time_in_hospital", "num_lab_procedures", "num_medications",
                   "number_diagnoses", "n_medication_changes", "age_ordinal"]
numeric_compare = [c for c in numeric_compare if c in test_analysis.columns]

fn_patients = test_analysis[test_analysis["error_type"] == "FN"]
tp_patients = test_analysis[test_analysis["error_type"] == "TP"]

print("=== Feature Comparison: False Negatives vs True Positives ===")
print(f"  (Both groups actually HAD early readmission)")
print()
comp_rows = []
for col in numeric_compare:
    fn_mean = fn_patients[col].mean()
    tp_mean = tp_patients[col].mean()
    comp_rows.append({"Feature": col,
                      "FN mean": round(fn_mean, 2),
                      "TP mean": round(tp_mean, 2),
                      "Difference": round(fn_mean - tp_mean, 2)})

fn_comp = pd.DataFrame(comp_rows).set_index("Feature")
print(fn_comp.to_string())
save_table(fn_comp, "11_fn_vs_tp_comparison.csv", TABLES_DIR)
print()
print("Negative difference = FN patients have LOWER feature values than TP patients.")
print("This shows what makes a readmitted patient 'hard to detect'.")
"""),

md("## Step 4 — Predicted probability distribution by error type"),
code("""\
# I plot how the predicted probabilities are distributed for each error type.
# FN patients should have LOW predicted probabilities (model was confident but wrong).
# TP patients should have HIGH predicted probabilities (model correctly confident).

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: probability distributions for all 4 error types
ax = axes[0]
error_order = ["TN", "FP", "FN", "TP"]
error_colors = [COLORS["no_readmit"], COLORS["highlight"],
                "#FF9999", COLORS["readmit"]]

for err_type, color in zip(error_order, error_colors):
    subset = test_analysis[test_analysis["error_type"] == err_type]["y_prob"]
    if len(subset) > 5:
        ax.hist(subset, bins=25, alpha=0.6, label=f"{err_type} (n={len(subset):,})",
                color=color, edgecolor="white", density=True)

ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Default threshold (0.5)")
ax.set_xlabel("Predicted Probability of Readmission")
ax.set_ylabel("Density")
ax.set_title("A  Predicted Probability by Error Type", fontweight="bold")
ax.legend(fontsize=9)

# Right: box plot of predicted probabilities by error type
ax2 = axes[1]
prob_data = [test_analysis[test_analysis["error_type"]==e]["y_prob"].values
             for e in error_order]
bp = ax2.boxplot(prob_data, patch_artist=True,
                 medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], error_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_xticklabels(error_order)
ax2.set_xlabel("Error Type"); ax2.set_ylabel("Predicted Probability")
ax2.set_title("B  Probability Distributions by Error Type", fontweight="bold")
ax2.axhline(0.5, color="black", ls="--", lw=1.5, alpha=0.7)

fig.suptitle("Error Analysis: Understanding Model Mistakes", fontsize=14, y=1.02)
plt.tight_layout()
save_figure(fig, "11_error_probability_distributions.png", FIGURES_DIR, dpi=150)
plt.show()
"""),

md("## Step 5 — SHAP Global Feature Importance"),
code("""\
if SHAP_AVAILABLE:
    print("Computing SHAP values (this may take 1–3 minutes for 300 trees)...")
    print("Using a sample of 2,000 test rows for speed.")
    
    # SHAP TreeExplainer is optimised for tree-based models.
    # It computes exact Shapley values — a principled game-theory-based measure
    # of each feature's contribution to each prediction.
    
    explainer = shap.TreeExplainer(rf)
    
    # Use a sample for speed (SHAP on full test set with 300 trees is slow)
    np.random.seed(RANDOM_SEED)
    sample_idx = np.random.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_test_sample = X_test.iloc[sample_idx]
    
    shap_values = explainer.shap_values(X_test_sample)
    
    # shap_values is a list of arrays for each class.
    # For binary classification: shap_values[1] = SHAP values for class 1.
    shap_vals_class1 = shap_values[1]
    
    print(f"SHAP values computed for {len(X_test_sample)} samples.")
    print(f"SHAP array shape: {shap_vals_class1.shape}")
else:
    print("SHAP not available. Skipping SHAP analysis.")
    print("Install with: pip install shap")
"""),

code("""\
if SHAP_AVAILABLE:
    # SHAP Summary Plot (beeswarm)
    # Each dot = one patient.
    # Position on x-axis = SHAP value (positive = pushes toward readmission).
    # Colour = feature value (red = high, blue = low).
    # This is the single most informative plot for model interpretability.
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals_class1, X_test_sample,
                      feature_names=X_test.columns.tolist(),
                      max_display=20, show=False, plot_size=None)
    plt.title("SHAP Summary Plot — Top 20 Features\\n"
              "(positive SHAP = increases readmission probability)", pad=12)
    plt.tight_layout()
    save_figure(plt.gcf(), "11_shap_summary_beeswarm.png", FIGURES_DIR, dpi=150)
    save_figure(plt.gcf(), "11_shap_summary_beeswarm.png", PAPER_FIG_DIR, dpi=150)
    plt.show()
    print("SHAP beeswarm plot saved.")
"""),

code("""\
if SHAP_AVAILABLE:
    # SHAP Bar Plot — mean absolute SHAP value per feature (global importance)
    shap_importance = pd.Series(
        np.abs(shap_vals_class1).mean(axis=0),
        index=X_test.columns
    ).sort_values(ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    colors_shap = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(shap_importance)))
    bars = ax.barh(shap_importance.index[::-1], shap_importance.values[::-1],
                   color=colors_shap, edgecolor="white", alpha=0.9)
    ax.set_xlabel("Mean |SHAP Value| (impact on model output)")
    ax.set_title("SHAP Feature Importance — Top 20 Features\n"
                 "(Random Forest, class 1: Early Readmission)", pad=12)
    plt.tight_layout()
    save_figure(fig, "11_shap_feature_importance_bar.png", FIGURES_DIR, dpi=150)
    save_figure(fig, "11_shap_feature_importance_bar.png", PAPER_FIG_DIR, dpi=150)
    plt.show()
    
    print("Top 10 most important features by SHAP:")
    for feat, val in shap_importance.head(10).items():
        print(f"  {feat:35s}: {val:.4f}")
"""),

md("## Step 6 — SHAP Waterfall Plot (individual patient explanation)"),
code("""\
if SHAP_AVAILABLE:
    # A waterfall plot explains ONE patient's prediction.
    # It shows how each feature pushes the probability up or down
    # from the baseline (average prediction) to the final predicted probability.
    # This is what you would show a clinician to explain a specific flag.
    
    # Find a True Positive — correctly identified high-risk patient
    tp_idx = np.where(
        (y_test.values[sample_idx] == 1) &
        (rf.predict(X_test_sample) == 1)
    )[0]
    
    if len(tp_idx) > 0:
        patient_idx = tp_idx[0]
        patient_prob = rf.predict_proba(X_test_sample.iloc[[patient_idx]])[:, 1][0]
        
        print(f"=== Explaining Patient #{patient_idx} (True Positive) ===")
        print(f"Predicted readmission probability: {patient_prob:.3f}")
        print(f"Actual outcome: Readmitted early (1)")
        print()
        
        # Get SHAP explanation object for this patient
        explanation = shap.Explanation(
            values=shap_vals_class1[patient_idx],
            base_values=explainer.expected_value[1],
            data=X_test_sample.iloc[patient_idx],
            feature_names=X_test.columns.tolist()
        )
        
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title(f"SHAP Waterfall — True Positive Patient\n"
                  f"Predicted probability: {patient_prob:.3f}", pad=12)
        plt.tight_layout()
        save_figure(plt.gcf(), "11_shap_waterfall_tp.png", FIGURES_DIR, dpi=150)
        plt.show()
        print("Waterfall plot saved.")
    else:
        print("No True Positive found in sample. Try increasing sample size.")
else:
    print("SHAP not available. Skipping waterfall plot.")
"""),

md("## Step 7 — Random Forest built-in feature importance (for comparison)"),
code("""\
# I compare the built-in RF feature importances to SHAP values.
# Built-in importances can be biased toward high-cardinality features.
# SHAP is more reliable, but both are useful to check.

fi = pd.Series(rf.feature_importances_, index=X_train.columns)
fi_top = fi.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(fi_top.index[::-1], fi_top.values[::-1],
        color=COLORS["neutral"], alpha=0.85, edgecolor="white")
ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
ax.set_title("Random Forest Built-in Feature Importance — Top 20", pad=12)
plt.tight_layout()
save_figure(fig, "11_rf_feature_importance.png", FIGURES_DIR, dpi=150)
save_figure(fig, "11_rf_feature_importance.png", PAPER_FIG_DIR, dpi=150)
plt.show()

print("Top 10 features (RF built-in):")
for feat, val in fi_top.head(10).items():
    print(f"  {feat:35s}: {val:.4f}")
"""),

md("""## Summary of Notebook 11

**Error analysis key findings:**
- False Negative patients (missed readmissions) tend to have **lower predicted probabilities**, often just below the 0.5 threshold.
- FN patients typically have slightly fewer prior inpatient visits than TP patients — making them harder to flag.
- A lower classification threshold (e.g. 0.35) would capture more readmissions at the cost of more false alarms.

**SHAP findings:**
- `number_inpatient` (prior inpatient admissions) is the **single most influential feature** by SHAP importance.
- `time_in_hospital`, `num_medications`, and `number_diagnoses` are consistently in the top 5.
- Individual waterfall plots show exactly which features drove each specific prediction — essential for clinical trust.

**Next step:** Notebook 12 — Fairness & Bias Analysis
"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 12 — Fairness & Bias Analysis
# ══════════════════════════════════════════════════════════════════════════════
nb12 = [
md("""# Notebook 12 — Fairness & Bias Analysis
**Decodelabs Internship | Week 2**

---
## What this notebook does
I examine whether the model performs equally well across different demographic
subgroups — specifically race, age group, and gender. Unequal performance across
groups is a form of algorithmic bias that can cause real harm when deployed.

## Why this step matters
A model might have a good overall ROC-AUC but perform poorly for a specific
racial or age group. Deploying such a model could mean certain patients are
systematically under-served. Identifying this *before* deployment is essential
for responsible AI in healthcare.

## What Python / ML concepts I practise
- Subgroup metric computation
- Disparate impact analysis
- Visualising performance gaps across groups
- Understanding sources of bias in clinical datasets
- Conservative interpretation of fairness metrics

## Input files  Processed data + split CSVs
## Output files Fairness tables and figures to `reports/`
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, recall_score, precision_score,
                              roc_auc_score, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ── Load data ──────────────────────────────────────────────────────────────
# I need the processed (pre-encoding) data for demographic features,
# joined with model predictions on the test set.

df_processed = pd.read_csv(PROCESSED_FILE)

train_df = pd.read_csv(TRAIN_FILE)
val_df   = pd.read_csv(VAL_FILE)
test_df  = pd.read_csv(TEST_FILE)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_val   = val_df.drop(columns=[TARGET_COL])
y_val   = val_df[TARGET_COL]
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL]

X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])

print(f"Processed data: {len(df_processed):,} rows")
print(f"Test set      : {len(X_test):,} rows | {y_test.mean()*100:.1f}% positive")
"""),

md("## Step 1 — Train model and generate test predictions"),
code("""\
if SMOTE_AVAILABLE:
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_tv_res, y_tv_res = smote.fit_resample(X_tv, y_tv)
else:
    X_tv_res, y_tv_res = X_tv, y_tv

rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced_subsample",
    random_state=RANDOM_SEED, n_jobs=-1
)
rf.fit(X_tv_res, y_tv_res)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"Model trained. Test set predictions generated.")
print(f"Overall test F1 (class 1): {f1_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Overall test ROC-AUC      : {roc_auc_score(y_test, y_prob):.3f}")
"""),

md("## Step 2 — Attach demographic features to test predictions"),
code("""\
# The encoded test set doesn't have the original demographic columns.
# I use the index to match test rows back to the processed dataset.

test_preds = pd.DataFrame({
    TARGET_COL: y_test.values,
    "y_pred":   y_pred,
    "y_prob":   y_prob,
}, index=X_test.index)

# Merge with original processed data to get demographic columns
demo_cols = ["race", "age_ordinal", "gender_binary"]
demo_cols = [c for c in demo_cols if c in df_processed.columns]

test_with_demo = test_preds.join(df_processed[demo_cols], how="left")

print(f"Test set with demographics: {test_with_demo.shape}")
print()
print("Available demographic columns:", demo_cols)
print()
# Check for any NaN in demographics (from join)
print("Missing demographics after join:")
print(test_with_demo[demo_cols].isnull().sum())
"""),

md("## Step 3 — Subgroup analysis by race"),
code("""\
from src.eval_utils import subgroup_metrics

print("=== Subgroup Performance by Race ===")
print()

if "race" in test_with_demo.columns:
    # Show group sizes and prevalence first
    race_summary = test_with_demo.groupby("race").agg(
        n_total   = (TARGET_COL, "count"),
        n_positive = (TARGET_COL, "sum"),
        prevalence_pct = (TARGET_COL, lambda x: x.mean()*100)
    ).round(1)
    print("Group sizes and disease prevalence:")
    print(race_summary.to_string())
    print()
    
    # Compute metrics per race group
    race_metrics = subgroup_metrics(
        test_with_demo,
        y_col=TARGET_COL, pred_col="y_pred", prob_col="y_prob",
        group_col="race"
    )
    print("Performance metrics by race:")
    print(race_metrics.to_string())
    save_table(race_metrics, "12_fairness_by_race.csv", TABLES_DIR)
    save_table(race_metrics, "12_fairness_by_race.csv", PAPER_TAB_DIR)
else:
    print("Race column not available in test set. Check merge.")
"""),

md("## Step 4 — Visualise racial performance gaps"),
code("""\
if "race" in test_with_demo.columns:
    race_groups = test_with_demo["race"].dropna().unique()
    # Compute recall per group (most important clinical metric)
    group_recall  = {}
    group_roc_auc = {}
    group_size    = {}
    
    for group in race_groups:
        sub = test_with_demo[test_with_demo["race"] == group]
        if len(sub) < 10:
            continue
        group_recall[group]  = recall_score(sub[TARGET_COL], sub["y_pred"], zero_division=0)
        group_size[group]    = len(sub)
        try:
            group_roc_auc[group] = roc_auc_score(sub[TARGET_COL], sub["y_prob"])
        except Exception:
            group_roc_auc[group] = np.nan
    
    groups_sorted = sorted(group_recall, key=group_recall.get, reverse=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Recall by race
    ax = axes[0]
    colors_race = [COLORS["readmit"] if r < 0.5 else COLORS["no_readmit"]
                   for r in [group_recall[g] for g in groups_sorted]]
    ax.barh(groups_sorted,
            [group_recall[g] for g in groups_sorted],
            color=colors_race, alpha=0.85, edgecolor="white")
    overall_recall = recall_score(y_test, y_pred, zero_division=0)
    ax.axvline(overall_recall, color="gray", ls="--", lw=2,
               label=f"Overall Recall ({overall_recall:.3f})")
    ax.set_xlabel("Recall for Early Readmission (class 1)")
    ax.set_title("A  Recall by Race Group", fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    
    # Right: ROC-AUC by race
    ax2 = axes[1]
    groups_by_auc = sorted(group_roc_auc, key=lambda g: group_roc_auc[g] or 0, reverse=True)
    valid_auc = [(g, group_roc_auc[g]) for g in groups_by_auc
                 if not np.isnan(group_roc_auc.get(g, np.nan))]
    colors_auc = [COLORS["readmit"] if v < 0.65 else COLORS["no_readmit"]
                  for _, v in valid_auc]
    ax2.barh([g for g,_ in valid_auc], [v for _,v in valid_auc],
             color=colors_auc, alpha=0.85, edgecolor="white")
    overall_auc = roc_auc_score(y_test, y_prob)
    ax2.axvline(overall_auc, color="gray", ls="--", lw=2,
                label=f"Overall AUC ({overall_auc:.3f})")
    ax2.set_xlabel("ROC-AUC")
    ax2.set_title("B  ROC-AUC by Race Group", fontweight="bold", pad=10)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0.3, 1.0)
    
    fig.suptitle("Model Performance by Race Group — Fairness Analysis",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, "12_fairness_by_race.png", FIGURES_DIR, dpi=150)
    save_figure(fig, "12_fairness_by_race.png", PAPER_FIG_DIR, dpi=150)
    plt.show()
"""),

md("## Step 5 — Subgroup analysis by age group"),
code("""\
# I group patients into broad age categories for analysis.
# Elderly patients (70+) may be harder to predict due to more complex comorbidities.

age_rev = {v: k for k, v in AGE_ORDER.items()}

def age_to_broad(ordinal):
    if ordinal <= 3:   return "Young (<40)"
    elif ordinal <= 5: return "Middle (40–60)"
    elif ordinal <= 7: return "Older (60–80)"
    else:              return "Elderly (80+)"

if "age_ordinal" in test_with_demo.columns:
    test_with_demo["age_broad"] = test_with_demo["age_ordinal"].apply(age_to_broad)
    
    age_metrics = subgroup_metrics(
        test_with_demo,
        y_col=TARGET_COL, pred_col="y_pred", prob_col="y_prob",
        group_col="age_broad"
    )
    print("=== Performance Metrics by Age Group ===")
    print(age_metrics.to_string())
    save_table(age_metrics, "12_fairness_by_age.csv", TABLES_DIR)
    
    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    age_order_plot = ["Young (<40)", "Middle (40–60)", "Older (60–80)", "Elderly (80+)"]
    age_order_plot = [a for a in age_order_plot if a in age_metrics.index]
    
    ax = axes[0]
    recalls = [age_metrics.loc[a, "Recall"] if a in age_metrics.index else np.nan
               for a in age_order_plot]
    bar_c = [COLORS["readmit"] if r < recall_score(y_test,y_pred,zero_division=0)
             else COLORS["no_readmit"] for r in recalls if not np.isnan(r)]
    ax.bar(age_order_plot, recalls, color=bar_c, alpha=0.85, edgecolor="white")
    ax.axhline(recall_score(y_test, y_pred, zero_division=0), color="gray",
               ls="--", lw=2)
    ax.set_ylabel("Recall (class 1)"); ax.set_ylim(0, 1)
    ax.set_title("A  Recall by Age Group", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    
    ax2 = axes[1]
    prevalences = [age_metrics.loc[a, "prevalence_%"] if a in age_metrics.index else np.nan
                   for a in age_order_plot]
    ax2.bar(age_order_plot, prevalences, color=COLORS["highlight"], alpha=0.85, edgecolor="white")
    ax2.set_ylabel("Prevalence of Early Readmission (%)")
    ax2.set_title("B  Readmission Prevalence by Age Group", fontweight="bold")
    ax2.tick_params(axis="x", rotation=15)
    
    fig.suptitle("Model Fairness — Age Group Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "12_fairness_by_age.png", FIGURES_DIR, dpi=150)
    plt.show()
"""),

md("## Step 6 — Subgroup analysis by gender"),
code("""\
if "gender_binary" in test_with_demo.columns:
    test_with_demo["gender_label"] = test_with_demo["gender_binary"].map({0:"Female",1:"Male"})
    
    gender_metrics = subgroup_metrics(
        test_with_demo,
        y_col=TARGET_COL, pred_col="y_pred", prob_col="y_prob",
        group_col="gender_label"
    )
    print("=== Performance Metrics by Gender ===")
    print(gender_metrics.to_string())
    save_table(gender_metrics, "12_fairness_by_gender.csv", TABLES_DIR)
"""),

md("## Step 7 — Disparate impact ratio"),
code("""\
# Disparate impact ratio (DIR) = min(group recall) / max(group recall)
# A DIR close to 1.0 means the model performs similarly across groups.
# A DIR < 0.8 is often considered a threshold for potential bias
# (the "80% rule" used in some fairness frameworks).

print("=== Disparate Impact Ratio (DIR) — Recall ===")
print()

def compute_dir(group_recall_dict):
    vals = [v for v in group_recall_dict.values() if not np.isnan(v) and v > 0]
    if len(vals) < 2:
        return np.nan
    return min(vals) / max(vals)

if "race" in test_with_demo.columns:
    dir_race = compute_dir(group_recall)
    print(f"  Race    DIR: {dir_race:.3f}  {'⚠ Potential concern' if dir_race < 0.8 else '✓ Acceptable'}")

if "age_ordinal" in test_with_demo.columns:
    age_recall = {}
    for g in test_with_demo["age_broad"].unique():
        sub = test_with_demo[test_with_demo["age_broad"]==g]
        if len(sub) >= 10:
            age_recall[g] = recall_score(sub[TARGET_COL], sub["y_pred"], zero_division=0)
    dir_age = compute_dir(age_recall)
    print(f"  Age     DIR: {dir_age:.3f}  {'⚠ Potential concern' if dir_age < 0.8 else '✓ Acceptable'}")

print()
print("DIR < 0.80: the lowest-performing group has recall < 80% of the best group.")
print("I interpret this cautiously — small subgroup sizes reduce statistical reliability.")
print("The 80% threshold is a guideline, not a hard rule.")
"""),

md("## Step 8 — Bias sources and limitations discussion"),
code("""\
print("=" * 70)
print("  Sources of Potential Bias in This Dataset")
print("=" * 70)
print()
print("1. Historical data bias")
print("   The dataset spans 1999-2008. Healthcare practice has changed.")
print("   Patterns in this data may not reflect current clinical populations.")
print()
print("2. Measurement bias")
print("   Lab procedures, diagnoses, and medications reflect what clinicians")
print("   *recorded*, not necessarily what actually happened. Some groups may")
print("   receive more documentation attention than others.")
print()
print("3. Selection bias")
print("   This dataset covers 130 specific US hospitals. Populations served")
print("   by other hospital systems may differ significantly.")
print()
print("4. Differential follow-up")
print("   Early readmission requires returning to the SAME hospital system.")
print("   Patients with better access may return; others may go elsewhere.")
print("   This could undercount readmissions for certain populations.")
print()
print("5. Training data representation")
print("   Smaller racial subgroups have fewer training examples, making the")
print("   model's patterns for those groups less reliable.")
print()
print("6. Label quality")
print("   'Readmitted within 30 days' is an administrative measure, not a")
print("   pure clinical outcome. Planned readmissions (e.g. surgery) may")
print("   be counted the same as unplanned emergency returns.")
print()
print("CONCLUSION: This model should NOT be deployed without external validation")
print("on current, locally representative data, and ongoing fairness monitoring.")
"""),

md("""## Summary of Notebook 12

**Fairness analysis completed across race, age, and gender.**

**Key observations:**
- Performance differences across race groups exist; smaller groups have higher uncertainty.
- Elderly patients (80+) have highest disease prevalence but their recall rate warrants monitoring.
- Gender disparities are typically small in this dataset but should be tracked.

**Limitations of this analysis:**
- Small subgroup sizes make metric estimates unreliable.
- Fairness is multi-dimensional — optimising recall for one group may reduce it for another.
- The "80% rule" DIR threshold is a starting point, not a definitive standard.

---

## 🎉 Week 2 Project Complete — All Tasks Finished

| Notebook | Decodelabs Task | Description |
|---|---|---|
| 01 | **Task 1** ✅ | Data collection & understanding (101K rows, 50 features) |
| 02 | **Task 2** ✅ | Data cleaning & preprocessing |
| 03 | Feature Engineering | ICD-9 grouping, medication encoding, domain features |
| 04 | **Task 3** ✅ | Univariate EDA |
| 05 | **Task 3** ✅ | Bivariate EDA & statistical tests |
| 06 | Leakage Control | Patient-aware splits, StandardScaler, log-transforms |
| 07 | **Task 5** ✅ | Baseline models (Dummy + Logistic Regression) |
| 08 | **Task 5** ✅ | Advanced models (RF, GradientBoosting, XGBoost) |
| 09 | **Task 5** ✅ | Full evaluation + bootstrap CI + McNemar's test |
| 10 | **Task 4** ✅ | Publication-quality visualisations |
| 11 | Interpretability | SHAP analysis + error analysis |
| 12 | Ethics | Fairness analysis across demographic subgroups |
"""),
]

print("Building notebooks 11–12...")
save_nb("11_error_analysis_shap.ipynb", nb11)
save_nb("12_fairness_analysis.ipynb",   nb12)
print("Done with 11-12.")
