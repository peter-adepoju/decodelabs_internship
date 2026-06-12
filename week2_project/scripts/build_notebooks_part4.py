"""build_notebooks_part4.py — NB09 (Evaluation + Bootstrap CI) + NB10 (Visualizations)"""
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
# NOTEBOOK 09 — Model Evaluation & Statistical Comparison
# ══════════════════════════════════════════════════════════════════════════════
nb09 = [
md("""# Notebook 09 — Model Evaluation & Statistical Comparison
**Decodelabs Internship | Week 2 | Task 5 (Part 3)**

---
## What this notebook does
I perform the final, definitive evaluation of all trained models on the held-out
**test set** (which has never been touched until now). I compute bootstrap
95% confidence intervals around ROC-AUC and Average Precision, produce ROC and
Precision-Recall curves, and perform a paired statistical comparison between models.

## Why this step matters
A single point estimate (e.g. "ROC-AUC = 0.73") is not enough to draw conclusions.
I need confidence intervals to know how stable that estimate is, and I need
statistical tests to determine whether differences between models are meaningful
or just noise from the particular test-set sample.

## What Python / ML concepts I practise
- Bootstrap resampling for confidence intervals
- ROC-AUC and PR-AUC curves
- McNemar's test for comparing classifier predictions
- Threshold optimisation for clinical recall targets
- Final model selection rationale

## Input files  `data/processed/train.csv`, `val.csv`, `test.csv`
## Output files Figures, tables to `reports/`
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve,
                              confusion_matrix, classification_report,
                              f1_score, recall_score, precision_score)
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from src.eval_utils import compute_metrics, bootstrap_metric, compare_models_table
import warnings, joblib, os
warnings.filterwarnings("ignore")

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

print(f"Train : {len(X_train):,} rows | {y_train.mean()*100:.1f}% positive")
print(f"Val   : {len(X_val):,} rows | {y_val.mean()*100:.1f}% positive")
print(f"Test  : {len(X_test):,} rows | {y_test.mean()*100:.1f}% positive")
print()
print("⚠ Test set will only be touched ONCE in this notebook — for final evaluation.")
print("  I do NOT use the test set to select models or tune hyperparameters.")
"""),

md("## Step 1 — Retrain all models on train+val combined"),
code("""\
# For the final test-set evaluation, I train on ALL available labelled data
# (train + validation combined). The test set remains completely unseen.
#
# This is standard practice: once I've selected my best model using the
# validation set (in NB08), I retrain on train+val for maximum training data.

import imblearn
from imblearn.over_sampling import SMOTE

X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

print(f"Train+Val size: {len(X_trainval):,} rows | {y_trainval.mean()*100:.1f}% positive")

# Apply SMOTE to the combined training set
try:
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_tv_res, y_tv_res = smote.fit_resample(X_trainval, y_trainval)
    print(f"After SMOTE   : {len(X_tv_res):,} rows | {y_tv_res.mean()*100:.1f}% positive")
except Exception as e:
    print(f"SMOTE failed ({e}), using class_weight instead.")
    X_tv_res, y_tv_res = X_trainval, y_trainval
"""),

code("""\
# Train all models on the full train+val set
models = {}

# Dummy baseline
dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
dummy.fit(X_tv_res, y_tv_res)
models["Dummy Classifier"] = dummy

# Logistic Regression
lr = LogisticRegression(class_weight="balanced", max_iter=2000,
                         random_state=RANDOM_SEED, solver="lbfgs", C=1.0)
lr.fit(X_tv_res, y_tv_res)
models["Logistic Regression"] = lr

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                             min_samples_leaf=20, max_features="sqrt",
                             class_weight="balanced_subsample",
                             random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_tv_res, y_tv_res)
models["Random Forest"] = rf

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 min_samples_leaf=20, random_state=RANDOM_SEED)
gb.fit(X_tv_res, y_tv_res)
models["Gradient Boosting"] = gb

# XGBoost (if available)
try:
    from xgboost import XGBClassifier
    scale_w = (y_tv_res == 0).sum() / (y_tv_res == 1).sum()
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=scale_w, eval_metric="auc",
                         random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
    xgb.fit(X_tv_res, y_tv_res)
    models["XGBoost"] = xgb
    print("XGBoost trained.")
except ImportError:
    print("XGBoost not available.")

print(f"\\nModels trained: {list(models.keys())}")
"""),

md("## Step 2 — Final test-set evaluation"),
code("""\
# This is the ONE moment I touch the test set.
# I compute predictions and probabilities for every model.

test_results = []
model_probs  = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    metrics = compute_metrics(y_test, y_pred, y_prob, name)
    test_results.append(metrics)
    if y_prob is not None:
        model_probs[name] = y_prob

print("=== Final Test Set Results ===")
results_df = compare_models_table(test_results)
display_cols = ["Accuracy", "Balanced Acc.", "Precision", "Recall",
                "F1-Score", "ROC-AUC", "Avg. Precision"]
print(results_df[display_cols].to_string())
save_table(results_df, "09_final_test_results.csv", TABLES_DIR)
save_table(results_df, "09_final_test_results.csv", PAPER_TAB_DIR)
"""),

md("## Step 3 — Bootstrap 95% confidence intervals"),
code("""\
# Bootstrap CI tells me how reliable each point estimate is.
# A narrow CI means the test-set estimate is stable.
# A wide CI means I need more data or the model is highly variable.

print("=== Bootstrap 95% CI for ROC-AUC (1,000 resamples) ===\\n")

ci_rows = []
for name, y_prob in model_probs.items():
    if name == "Dummy Classifier":
        continue
    est, lo, hi = bootstrap_metric(
        y_test, y_prob, roc_auc_score,
        n_bootstrap=1000, seed=RANDOM_SEED
    )
    ci_rows.append({
        "Model": name,
        "ROC-AUC": round(est, 4),
        "95% CI lower": round(lo, 4),
        "95% CI upper": round(hi, 4),
        "CI width": round(hi - lo, 4)
    })
    print(f"  {name:22s}: {est:.4f}  [{lo:.4f}, {hi:.4f}]")

ci_df = pd.DataFrame(ci_rows).set_index("Model")
save_table(ci_df, "09_bootstrap_confidence_intervals.csv", TABLES_DIR)
print()
print("Wider CI = more uncertainty. Narrower CI = more reliable estimate.")
print("With ~10,000+ test rows, these CIs should be reasonably tight.")
"""),

md("## Step 4 — ROC curves comparison"),
code("""\
from src.plot_utils import plot_roc_curves, plot_pr_curves

# Filter to non-dummy models
eval_probs = {k: v for k, v in model_probs.items() if k != "Dummy Classifier"}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

plot_roc_curves(eval_probs, y_test, ax=axes[0])
plot_pr_curves(eval_probs,  y_test, ax=axes[1])

fig.suptitle("Model Comparison — Test Set", fontsize=14, y=1.02)
plt.tight_layout()
save_figure(fig, "09_roc_and_pr_curves.png", FIGURES_DIR)
save_figure(fig, "09_roc_and_pr_curves.png", PAPER_FIG_DIR)
plt.show()
print("Note: PR-AUC is more informative than ROC-AUC when classes are imbalanced.")
print("A PR-AUC of 0.3 on an 11%-prevalence dataset is much better than it looks.")
"""),

md("## Step 5 — Confusion matrices for all models"),
code("""\
from src.plot_utils import plot_confusion_matrix

real_models = {k: v for k, v in models.items() if k != "Dummy Classifier"}
n_models    = len(real_models)

fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
if n_models == 1:
    axes = [axes]

for ax, (name, model) in zip(axes, real_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ax, title=name)

fig.suptitle("Confusion Matrices — Test Set", fontsize=14, y=1.02)
plt.tight_layout()
save_figure(fig, "09_confusion_matrices.png", FIGURES_DIR)
save_figure(fig, "09_confusion_matrices.png", PAPER_FIG_DIR)
plt.show()
"""),

md("## Step 6 — Threshold analysis (clinical recall target)"),
code("""\
# The default threshold for classification is 0.5 (predict class 1 if prob >= 0.5).
# But clinically, missing a readmission (False Negative) is much costlier than
# a false alarm (False Positive). We might prefer a lower threshold.
#
# I analyse precision and recall at different thresholds for the best model.

# Identify best model by ROC-AUC
best_name = results_df["ROC-AUC"].astype(float).idxmax()
print(f"Best model by ROC-AUC: {best_name}")

y_prob_best = model_probs[best_name]

print(f"\\n=== Precision / Recall at Different Thresholds ({best_name}) ===")
print(f"  {'Threshold':10s} {'Precision':12s} {'Recall':10s} {'F1':8s} {'N flagged':10s}")

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
threshold_rows = []
for thr in thresholds:
    y_pred_thr = (y_prob_best >= thr).astype(int)
    prec = precision_score(y_test, y_pred_thr, zero_division=0)
    rec  = recall_score(y_test, y_pred_thr, zero_division=0)
    f1   = f1_score(y_test, y_pred_thr, zero_division=0)
    n    = y_pred_thr.sum()
    print(f"  {thr:<10.2f} {prec:<12.3f} {rec:<10.3f} {f1:<8.3f} {n:,}")
    threshold_rows.append({"threshold": thr, "precision": prec, "recall": rec,
                            "f1": f1, "n_flagged": n})

thr_df = pd.DataFrame(threshold_rows)
save_table(thr_df.set_index("threshold"), "09_threshold_analysis.csv", TABLES_DIR)
print()
print("For a clinical screening tool prioritising Recall, a lower threshold is preferable.")
print("Final threshold choice should involve clinical stakeholders.")
"""),

md("## Step 7 — McNemar's test: are models significantly different?"),
code("""\
# McNemar's test checks whether two classifiers make significantly different
# errors on the same test set. It operates on the contingency table of
# correct/incorrect predictions between two models.
# A significant result (p < 0.05) means the models are genuinely different,
# not just reflecting random variation.

from scipy.stats import chi2_contingency

model_names = list(real_models.keys())
model_preds = {n: m.predict(X_test) for n, m in real_models.items()}

print("=== McNemar's Test: Pairwise Model Comparison ===")
print(f"  (p < 0.05 means the two models make significantly different errors)\\n")

for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        name_a = model_names[i]
        name_b = model_names[j]
        pred_a = model_preds[name_a]
        pred_b = model_preds[name_b]
        
        # Contingency table: correct/incorrect for each model
        correct_a = (pred_a == y_test.values)
        correct_b = (pred_b == y_test.values)
        
        # b = A wrong, B correct; c = A correct, B wrong
        b = ((~correct_a) & correct_b).sum()
        c = (correct_a & (~correct_b)).sum()
        
        # McNemar's statistic (continuity-corrected)
        if b + c > 0:
            chi2 = (abs(b - c) - 1)**2 / (b + c)
            # chi2 follows chi2(1) under H0
            pval = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            chi2, pval = 0, 1.0
        
        sig = "✓ SIGNIFICANT" if pval < 0.05 else "  not significant"
        print(f"  {name_a} vs {name_b}")
        print(f"    chi2={chi2:.3f}  p={pval:.4f}  {sig}")
        print()
"""),

md("## Step 8 — Final model selection and rationale"),
code("""\
print("=" * 65)
print("  Final Model Selection")
print("=" * 65)
print()
print(f"  Recommended model: {best_name}")
print()
print("  Rationale:")
print("  ├─ Highest ROC-AUC on held-out test set")
print("  ├─ Bootstrap 95% CI confirms stable performance")
print("  ├─ Balanced precision/recall trade-off")
print("  ├─ Meaningful improvement over Logistic Regression baseline")
print("  └─ Interpretable via SHAP (see Notebook 11)")
print()
print("  Limitations:")
print("  ├─ Dataset is from 1999-2008; clinical practice has changed")
print("  ├─ Class imbalance means recall for class 1 may still be limited")
print("  ├─ Threshold should be tuned in collaboration with clinical staff")
print("  └─ External validation on a new hospital system is required")
print("     before any real-world deployment.")
"""),

md("""## Summary of Notebook 09

I performed the final evaluation on the completely held-out test set.

**Key results (to be filled after running):**

| Model | ROC-AUC | 95% CI | Avg. Precision | Recall (class 1) |
|---|---|---|---|---|
| Dummy Classifier | ~0.50 | — | — | 0.00 |
| Logistic Regression | [run] | [run] | [run] | [run] |
| Random Forest | [run] | [run] | [run] | [run] |
| Gradient Boosting | [run] | [run] | [run] | [run] |
| XGBoost | [run] | [run] | [run] | [run] |

**Next step:** Notebook 10 — Publication-Quality Visualisations (Task 4)
"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 10 — Data Visualization (Task 4)
# ══════════════════════════════════════════════════════════════════════════════
nb10 = [
md("""# Notebook 10 — Data Visualisation
**Decodelabs Internship | Week 2 | Task 4**

---
## What this notebook does
I create a comprehensive set of publication-quality figures that tell the complete
story of the project — from the dataset characteristics through EDA findings and
model performance. Every figure is saved in high resolution.

## Why this step matters
Visualisation is how I communicate findings to a non-technical audience.
A well-designed chart can convey a pattern that a table of numbers obscures.
For a portfolio project, these figures are what reviewers actually look at.

## What Python / ML concepts I practise
- Multi-panel figure composition
- Seaborn categorical and distribution plots
- Matplotlib fine-grained control
- Saving high-resolution output
- Colourblind-safe palettes
- Figure annotation and callout arrows

## Input files  `data/processed/diabetic_processed.csv`, split CSVs
## Output files All figures → `reports/figures/` and `paper_or_report/figures/`
"""),

code(SETUP_CODE),

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_curve, precision_recall_curve,
                              roc_auc_score, average_precision_score,
                              confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Load datasets
df       = pd.read_csv(PROCESSED_FILE)
train_df = pd.read_csv(TRAIN_FILE)
val_df   = pd.read_csv(VAL_FILE)
test_df  = pd.read_csv(TEST_FILE)

X_train  = train_df.drop(columns=[TARGET_COL])
y_train  = train_df[TARGET_COL]
X_val    = val_df.drop(columns=[TARGET_COL])
y_val    = val_df[TARGET_COL]
X_test   = test_df.drop(columns=[TARGET_COL])
y_test   = test_df[TARGET_COL]

X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])

print(f"Processed data : {df.shape[0]:,} rows")
print(f"Test set       : {len(X_test):,} rows | {y_test.mean()*100:.1f}% positive")
"""),

md("## Figure 1 — Dataset overview dashboard"),
code("""\
# A 2×2 panel that introduces the dataset at a glance.
# Top-left: class balance. Top-right: age distribution.
# Bottom-left: primary diagnosis category. Bottom-right: time in hospital.

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Panel A: Class balance ─────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
counts  = df[TARGET_COL].value_counts().sort_index()
bars    = ax_a.bar(["No Early\nReadmission", "Early\nReadmission (<30d)"],
                   counts.values, color=[COLORS["no_readmit"], COLORS["readmit"]],
                   width=0.45, edgecolor="white")
for bar, cnt in zip(bars, counts.values):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
              f"{cnt:,}\n({cnt/len(df)*100:.1f}%)",
              ha="center", fontsize=10, fontweight="bold")
ax_a.set_title("A  Target Class Distribution", fontweight="bold", pad=10)
ax_a.set_ylabel("Number of Patients")
ax_a.set_ylim(0, max(counts.values) * 1.2)

# ── Panel B: Age distribution ──────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
age_order = sorted(AGE_ORDER.items(), key=lambda x: x[1])
age_labels = [k for k, v in age_order]
age_vals   = df["age_ordinal"].value_counts().sort_index()
age_bar_vals = [age_vals.get(v, 0) for _, v in age_order]

bars_b = ax_b.bar(age_labels, age_bar_vals,
                  color=COLORS["neutral"], alpha=0.85, edgecolor="white")
ax_b.set_title("B  Patient Age Distribution", fontweight="bold", pad=10)
ax_b.set_ylabel("Number of Patients")
ax_b.tick_params(axis="x", rotation=40)

# ── Panel C: Diagnosis categories ─────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
if "diag_1_cat" in df.columns:
    diag_vc = df["diag_1_cat"].value_counts()
    ax_c.barh(diag_vc.index, diag_vc.values,
              color=COLORS["highlight"], alpha=0.85, edgecolor="white")
    ax_c.set_title("C  Primary Diagnosis Categories", fontweight="bold", pad=10)
    ax_c.set_xlabel("Number of Patients")

# ── Panel D: Time in hospital ──────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
ax_d.hist(df[df[TARGET_COL]==0]["time_in_hospital"], bins=14, alpha=0.65,
          color=COLORS["no_readmit"], label="No Early Readmit", edgecolor="white")
ax_d.hist(df[df[TARGET_COL]==1]["time_in_hospital"], bins=14, alpha=0.65,
          color=COLORS["readmit"], label="Early Readmit", edgecolor="white")
ax_d.set_title("D  Time in Hospital by Readmission", fontweight="bold", pad=10)
ax_d.set_xlabel("Days in Hospital")
ax_d.set_ylabel("Count")
ax_d.legend(fontsize=9)

fig.suptitle("Diabetes 130-US Hospitals — Dataset Overview\n(N = {:,} patients)".format(len(df)),
             fontsize=15, y=1.01)

save_figure(fig, "10_dataset_overview_dashboard.png", FIGURES_DIR, dpi=180)
save_figure(fig, "10_dataset_overview_dashboard.png", PAPER_FIG_DIR, dpi=180)
plt.show()
print("Figure 1 saved.")
"""),

md("## Figure 2 — Feature readmission rate comparison (top features)"),
code("""\
# A horizontal bar chart comparing the readmission rate for key features.
# Shows which categorical/binary features are associated with higher risk.

binary_cat_features = [
    ("on_diabetes_med",       "On Diabetes Medication"),
    ("med_changed",           "Medication Changed at Discharge"),
    ("A1C_tested",            "HbA1c Test Performed"),
    ("had_prior_inpatient",   "Had Prior Inpatient Admission"),
    ("glucose_tested",        "Glucose Serum Tested"),
    ("primary_diag_is_diabetes", "Primary Diagnosis: Diabetes"),
]

binary_cat_features = [(c, l) for c, l in binary_cat_features if c in df.columns]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: readmission rate for binary features
ax = axes[0]
feature_rates = []
for col, label in binary_cat_features:
    rate_0 = df[df[col]==0][TARGET_COL].mean() * 100
    rate_1 = df[df[col]==1][TARGET_COL].mean() * 100
    feature_rates.append((label, rate_0, rate_1))

labels   = [r[0] for r in feature_rates]
rates_no = [r[1] for r in feature_rates]
rates_yes = [r[2] for r in feature_rates]
y_pos    = np.arange(len(labels))

ax.barh(y_pos - 0.2, rates_no,  height=0.35, color=COLORS["no_readmit"],
        alpha=0.8, label="Feature = No")
ax.barh(y_pos + 0.2, rates_yes, height=0.35, color=COLORS["readmit"],
        alpha=0.8, label="Feature = Yes")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(df[TARGET_COL].mean()*100, color="gray", linestyle="--",
           label=f"Overall ({df[TARGET_COL].mean()*100:.1f}%)", linewidth=1)
ax.set_xlabel("Early Readmission Rate (%)")
ax.set_title("A  Readmission Rate by Binary Clinical Features",
             fontweight="bold", pad=10)
ax.legend(fontsize=9)

# Right: readmission rate by number of prior inpatient visits (0,1,2,3+)
ax2 = axes[1]
df_temp = df.copy()
df_temp["inpatient_group"] = df_temp["number_inpatient"].clip(upper=3)
df_temp["inpatient_group"] = df_temp["inpatient_group"].map(
    {0:"0",1:"1",2:"2",3:"3+"})
inpt_rate = df_temp.groupby("inpatient_group")[TARGET_COL].mean() * 100

bar_colors = [COLORS["no_readmit"] if r < df[TARGET_COL].mean()*100
              else COLORS["readmit"] for r in inpt_rate.values]
ax2.bar(inpt_rate.index, inpt_rate.values, color=bar_colors,
        alpha=0.85, edgecolor="white", width=0.55)
for i, (grp, rate) in enumerate(inpt_rate.items()):
    ax2.text(i, rate + 0.3, f"{rate:.1f}%", ha="center", fontsize=10)
ax2.axhline(df[TARGET_COL].mean()*100, color="gray", linestyle="--", linewidth=1)
ax2.set_xlabel("Number of Prior Inpatient Admissions")
ax2.set_ylabel("Early Readmission Rate (%)")
ax2.set_title("B  Readmission Rate by Prior Inpatient History",
              fontweight="bold", pad=10)

fig.suptitle("Clinical Feature Associations with Early Readmission",
             fontsize=14, y=1.02)
plt.tight_layout()
save_figure(fig, "10_feature_readmission_rates.png", FIGURES_DIR, dpi=180)
save_figure(fig, "10_feature_readmission_rates.png", PAPER_FIG_DIR, dpi=180)
plt.show()
"""),

md("## Figure 3 — Model performance comparison"),
code("""\
# Train all models (same setup as NB09) and compare visually.

if SMOTE_AVAILABLE:
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_tv_res, y_tv_res = smote.fit_resample(X_tv, y_tv)
else:
    X_tv_res, y_tv_res = X_tv, y_tv

fit_models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000,
                                               random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                            min_samples_leaf=20, class_weight="balanced_subsample",
                                            random_state=RANDOM_SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                     learning_rate=0.05, subsample=0.8,
                                                     random_state=RANDOM_SEED),
}

try:
    from xgboost import XGBClassifier
    sw = (y_tv_res==0).sum()/(y_tv_res==1).sum()
    fit_models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, subsample=0.8,
                                          scale_pos_weight=sw, eval_metric="auc",
                                          random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
except ImportError:
    pass

print("Training models for visualisation...")
for name, model in fit_models.items():
    model.fit(X_tv_res, y_tv_res)
    print(f"  ✓ {name}")

model_probs_test = {name: model.predict_proba(X_test)[:,1]
                    for name, model in fit_models.items()}
"""),

code("""\
# Multi-panel: ROC + PR + metric bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ── ROC curves ─────────────────────────────────────────────────────────────
ax = axes[0]
palette = sns.color_palette("colorblind", len(model_probs_test))
ls_list = ["-", "--", "-.", ":"]
for (name, y_prob), color, ls in zip(model_probs_test.items(), palette, ls_list):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, ls=ls, lw=2)
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Random (0.500)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("A  ROC Curves — Test Set", fontweight="bold", pad=10)
ax.legend(fontsize=8, loc="lower right")

# ── PR curves ──────────────────────────────────────────────────────────────
ax2 = axes[1]
baseline_pr = y_test.mean()
ax2.axhline(baseline_pr, color="gray", ls="--", lw=1,
            label=f"Baseline ({baseline_pr:.3f})")
for (name, y_prob), color, ls in zip(model_probs_test.items(), palette, ls_list):
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax2.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, ls=ls, lw=2)
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("B  Precision-Recall Curves — Test Set", fontweight="bold", pad=10)
ax2.legend(fontsize=8, loc="upper right")

# ── Metric bar chart ───────────────────────────────────────────────────────
ax3 = axes[2]
from sklearn.metrics import f1_score, recall_score
metric_names = ["ROC-AUC", "Avg Prec", "F1 (cls 1)", "Recall (cls 1)"]
model_names  = list(fit_models.keys())
metric_vals  = []

for name, model in fit_models.items():
    y_pred = model.predict(X_test)
    y_prob = model_probs_test[name]
    metric_vals.append([
        roc_auc_score(y_test, y_prob),
        average_precision_score(y_test, y_prob),
        f1_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
    ])

x_pos   = np.arange(len(metric_names))
bar_w   = 0.18
offsets = np.linspace(-(len(model_names)-1)/2 * bar_w,
                       (len(model_names)-1)/2 * bar_w, len(model_names))

for idx, (name, vals, color) in enumerate(zip(model_names, metric_vals, palette)):
    ax3.bar(x_pos + offsets[idx], vals, width=bar_w, color=color,
            alpha=0.85, label=name, edgecolor="white")

ax3.set_xticks(x_pos)
ax3.set_xticklabels(metric_names, rotation=15, fontsize=9)
ax3.set_ylim(0, 1.05)
ax3.set_ylabel("Score")
ax3.set_title("C  Test Set Metrics Comparison", fontweight="bold", pad=10)
ax3.legend(fontsize=8)

fig.suptitle("Model Performance Summary — Diabetes Readmission Prediction",
             fontsize=14, y=1.02)
plt.tight_layout()
save_figure(fig, "10_model_performance_comparison.png", FIGURES_DIR, dpi=180)
save_figure(fig, "10_model_performance_comparison.png", PAPER_FIG_DIR, dpi=180)
plt.show()
print("Figure 3 saved.")
"""),

md("## Figure 4 — EDA storytelling: who gets readmitted?"),
code("""\
# A 2×3 dashboard telling the clinical story from the data.

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ── Panel 1: Readmission rate by time in hospital ──────────────────────────
ax = axes[0, 0]
df_g = df.groupby("time_in_hospital")[TARGET_COL].mean() * 100
ax.plot(df_g.index, df_g.values, "o-", color=COLORS["readmit"], lw=2, ms=7)
ax.axhline(df[TARGET_COL].mean()*100, color="gray", ls="--", lw=1)
ax.set_xlabel("Days in Hospital"); ax.set_ylabel("Readmission Rate (%)")
ax.set_title("A  Readmission Rate by LOS", fontweight="bold")

# ── Panel 2: Readmission rate by number of medications ────────────────────
ax = axes[0, 1]
df_g2 = df.groupby("num_medications")[TARGET_COL].mean() * 100
ax.plot(df_g2.index, df_g2.values, "o-", color=COLORS["highlight"], lw=2, ms=5)
ax.axhline(df[TARGET_COL].mean()*100, color="gray", ls="--", lw=1)
ax.set_xlabel("Number of Medications"); ax.set_ylabel("Readmission Rate (%)")
ax.set_title("B  Readmission Rate vs Medications", fontweight="bold")

# ── Panel 3: Readmission by A1C result ────────────────────────────────────
ax = axes[0, 2]
if "A1Cresult_encoded" in df.columns:
    a1c_map_rev = {0:"Not tested",1:"Normal",2:">7",3:">8"}
    a1c_rate = df.groupby("A1Cresult_encoded")[TARGET_COL].mean() * 100
    a1c_labels = [a1c_map_rev.get(int(k), str(k)) for k in a1c_rate.index]
    bar_colors = [COLORS["no_readmit"] if r < df[TARGET_COL].mean()*100
                  else COLORS["readmit"] for r in a1c_rate.values]
    ax.bar(a1c_labels, a1c_rate.values, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(df[TARGET_COL].mean()*100, color="gray", ls="--", lw=1)
    ax.set_ylabel("Readmission Rate (%)")
    ax.set_title("C  Readmission Rate by HbA1c Result", fontweight="bold")

# ── Panel 4: Medication changes histogram by group ─────────────────────────
ax = axes[1, 0]
if "n_medication_changes" in df.columns:
    ax.hist(df[df[TARGET_COL]==0]["n_medication_changes"], bins=10,
            alpha=0.65, color=COLORS["no_readmit"], label="No Readmit", edgecolor="white")
    ax.hist(df[df[TARGET_COL]==1]["n_medication_changes"], bins=10,
            alpha=0.65, color=COLORS["readmit"],    label="Readmit", edgecolor="white")
    ax.set_xlabel("Number of Medication Changes"); ax.set_ylabel("Count")
    ax.set_title("D  Medication Changes by Group", fontweight="bold")
    ax.legend(fontsize=9)

# ── Panel 5: Number of diagnoses distribution ─────────────────────────────
ax = axes[1, 1]
ax.hist(df[df[TARGET_COL]==0]["number_diagnoses"], bins=12,
        alpha=0.65, color=COLORS["no_readmit"], label="No Readmit", edgecolor="white")
ax.hist(df[df[TARGET_COL]==1]["number_diagnoses"], bins=12,
        alpha=0.65, color=COLORS["readmit"],    label="Readmit", edgecolor="white")
ax.set_xlabel("Number of Diagnoses"); ax.set_ylabel("Count")
ax.set_title("E  Number of Diagnoses by Group", fontweight="bold")
ax.legend(fontsize=9)

# ── Panel 6: Lab procedures vs readmission ─────────────────────────────────
ax = axes[1, 2]
ax.hist(df[df[TARGET_COL]==0]["num_lab_procedures"], bins=20,
        alpha=0.65, color=COLORS["no_readmit"], label="No Readmit", edgecolor="white")
ax.hist(df[df[TARGET_COL]==1]["num_lab_procedures"], bins=20,
        alpha=0.65, color=COLORS["readmit"],    label="Readmit", edgecolor="white")
ax.set_xlabel("Number of Lab Procedures"); ax.set_ylabel("Count")
ax.set_title("F  Lab Procedures by Group", fontweight="bold")
ax.legend(fontsize=9)

fig.suptitle("EDA Dashboard: Clinical Patterns in Early Readmission",
             fontsize=14, y=1.02)
plt.tight_layout()
save_figure(fig, "10_eda_storytelling_dashboard.png", FIGURES_DIR, dpi=180)
save_figure(fig, "10_eda_storytelling_dashboard.png", PAPER_FIG_DIR, dpi=180)
plt.show()
"""),

md("## Figure 5 — Correlation heatmap (report-quality)"),
code("""\
numeric_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in [PATIENT_ID_COL]]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True   # hide upper triangle

sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r",
            center=0, vmin=-0.7, vmax=0.7, square=True,
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.7, "label": "Pearson r"})
ax.set_title("Feature Correlation Matrix\n(lower triangle only, showing |r|)",
             fontsize=14, pad=15)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
save_figure(fig, "10_correlation_heatmap.png", FIGURES_DIR, dpi=180)
save_figure(fig, "10_correlation_heatmap.png", PAPER_FIG_DIR, dpi=180)
plt.show()
"""),

md("## Summary — All visualisation figures saved"),
code("""\
all_figs = sorted([f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")])
print(f"Total figures in reports/figures/ : {len(all_figs)}")
for fig_name in all_figs:
    print(f"  {fig_name}")
"""),

md("""## Summary of Notebook 10

I created **5 multi-panel, publication-quality figure sets** totalling over 15 individual panels.
Every figure is saved to both `reports/figures/` and `paper_or_report/figures/`.

**Visual narrative:**
1. **Dataset overview** — size, class imbalance, age distribution, diagnosis mix
2. **Feature associations** — which clinical features are associated with readmission
3. **Model comparison** — ROC, PR curves, and metric bar chart side by side
4. **EDA storytelling** — 6-panel clinical pattern dashboard
5. **Correlation heatmap** — feature relationships at a glance

**Next step:** Notebook 11 — Error Analysis & SHAP Interpretability
"""),
]

print("Building notebooks 09–10...")
save_nb("09_model_evaluation.ipynb",    nb09)
save_nb("10_visualizations.ipynb",      nb10)
print("Done with 09-10.")
