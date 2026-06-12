"""
config.py — Central configuration for the Diabetes Readmission project.

Every notebook imports from here so that paths, constants, and settings
are defined in exactly one place. Changing a path here updates every notebook.
"""

import os

# ── Project root ─────────────────────────────────────────────────────────────
# __file__ is this file's path.  Two dirname calls take us to the project root.
CONFIG_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)

# ── Data paths ───────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
INTERIM_DIR   = os.path.join(PROJECT_ROOT, "data", "interim")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

RAW_FILE       = os.path.join(RAW_DIR,       "diabetic_data.csv")
IDS_MAP_FILE   = os.path.join(RAW_DIR,       "IDs_mapping.csv")
INTERIM_FILE   = os.path.join(INTERIM_DIR,   "diabetic_interim.csv")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "diabetic_processed.csv")
TRAIN_FILE     = os.path.join(PROCESSED_DIR, "train.csv")
VAL_FILE       = os.path.join(PROCESSED_DIR, "val.csv")
TEST_FILE      = os.path.join(PROCESSED_DIR, "test.csv")

# ── Report paths ─────────────────────────────────────────────────────────────
FIGURES_DIR    = os.path.join(PROJECT_ROOT, "reports", "figures")
TABLES_DIR     = os.path.join(PROJECT_ROOT, "reports", "tables")
PAPER_FIG_DIR  = os.path.join(PROJECT_ROOT, "paper_or_report", "figures")
PAPER_TAB_DIR  = os.path.join(PROJECT_ROOT, "paper_or_report", "tables")

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Modelling constants ──────────────────────────────────────────────────────
TEST_SIZE       = 0.15   # 15 % held-out test set
VAL_SIZE        = 0.15   # 15 % validation set  (from remaining after test split)
TARGET_COL      = "readmitted_binary"
PATIENT_ID_COL  = "patient_nbr"

# ── Download URL ──────────────────────────────────────────────────────────────
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"

# ── Medication columns (24 drug columns in the dataset) ──────────────────────
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# ── ICD-9 disease category mapping ───────────────────────────────────────────
# Groups the first ICD-9 diagnosis code into broad clinical categories.
# Based on the standard groupings used in the UCI dataset paper.
def icd9_to_category(code):
    """
    Maps a raw ICD-9 diagnosis code string to a broad disease category.

    Args:
        code (str): raw diag_1, diag_2, or diag_3 value from the dataset.

    Returns:
        str: disease category label.
    """
    if code in ("?", "E", "V") or (isinstance(code, str) and code.startswith(("E", "V"))):
        return "Other / External"
    try:
        c = float(code)
    except (ValueError, TypeError):
        return "Other / External"

    if 390 <= c <= 459 or c == 785:
        return "Circulatory"
    elif 460 <= c <= 519 or c == 786:
        return "Respiratory"
    elif 520 <= c <= 579 or c == 787:
        return "Digestive"
    elif c == 250:
        return "Diabetes"
    elif 800 <= c <= 999:
        return "Injury"
    elif 710 <= c <= 739:
        return "Musculoskeletal"
    elif 580 <= c <= 629 or c == 788:
        return "Genitourinary"
    elif 140 <= c <= 239:
        return "Neoplasm"
    else:
        return "Other"

# ── Age group ordinal mapping ─────────────────────────────────────────────────
AGE_ORDER = {
    "[0-10)":  0,  "[10-20)": 1,  "[20-30)": 2,  "[30-40)": 3,
    "[40-50)": 4,  "[50-60)": 5,  "[60-70)": 6,  "[70-80)": 7,
    "[80-90)": 8,  "[90-100)": 9
}

# ── Plot defaults ─────────────────────────────────────────────────────────────
FIGSIZE_WIDE   = (14, 5)
FIGSIZE_SQUARE = (8, 7)
FIGSIZE_TALL   = (8, 10)
COLORS = {
    "no_readmit": "#4878CF",
    "readmit":    "#D65F5F",
    "neutral":    "#6ACC65",
    "highlight":  "#E8851A",
}
PALETTE = [COLORS["no_readmit"], COLORS["readmit"]]

def ensure_dirs():
    """Create all output directories if they don't exist yet."""
    for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
              FIGURES_DIR, TABLES_DIR, PAPER_FIG_DIR, PAPER_TAB_DIR]:
        os.makedirs(d, exist_ok=True)
