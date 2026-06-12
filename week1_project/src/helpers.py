"""
helpers.py — Reusable utility functions for the Heart Disease Analysis project.

This file contains small, clearly-named helper functions that are used
across multiple notebooks. Each function is explained with a docstring.

For a beginner: think of this file as a toolbox. Instead of writing the
same code in every notebook, I put repeated patterns here and import them.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ─── PATH HELPERS ──────────────────────────────────────────────────────────────

def get_project_root():
    """
    Returns the absolute path to the project root folder.
    
    This works by finding the directory that contains 'src/helpers.py'.
    It lets any notebook find project folders reliably, no matter where
    the notebook is opened from.
    
    Returns:
        str: absolute path to the project root.
    """
    # __file__ is the path to this helpers.py file
    # os.path.dirname goes up one level (to src/)
    # another os.path.dirname goes up one more level (to the project root)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    return project_root


def get_data_path(subfolder="raw", filename=""):
    """
    Returns the full path to a file inside the data/ folder.
    
    Args:
        subfolder (str): One of "raw" or "processed". Default is "raw".
        filename (str): The filename to append. If empty, returns the folder path.
    
    Returns:
        str: full path to the file or folder.
    
    Example:
        get_data_path("raw", "heart.csv")  →  ".../data/raw/heart.csv"
    """
    root = get_project_root()
    folder = os.path.join(root, "data", subfolder)
    if filename:
        return os.path.join(folder, filename)
    return folder


def get_figures_path(filename=""):
    """
    Returns the path to the reports/figures/ folder (or a specific figure file).
    
    Args:
        filename (str): Optional filename to append.
    
    Returns:
        str: path to the figures folder or a specific figure file.
    """
    root = get_project_root()
    folder = os.path.join(root, "reports", "figures")
    os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist
    if filename:
        return os.path.join(folder, filename)
    return folder


def get_tables_path(filename=""):
    """
    Returns the path to the reports/tables/ folder (or a specific table file).
    
    Args:
        filename (str): Optional filename to append.
    
    Returns:
        str: path to the tables folder or a specific table file.
    """
    root = get_project_root()
    folder = os.path.join(root, "reports", "tables")
    os.makedirs(folder, exist_ok=True)
    if filename:
        return os.path.join(folder, filename)
    return folder


# ─── SAVING HELPERS ────────────────────────────────────────────────────────────

def save_figure(fig, filename, dpi=150):
    """
    Saves a matplotlib figure to the reports/figures/ folder.
    
    I use this instead of plt.savefig() directly so that every figure
    always goes to the same, correct folder automatically.
    
    Args:
        fig: a matplotlib Figure object (e.g. from plt.figure() or fig, ax = plt.subplots())
        filename (str): filename including extension, e.g. "correlation_heatmap.png"
        dpi (int): dots per inch. 150 is good for screen; use 300 for print.
    
    Returns:
        str: the full path where the figure was saved.
    
    Example:
        fig, ax = plt.subplots()
        ax.hist(df["age"])
        save_figure(fig, "age_histogram.png")
    """
    path = get_figures_path(filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved → {path}")
    return path


def save_table(df, filename):
    """
    Saves a pandas DataFrame as a CSV to the reports/tables/ folder.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): filename including .csv extension.
    
    Returns:
        str: full path where the table was saved.
    
    Example:
        summary = df.describe()
        save_table(summary, "descriptive_statistics.csv")
    """
    path = get_tables_path(filename)
    df.to_csv(path)
    print(f"Table saved  → {path}")
    return path


# ─── DISPLAY HELPERS ───────────────────────────────────────────────────────────

def print_section(title):
    """
    Prints a clean section header in the notebook output.
    
    Useful for separating different parts of a long notebook cell output.
    
    Args:
        title (str): section heading text.
    
    Example:
        print_section("Missing Value Check")
        → prints: ══════════════════════════
                   Missing Value Check
                   ══════════════════════════
    """
    line = "═" * (len(title) + 4)
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def summarise_dataframe(df, name="DataFrame"):
    """
    Prints a quick structured summary of any DataFrame.
    
    This is useful at the start of any notebook to quickly understand
    the data — shape, dtypes, missing values, and a sample.
    
    Args:
        df (pd.DataFrame): the DataFrame to summarise.
        name (str): a label for the print output.
    """
    print_section(f"Summary: {name}")
    print(f"Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage  : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"\nColumn dtypes:")
    print(df.dtypes.to_string())
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
    print(missing_df[missing_df["missing_count"] > 0].to_string() if missing.sum() > 0 else "  None — all columns complete.")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())


# ─── COLUMN NAME MAP ───────────────────────────────────────────────────────────

# This dictionary maps the original UCI column codes to readable names.
# I define it here so every notebook uses the same column names consistently.
COLUMN_NAMES = [
    "age",          # age in years
    "sex",          # 1 = male, 0 = female
    "cp",           # chest pain type (1–4)
    "trestbps",     # resting blood pressure (mm Hg)
    "chol",         # serum cholesterol (mg/dl)
    "fbs",          # fasting blood sugar > 120 mg/dl (1=true, 0=false)
    "restecg",      # resting ECG results (0–2)
    "thalach",      # max heart rate achieved
    "exang",        # exercise induced angina (1=yes, 0=no)
    "oldpeak",      # ST depression (exercise vs rest)
    "slope",        # slope of peak exercise ST segment (1–3)
    "ca",           # number of major vessels coloured by fluoroscopy (0–3)
    "thal",         # thalassemia (3=normal, 6=fixed defect, 7=reversable)
    "target"        # diagnosis (0=no disease, 1-4=disease present → binarised to 0/1)
]

# Human-readable labels for the chest pain types
CP_LABELS = {
    1: "Typical angina",
    2: "Atypical angina",
    3: "Non-anginal pain",
    4: "Asymptomatic"
}

# Human-readable labels for the thal (thalassemia) column
THAL_LABELS = {
    3: "Normal",
    6: "Fixed defect",
    7: "Reversable defect"
}
