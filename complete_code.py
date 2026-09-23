"""
complete_code.py
================
Dark Pattern Detection & Analysis System
Y.Anantha_darkpatterns

Single combined source file containing all project code:
  SECTION 1  — Imports & Shared Paths
  SECTION 2  — Data Loading     (load_data)
  SECTION 3  — Data Cleaning    (clean_data)
  SECTION 4  — Data Analysis    (analyze_data)
  SECTION 5  — Charts           (save_category_bar_chart, save_label_pie_chart,
                                  save_model_accuracy_chart, save_confusion_matrix_chart)
  SECTION 6  — Model Training   (evaluate_model, train_and_evaluate)
  SECTION 7  — KPIs             (calculate_kpis, print_kpis)
  SECTION 8  — Screenshots      (wait_for_streamlit, click_sidebar_page, take_screenshots)
  SECTION 9  — Main Entry Point (runs all steps in order when executed directly)

Run the full pipeline with:
    python complete_code.py

To launch the Streamlit app use:
    streamlit run Y.Anantha_darkpatterns_app.py --server.port 8502
"""

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & SHARED PATHS
# ═══════════════════════════════════════════════════════════════
import os
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts & Streamlit)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from playwright.sync_api import sync_playwright

# ── Paths ──────────────────────────────────────────────────────
DATA_PATH       = os.path.join("data", "darkpatterndataset.csv")
CLEANED_PATH    = os.path.join("data", "cleaned_darkpatterns.csv")
MODELS_DIR      = "models"
CHARTS_DIR      = "charts"
SCREENSHOTS_DIR = "screenshots"
RESULTS_PATH    = os.path.join(MODELS_DIR, "model_results.json")
KPI_PATH        = "kpis.json"
APP_URL         = "http://localhost:8502"

# ── Ensure output folders exist ───────────────────────────────
os.makedirs(CHARTS_DIR,      exist_ok=True)
os.makedirs(MODELS_DIR,      exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and print a basic overview."""
    df = pd.read_csv(path)

    print("=" * 55)
    print("STEP 1 – LOAD DATA")
    print("=" * 55)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print(f"Names   : {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Identify and fix data quality issues.
    Returns the cleaned DataFrame and a summary dict for reporting.
    """
    print("\n" + "=" * 55)
    print("STEP 2 – CLEAN DATA")
    print("=" * 55)

    summary = {}

    # --- 2a. Missing values -------------------------------------------------
    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing)
    summary["missing_before"] = missing.to_dict()

    # Drop rows where 'text' or 'label' is missing (essential columns)
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    # Fill missing 'Pattern Category' with 'Unknown'
    df["Pattern Category"] = df["Pattern Category"].fillna("Unknown")
    summary["rows_dropped_missing"] = before - len(df)
    print(f"\n  >> Dropped {before - len(df)} rows with missing 'text' or 'label'.")
    print(  "  >> Filled missing 'Pattern Category' with 'Unknown'.")

    # --- 2b. Duplicate rows -------------------------------------------------
    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows found: {dupes}")
    summary["duplicates_found"] = int(dupes)
    df = df.drop_duplicates()
    print(f"  >> Removed {dupes} duplicate rows.")

    # --- 2c. Whitespace / empty text ----------------------------------------
    df["text"] = df["text"].astype(str).str.strip()
    empty_text = (df["text"] == "").sum()
    df = df[df["text"] != ""]
    summary["empty_text_removed"] = int(empty_text)
    print(f"\nEmpty text strings removed: {empty_text}")

    # --- 2d. Label column – keep only 0 and 1 --------------------------------
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    invalid_labels = df["label"].isnull().sum()
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    summary["invalid_labels_removed"] = int(invalid_labels)
    print(f"Invalid label values removed : {invalid_labels}")

    # --- 2e. Pattern Category – strip whitespace ----------------------------
    df["Pattern Category"] = df["Pattern Category"].str.strip()

    # --- Final state --------------------------------------------------------
    print(f"\nCleaned dataset shape: {df.shape}")
    summary["final_rows"]    = len(df)
    summary["final_columns"] = df.shape[1]
    summary["missing_after"] = df.isnull().sum().to_dict()

    # Save
    df.to_csv(CLEANED_PATH, index=False)
    print(f"\nCleaned data saved >> {CLEANED_PATH}")
    return df, summary


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════
def analyze_data(df: pd.DataFrame) -> dict:
    """Compute counts, distributions, and basic statistics."""
    print("\n" + "=" * 55)
    print("STEP 3 – ANALYZE DATA")
    print("=" * 55)

    stats = {}

    # Label distribution
    label_counts = df["label"].value_counts().rename({1: "Dark Pattern", 0: "Not Dark Pattern"})
    stats["label_counts"] = label_counts
    print("\nLabel distribution:")
    print(label_counts)

    # Pattern category counts
    cat_counts = df["Pattern Category"].value_counts()
    stats["category_counts"] = cat_counts
    print("\nPattern Category distribution:")
    print(cat_counts)

    # Most common dark pattern types (excluding 'Not Dark Pattern')
    dark_only = df[df["label"] == 1]["Pattern Category"].value_counts()
    stats["dark_type_counts"] = dark_only
    print("\nMost common dark pattern types:")
    print(dark_only)

    # Basic statistics
    print("\nBasic statistics (numeric columns):")
    desc = df.describe(include="all")
    print(desc)
    stats["describe"] = desc

    return stats


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — CHARTS
# ═══════════════════════════════════════════════════════════════
def save_category_bar_chart(df: pd.DataFrame) -> str:
    """Bar chart: count of each dark pattern category."""
    cat_counts = df["Pattern Category"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Dark Pattern Category Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Pattern Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "category_bar.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved chart >> {path}")
    return path


def save_label_pie_chart(df: pd.DataFrame) -> str:
    """Pie chart: dark pattern vs not dark pattern."""
    label_counts = df["label"].value_counts()
    labels = ["Not Dark Pattern" if l == 0 else "Dark Pattern" for l in label_counts.index]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        label_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#4CAF50", "#F44336"],
    )
    ax.set_title("Dark Pattern vs Not Dark Pattern", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "label_pie.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved chart >> {path}")
    return path


def save_model_accuracy_chart(results: dict) -> str:
    """Bar chart: model accuracy comparison."""
    models     = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in models]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, accuracies, color=["#2196F3", "#FF9800"], edgecolor="white", width=0.4)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy")
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "model_accuracy.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved chart >> {path}")
    return path


def save_confusion_matrix_chart(cm_data, model_name: str) -> str:
    """Heatmap of the confusion matrix for the best model."""
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Dark Pattern", "Dark Pattern"],
        yticklabels=["Not Dark Pattern", "Dark Pattern"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved chart >> {path}")
    return path


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Return a dict with all evaluation metrics for one model."""
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"Model : {name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Dark Pattern", "Dark Pattern"]))
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy" : float(acc),
        "precision": float(prec),
        "recall"   : float(rec),
        "f1"       : float(f1),
        "cm"       : cm.tolist(),
    }


def train_and_evaluate() -> tuple[dict, str, TfidfVectorizer]:
    """Full ML pipeline: vectorise, split, train, evaluate, save best model."""
    print("\n" + "=" * 55)
    print("STEP 4 – MODEL TRAINING & EVALUATION")
    print("=" * 55)

    df = pd.read_csv(CLEANED_PATH)
    print(f"Loaded cleaned data: {df.shape}")

    X = df["text"].astype(str)
    y = df["label"].astype(int)

    # TF-IDF vectorisation
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        stop_words="english",
    )

    # 80 / 20 stratified split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)
    print(f"\nTrain size : {X_train.shape[0]}")
    print(f"Test size  : {X_test.shape[0]}")

    # Train both models
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Evaluate both models
    results = {}
    results["Logistic Regression"] = evaluate_model(
        "Logistic Regression", lr_model, X_test, y_test
    )
    results["Naive Bayes"] = evaluate_model(
        "Naive Bayes", nb_model, X_test, y_test
    )

    # Pick best model by F1
    best_name  = max(results, key=lambda m: results[m]["f1"])
    best_model = lr_model if best_name == "Logistic Regression" else nb_model
    print(f"\n>>  Best model (by F1): {best_name}")

    # Save best model + vectorizer
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print(f"  Saved >> {MODELS_DIR}/best_model.pkl")
    print(f"  Saved >> {MODELS_DIR}/tfidf_vectorizer.pkl")

    # Save results JSON
    results_to_save = dict(results)
    results_to_save["best_model_name"] = best_name
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"  Results >> {RESULTS_PATH}")

    # Generate model charts
    save_model_accuracy_chart(results)
    save_confusion_matrix_chart(results[best_name]["cm"], best_name)

    return results, best_name, vectorizer


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — KPI CALCULATION
# ═══════════════════════════════════════════════════════════════
def calculate_kpis() -> dict:
    """Calculate all project KPIs from real data and return as a dict."""

    df_raw   = pd.read_csv(DATA_PATH)
    df_clean = pd.read_csv(CLEANED_PATH)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    total_raw   = len(df_raw)
    total_clean = len(df_clean)

    # Data Quality KPIs
    missing_count   = int(df_raw.isnull().sum().sum())
    duplicate_count = int(df_raw.duplicated().sum())
    total_cells     = total_raw * df_raw.shape[1]
    missing_rate    = round((missing_count / total_cells) * 100, 2)
    duplicate_rate  = round((duplicate_count / total_raw) * 100, 2)
    completeness    = round(100 - missing_rate, 2)

    data_quality = {
        "total_records_raw"      : total_raw,
        "total_records_cleaned"  : total_clean,
        "records_removed"        : total_raw - total_clean,
        "missing_value_rate_pct" : missing_rate,
        "duplicate_rate_pct"     : duplicate_rate,
        "data_completeness_pct"  : completeness,
    }

    # Model KPIs
    best_name = results["best_model_name"]
    lr = results["Logistic Regression"]
    nb = results["Naive Bayes"]

    def cm_breakdown(cm):
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

    model_kpis = {
        "Logistic Regression": {
            "accuracy"         : round(lr["accuracy"]  * 100, 2),
            "precision"        : round(lr["precision"] * 100, 2),
            "recall"           : round(lr["recall"]    * 100, 2),
            "f1_score"         : round(lr["f1"]        * 100, 2),
            "confusion_matrix" : cm_breakdown(lr["cm"]),
        },
        "Naive Bayes": {
            "accuracy"         : round(nb["accuracy"]  * 100, 2),
            "precision"        : round(nb["precision"] * 100, 2),
            "recall"           : round(nb["recall"]    * 100, 2),
            "f1_score"         : round(nb["f1"]        * 100, 2),
            "confusion_matrix" : cm_breakdown(nb["cm"]),
        },
        "best_model"      : best_name,
        "best_f1_score"   : round(results[best_name]["f1"] * 100, 2),
        "best_accuracy"   : round(results[best_name]["accuracy"] * 100, 2),
        "train_test_split": "80% train / 20% test",
        "test_samples"    : 472,
        "train_samples"   : 1884,
    }

    # Business KPIs
    dark_count  = int((df_clean["label"] == 1).sum())
    safe_count  = int((df_clean["label"] == 0).sum())
    dark_df     = df_clean[df_clean["label"] == 1]
    cat_counts  = dark_df["Pattern Category"].value_counts()
    top_cat     = str(cat_counts.index[0])
    top_cnt     = int(cat_counts.iloc[0])
    cat_share   = {cat: round((cnt / dark_count) * 100, 1) for cat, cnt in cat_counts.items()}
    det_cov     = round(results[best_name]["recall"] * 100, 2)

    business_kpis = {
        "total_samples"              : total_clean,
        "dark_pattern_count"         : dark_count,
        "not_dark_pattern_count"     : safe_count,
        "dark_pattern_prevalence_pct": round((dark_count / total_clean) * 100, 1),
        "top_dark_pattern_type"      : top_cat,
        "top_dark_pattern_count"     : top_cnt,
        "top_dark_pattern_share_pct" : round((top_cnt / dark_count) * 100, 1),
        "category_share_pct"         : cat_share,
        "detection_coverage_pct"     : det_cov,
        "categories_detected"        : int(cat_counts.shape[0]),
    }

    return {"data_quality": data_quality, "model": model_kpis, "business": business_kpis}


def print_kpis(kpis: dict):
    """Pretty-print all KPIs to the console."""
    dq = kpis["data_quality"]
    m  = kpis["model"]
    b  = kpis["business"]

    print("\n" + "=" * 55)
    print("STEP 5 – KPI SUMMARY")
    print("=" * 55)
    print(f"\n  [DATA QUALITY]")
    print(f"  Total records (raw)     : {dq['total_records_raw']}")
    print(f"  Total records (cleaned) : {dq['total_records_cleaned']}")
    print(f"  Missing value rate      : {dq['missing_value_rate_pct']}%")
    print(f"  Duplicate rate          : {dq['duplicate_rate_pct']}%")
    print(f"  Data completeness       : {dq['data_completeness_pct']}%")

    print(f"\n  [MODEL PERFORMANCE]")
    for mn in ["Logistic Regression", "Naive Bayes"]:
        mk = m[mn]
        tag = " <- BEST" if mn == m["best_model"] else ""
        print(f"  {mn}{tag}")
        print(f"    Accuracy={mk['accuracy']}%  Precision={mk['precision']}%  "
              f"Recall={mk['recall']}%  F1={mk['f1_score']}%")

    print(f"\n  [BUSINESS]")
    print(f"  Dark pattern prevalence : {b['dark_pattern_prevalence_pct']}%")
    print(f"  Top pattern type        : {b['top_dark_pattern_type']} ({b['top_dark_pattern_share_pct']}%)")
    print(f"  Detection coverage      : {b['detection_coverage_pct']}%")


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — SCREENSHOT AUTOMATION (Playwright)
# ═══════════════════════════════════════════════════════════════
# Pages to screenshot: (sidebar label text, output filename)
_SCREENSHOT_PAGES = [
    ("Home",                "01_home.png"),
    ("Data Overview",       "02_data_overview.png"),
    ("Charts",              "03_charts.png"),
    ("Model Results",       "04_model_results.png"),
    ("Detect Dark Pattern", "05_detect_empty.png"),
    ("Business Insights",   "06_business_insights.png"),
]
_DETECT_TEXT = "Hurry! Only 2 items left in stock"


def wait_for_streamlit(page):
    """Wait until Streamlit finishes loading (spinner disappears)."""
    try:
        page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=15000)
    except Exception:
        pass
    time.sleep(1.5)


def click_sidebar_page(page, label: str):
    """Click a sidebar radio button by its visible label text."""
    try:
        page.locator(f"label:has-text('{label}')").first.click()
    except Exception:
        page.locator(f"[data-testid='stSidebar'] >> text={label}").first.click()
    wait_for_streamlit(page)


def take_screenshots():
    """Open the Streamlit app and screenshot every sidebar page."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page    = context.new_page()

        print(f"\nOpening {APP_URL} ...")
        page.goto(APP_URL, wait_until="networkidle", timeout=30000)
        wait_for_streamlit(page)
        print("App loaded.")

        # Screenshot each sidebar page
        for label, filename in _SCREENSHOT_PAGES:
            print(f"  Navigating to: {label}")
            click_sidebar_page(page, label)
            out_path = os.path.join(SCREENSHOTS_DIR, filename)
            page.screenshot(path=out_path, full_page=True)
            print(f"    Saved: {out_path}")

        # Detect page with input text
        print("  Navigating to: Detect Dark Pattern (with input)")
        click_sidebar_page(page, "Detect Dark Pattern")
        page.locator("textarea").first.fill(_DETECT_TEXT)
        time.sleep(0.5)
        page.locator("button:has-text('Detect')").first.click()
        wait_for_streamlit(page)
        detect_path = os.path.join(SCREENSHOTS_DIR, "07_detect_result.png")
        page.screenshot(path=detect_path, full_page=True)
        print(f"    Saved: {detect_path}")

        browser.close()

    print("\nAll screenshots saved:")
    for fname in sorted(os.listdir(SCREENSHOTS_DIR)):
        if fname.endswith(".png"):
            sz = os.path.getsize(os.path.join(SCREENSHOTS_DIR, fname))
            print(f"  {fname}  ({sz:,} bytes)")


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — MAIN ENTRY POINT
# Runs all pipeline steps in order when executed directly.
# The Streamlit app (Y.Anantha_darkpatterns_app.py) must be
# started separately:
#     streamlit run Y.Anantha_darkpatterns_app.py --server.port 8502
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "#" * 55)
    print("# Dark Pattern Detection & Analysis System")
    print("# complete_code.py — Full Pipeline")
    print("#" * 55)

    # Step 1–3: Load, clean, analyse
    df_raw            = load_data()
    df_clean, summary = clean_data(df_raw)
    stats             = analyze_data(df_clean)

    # Step 4 (charts from analysis)
    print("\n" + "=" * 55)
    print("STEP 4 (part A) – DATA CHARTS")
    print("=" * 55)
    save_category_bar_chart(df_clean)
    save_label_pie_chart(df_clean)

    # Step 4 (model training)
    results, best_name, vectorizer = train_and_evaluate()

    # Step 5: KPIs
    kpis = calculate_kpis()
    print_kpis(kpis)
    with open(KPI_PATH, "w") as f:
        json.dump(kpis, f, indent=2)
    print(f"\nKPIs saved >> {KPI_PATH}  ({os.path.getsize(KPI_PATH):,} bytes)")

    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Cleaned data   >> {CLEANED_PATH}")
    print(f"  Best model     >> {MODELS_DIR}/best_model.pkl  ({best_name})")
    print(f"  KPIs           >> {KPI_PATH}")
    print(f"  Charts         >> {CHARTS_DIR}/")
    print(f"\nNext step — launch the Streamlit app:")
    print(f"  streamlit run Y.Anantha_darkpatterns_app.py --server.port 8502")
    print(f"\n  Local   : http://localhost:8502")
    print(f"  Network : http://192.168.1.5:8502")
    print("\ncomplete_code.py complete.")
