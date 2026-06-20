"""
=============================================================================
STAGE 1 — INDUSTRIAL DATA CLEANING
=============================================================================
Purpose : Load, clean, and prepare the raw melting dataset for ML training.
Output  : outputs/melting_cleaned_stage1.csv  (ML-ready)
          outputs/traceability_columns.csv     (IDs / codes for dashboard)
          outputs/cleaning_report.txt
=============================================================================
"""

import os
import re
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
RAW_FILE         = "melting_cleaned_final.xlsx"
OUTPUT_DIR       = "outputs"
OUTPUT_CLEAN     = os.path.join(OUTPUT_DIR, "melting_cleaned_stage1.csv")
OUTPUT_TRACE     = os.path.join(OUTPUT_DIR, "traceability_columns.csv")
OUTPUT_REPORT    = os.path.join(OUTPUT_DIR, "cleaning_report.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# COLUMN CLASSIFICATION
# ---------------------------------------------------------------------------
# Columns kept ONLY for traceability (never fed into ML models).
# These are IDs, codes, remarks, and QA decisions that would cause leakage.
TRACEABILITY_KEYWORDS = [
    "casting_sr","casting sr", "pattern no", "item", "furnace category", "furnace",
    "grade", "pigging", "aa", "zone", "last heel", "alloys",
    "boring", "punching", "furnace on", "furnace off",
    "tap_time", "pouring time",
]

# Additive columns that are CODED per defect class and cause perfect leakage.
# These are addition AMOUNTS (kg) that happen to be process decisions correlated
# directly with whether the batch was flagged — they CANNOT be used for prediction.
LEAKAGE_COLUMNS_EXACT = [
    "sulphur",    # addition amount coded per defect status — NOT chemistry S%
    "phos",       # addition amount coded per defect status — NOT chemistry P%
    "crom",       # addition amount coded per defect status — NOT chemistry Cr%
    "copper",     # copper additive kg — 450 kg for healthy vs 10 kg for defective (leakage)
    "heel_metal", # heel weight kg — 268 kg for healthy vs ~0.4 for defective (leakage)
    "nickel",     # nickel additive kg — 11 for healthy vs 15 for defective (leakage)
    "mg",         # Mg % column is bimodal: 373 (additive weight kg) vs 0.04 (chemistry %)
                  # The bimodal mixture makes it perfectly predictive — exclude
                  # Real Mg chemistry is captured via Mg % when in valid range (covered elsewhere)
]

# Columns that are time-stamp-like strings (removed from ML)
DATETIME_KEYWORDS = ["time", "date", "tap_time"]

# ---------------------------------------------------------------------------
# HELPER — standardise column names
# ---------------------------------------------------------------------------
def clean_col_name(col: str) -> str:
    """Lowercase, strip whitespace, remove % and special chars, replace spaces."""
    col = col.strip().lower()
    col = re.sub(r"[%\n\r]", "", col)          # remove % and newlines
    col = re.sub(r"[^a-z0-9_\s]", "", col)     # keep alphanumeric + space
    col = re.sub(r"\s+", "_", col).strip("_")  # spaces → underscore
    return col


# ---------------------------------------------------------------------------
# HELPER — convert a column to numeric safely
# ---------------------------------------------------------------------------
def safe_numeric(series: pd.Series) -> pd.Series:
    """Replace common non-numeric sentinels and coerce to float."""
    sentinels = ["-", "na", "null", "none", "n/a", "nan", "", " "]
    s = series.astype(str).str.strip().str.lower()
    s = s.replace(sentinels, np.nan)
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------------------
# HELPER — IQR-based outlier clipping
# ---------------------------------------------------------------------------
def clip_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 3.0) -> pd.DataFrame:
    """
    Clip values beyond median ± factor*IQR.
    Factor=3 is gentler than the standard 1.5 — suitable for industrial
    process data where legitimate extremes are common.
    """
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_clipped:
            log.info(f"  Clipped {n_clipped} outliers in '{col}' → [{lo:.4f}, {hi:.4f}]")
        df[col] = df[col].clip(lo, hi)
    return df


# ---------------------------------------------------------------------------
# HELPER — detect impossible industrial values
# ---------------------------------------------------------------------------
def flag_impossible_values(df: pd.DataFrame, report_lines: list) -> pd.DataFrame:
    """
    Identify and correct impossible physical values for cast iron process data.
    Corrections are logged but the row is NOT removed (data is precious).
    """
    checks = {
        # column : (min_valid, max_valid, description)
        "tapping_temp":  (1200, 1600, "Tapping temperature °C"),
        "pouring_temp":  (1200, 1600, "Pouring temperature °C"),
        "s_":            (0,    0.15,  "Sulfur %  (S%)"),
        "c_":            (2.5,  4.5,   "Carbon %  (C%)"),
        "si_":           (0.5,  5.0,   "Silicon % (Si%)"),
        "ce":            (2.5,  5.0,   "Carbon Equivalent"),
        "mg_":           (0,    0.10,  "Magnesium % (Mg%)"),
        "mg_recovery_":  (0,    2.0,   "Mg Recovery %"),
    }

    report_lines.append("\n=== IMPOSSIBLE VALUE CHECKS ===")
    for col, (lo, hi, desc) in checks.items():
        if col not in df.columns:
            continue
        bad = (df[col] < lo) | (df[col] > hi)
        n_bad = bad.sum()
        if n_bad:
            report_lines.append(f"  {desc}: {n_bad} rows outside [{lo}, {hi}] → set to NaN")
            df.loc[bad, col] = np.nan
        else:
            report_lines.append(f"  {desc}: OK (0 impossible values)")

    return df


# ---------------------------------------------------------------------------
# MAIN CLEANING PIPELINE
# ---------------------------------------------------------------------------
def run_stage1(input_file: str = RAW_FILE):
    report_lines = ["=" * 70, "INDUSTRIAL DATA CLEANING REPORT — STAGE 1", "=" * 70]

    # ------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------
    log.info(f"Loading: {input_file}")
    xl = pd.ExcelFile(input_file)
    report_lines.append(f"\nSheets found: {xl.sheet_names}")

    # Use first sheet (or the largest if multiple)
    df = pd.read_excel(input_file, sheet_name=0, dtype=str)
    report_lines.append(f"Raw shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. STANDARDISE COLUMN NAMES
    # ------------------------------------------------------------------
    original_cols = df.columns.tolist()
    df.columns = [clean_col_name(c) for c in df.columns]
    # Resolve duplicate names after cleaning
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    log.info(f"Cleaned column names. Columns: {list(df.columns)}")
    report_lines.append(f"\nColumn name mapping:")
    for orig, new in zip(original_cols, df.columns):
        report_lines.append(f"  '{orig}'  →  '{new}'")

    # ------------------------------------------------------------------
    # 3. REMOVE EMPTY ROWS & COLUMNS
    # ------------------------------------------------------------------
    n_before = len(df)
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    report_lines.append(f"\nEmpty rows removed: {n_before - len(df)}")

    # ------------------------------------------------------------------
    # 4. REMOVE DUPLICATES
    # ------------------------------------------------------------------
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_dupes = n_before - len(df)
    report_lines.append(f"Duplicate rows removed: {n_dupes}")

    # ------------------------------------------------------------------
    # 5. IDENTIFY TARGET COLUMN
    # ------------------------------------------------------------------
    # The target column is DEFECTED (or variations)
    target_col_raw = None
    for c in df.columns:
        if "defect" in c:
            target_col_raw = c
            break
    if target_col_raw is None:
        raise ValueError("Target column containing 'defect' not found!")

    log.info(f"Target column identified: '{target_col_raw}'")
    df.rename(columns={target_col_raw: "defect"}, inplace=True)
    df["defect"] = pd.to_numeric(df["defect"], errors="coerce").astype("Int64")

    # Drop rows where target is missing (can't train without label)
    n_before = len(df)
    df.dropna(subset=["defect"], inplace=True)
    report_lines.append(f"Rows dropped (missing target): {n_before - len(df)}")
    report_lines.append(f"Defect distribution:\n{df['defect'].value_counts().to_string()}")

    # ------------------------------------------------------------------
    # 6. SEPARATE TRACEABILITY COLUMNS
    # ------------------------------------------------------------------
    trace_cols = []
    for col in df.columns:
        if col == "defect":
            continue
        for kw in TRACEABILITY_KEYWORDS:
            if kw in col:
                trace_cols.append(col)
                break

    # Also catch any column that is clearly a string ID (>80% non-numeric)
    for col in df.columns:
        if col in trace_cols or col == "defect":
            continue
        numeric_converted = pd.to_numeric(df[col], errors="coerce")
        frac_numeric = numeric_converted.notna().mean()
        if frac_numeric < 0.2:
            trace_cols.append(col)

    trace_cols = list(dict.fromkeys(trace_cols))  # deduplicate, preserve order
    log.info(f"Traceability columns ({len(trace_cols)}): {trace_cols}")

    # Remove confirmed leakage columns (not ML-usable, not traceability either)
    leakage_found = [c for c in df.columns if c in LEAKAGE_COLUMNS_EXACT]
    if leakage_found:
        df.drop(columns=leakage_found, inplace=True)
        report_lines.append(f"\nLeakage columns REMOVED (additive codes correlated with target):\n  {leakage_found}")
        log.info(f"Removed leakage columns: {leakage_found}")
    report_lines.append(f"\nTraceability columns (excluded from ML):\n  {trace_cols}")

    # Save traceability separately
    df_trace = df[trace_cols + ["defect"]].copy()
    df_trace.to_csv(OUTPUT_TRACE, index=False)

    # Drop traceability from ML dataset
    ml_cols = [c for c in df.columns if c not in trace_cols]
    df = df[ml_cols].copy()

    # ------------------------------------------------------------------
    # 7. CONVERT NUMERIC COLUMNS
    # ------------------------------------------------------------------
    numeric_cols = []
    for col in df.columns:
        if col == "defect":
            continue
        converted = safe_numeric(df[col])
        frac_valid = converted.notna().mean()
        if frac_valid >= 0.3:   # at least 30% numeric → treat as numeric
            df[col] = converted
            numeric_cols.append(col)

    report_lines.append(f"\nNumeric columns identified: {len(numeric_cols)}")

    # ------------------------------------------------------------------
    # 8. MISSING VALUE REPORT (before imputation)
    # ------------------------------------------------------------------
    miss = df[numeric_cols].isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    report_lines.append(f"\n=== MISSING VALUES (before imputation) ===")
    for col, cnt in miss.items():
        pct = 100 * cnt / len(df)
        report_lines.append(f"  {col:<35} {cnt:>6} ({pct:.1f}%)")

    # Drop columns with > 60% missing (not recoverable)
    drop_high_missing = miss[miss / len(df) > 0.60].index.tolist()
    if drop_high_missing:
        df.drop(columns=drop_high_missing, inplace=True)
        numeric_cols = [c for c in numeric_cols if c not in drop_high_missing]
        report_lines.append(f"\nDropped (>60% missing): {drop_high_missing}")

    # MEDIAN imputation for remaining numeric columns
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    df[numeric_cols] = imp.fit_transform(df[numeric_cols])
    report_lines.append("\nMedian imputation applied to all numeric columns.")

    # ------------------------------------------------------------------
    # 9. DETECT & CORRECT IMPOSSIBLE VALUES
    # ------------------------------------------------------------------
    df = flag_impossible_values(df, report_lines)

    # Re-impute after impossible values set to NaN
    df[numeric_cols] = imp.fit_transform(df[numeric_cols])

    # ------------------------------------------------------------------
    # 10. CLIP EXTREME OUTLIERS (IQR × 3)
    # ------------------------------------------------------------------
    log.info("Clipping outliers …")
    report_lines.append("\n=== OUTLIER CLIPPING (IQR × 3) ===")
    df = clip_outliers_iqr(df, numeric_cols, factor=3.0)

    # ------------------------------------------------------------------
    # 11. HANDLE REMAINING CATEGORICAL COLUMNS
    # ------------------------------------------------------------------
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != "defect"]
    if cat_cols:
        # Fill missing with mode or 'UNKNOWN'
        for col in cat_cols:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) else "UNKNOWN"
            df[col].fillna(fill_val, inplace=True)
        report_lines.append(f"\nCategorical columns kept: {cat_cols}")
    else:
        report_lines.append("\nNo categorical columns remaining in ML set.")

    # ------------------------------------------------------------------
    # 12. FINAL SHAPE & SAVE
    # ------------------------------------------------------------------
    report_lines.append(f"\n=== FINAL CLEAN DATASET ===")
    report_lines.append(f"Shape           : {df.shape}")
    report_lines.append(f"Numeric columns : {len(numeric_cols)}")
    report_lines.append(f"Target balance  :\n{df['defect'].value_counts().to_string()}")

    df.to_csv(OUTPUT_CLEAN, index=False)
    log.info(f"Saved: {OUTPUT_CLEAN}")

    # ------------------------------------------------------------------
    # 13. SAVE CLEANING REPORT (UTF-8 SAFE)
    # ------------------------------------------------------------------

    # Windows default encoding (cp1252) cannot handle symbols like →.
    # So we explicitly save using UTF-8.

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log.info(f"Report saved: {OUTPUT_REPORT}")

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nFinal Dataset Shape : {df.shape}")
    print(f"Numeric Features    : {len(numeric_cols)}")
    print(f"Target Distribution :")
    print(df["defect"].value_counts())

    print(f"\nSaved Files:")
    print(f"  Cleaned Dataset  → {OUTPUT_CLEAN}")
    print(f"  Traceability CSV → {OUTPUT_TRACE}")
    print(f"  Cleaning Report  → {OUTPUT_REPORT}")

    return df, numeric_cols


    # ---------------------------------------------------------------------------
    # MAIN EXECUTION
    # ---------------------------------------------------------------------------

    if __name__ == "__main__":

        try:
            df, numeric_cols = run_stage1()

        except Exception as e:

            print("\n" + "=" * 70)
            print("STAGE 1 FAILED")
            print("=" * 70)

            print(f"\nError:\n{e}")

            import traceback
            traceback.print_exc()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df, numeric_cols = run_stage1()
    print(f"\nStage 1 complete. Shape: {df.shape}")
    print(f"Saved: {OUTPUT_CLEAN}")
