# Predictive Modelling & Prescriptive Analysis
# STEP 1: Build a model-ready dataset on top of the cleaned fact table
#
# This script:
#   - Loads _output_clean/_fact_cleaned_with_metrics.csv (non-financial sample)
#   - Filters to a core modelling sample (AR material, excludes bad AR composition)
#   - Adds extra ratios (aging shares, log size, profitability, CFO/Assets)
#   - Winsorizes new ratios for stability
#   - Creates target availability & high-risk flags
#   - Imputes missing values in key predictors using industry-year medians
#   - Writes:
#       _output_model/_fact_model_ready.csv
#       _output_model/_fact_model_ready_provision.csv
#       _output_model/_fact_model_ready_writeoff.csv

from pathlib import Path
import pandas as pd
import numpy as np

# ===================
# 1) SETTINGS / PATHS
# ===================
BASE_DIR = Path(
    r"F:\Master Resources\ACCT.745.01 - Acctg Info. & Analytics\Project 1\(1) Accounts Receivables Provision Data"
)
CLEAN_DIR = BASE_DIR / "_output_clean"
MODEL_DIR = BASE_DIR / "_output_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FACT_FILE = CLEAN_DIR / "_fact_cleaned_with_metrics.csv"   # cleaned, non-financial sample
MODEL_READY_FILE = MODEL_DIR / "_fact_model_ready.csv"
MODEL_READY_PROV_FILE = MODEL_DIR / "_fact_model_ready_provision.csv"
MODEL_READY_WRITEOFF_FILE = MODEL_DIR / "_fact_model_ready_writeoff.csv"

# Use same winsorization convention as cleaning script
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


# =========================
# 2) HELPER FUNCTIONS
# =========================
def winsorize_series(
    s: pd.Series,
    lower: float = WINSOR_LOWER,
    upper: float = WINSOR_UPPER
) -> pd.Series:
    """
    Winsorize a numeric Series at given lower/upper quantiles.
    Operates only on finite, non-null values; others are left as-is.
    """
    s = pd.to_numeric(s, errors="coerce")
    out = s.copy()
    mask = out.notna() & np.isfinite(out)
    if not mask.any():
        return out

    clean = out[mask]
    q_low = clean.quantile(lower)
    q_high = clean.quantile(upper)

    if pd.isna(q_low) or pd.isna(q_high):
        return out

    out.loc[mask & (out < q_low)] = q_low
    out.loc[mask & (out > q_high)] = q_high
    return out


def load_fact(path: Path) -> pd.DataFrame:
    """
    Load the cleaned fact table and enforce basic numeric types.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned fact file not found: {path}\n"
            f"Run the Cleaning, Transformation, and Summary script first."
        )

    df = pd.read_csv(path, low_memory=False)

    # Basic coercions
    for col in [
        "year", "industry_code", "accounts_receivables", "sales",
        "total_asset", "net_income", "cfo",
        "receivables_morethan6m", "receivables_lessthan6m"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ni_pos_cfo_neg_flag" in df.columns:
        df["ni_pos_cfo_neg_flag"] = (
            pd.to_numeric(df["ni_pos_cfo_neg_flag"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return df


def build_model_ready(df: pd.DataFrame):
    """
    Create the model-ready dataset:
      - core sample filter
      - extra ratios
      - winsorization of new ratios
      - target flags
      - high-risk flags
      - industry-year median imputation for key predictors
    """
    df = df.copy()

    # -----------------------------
    # 1) Core modelling sample mask
    # -----------------------------
    # AR materiality: prefer the flag from cleaning; fallback to AR>0 & Sales>0.
    if "ar_material_flag" in df.columns:
        mask_material = df["ar_material_flag"].astype(bool)
    else:
        mask_material = (df["accounts_receivables"] > 0) & (df["sales"] > 0)

    # Exclude rows where AR composition is explicitly "bad" (ar_comp_ok == 0/False).
    ar_comp_numeric = pd.to_numeric(df.get("ar_comp_ok"), errors="coerce")
    mask_bad_comp = ar_comp_numeric.eq(0)

    core_mask = mask_material & ~mask_bad_comp
    df = df.loc[core_mask].copy()

    # ---------------------------
    # 2) AR ageing composition
    # ---------------------------
    mask_ar_pos = df["accounts_receivables"] > 0
    df["pct_over6m"] = np.nan
    df["pct_under6m"] = np.nan

    df.loc[mask_ar_pos, "pct_over6m"] = (
        df.loc[mask_ar_pos, "receivables_morethan6m"] /
        df.loc[mask_ar_pos, "accounts_receivables"]
    )
    df.loc[mask_ar_pos, "pct_under6m"] = (
        df.loc[mask_ar_pos, "receivables_lessthan6m"] /
        df.loc[mask_ar_pos, "accounts_receivables"]
    )

    # ---------------------------
    # 3) Additional ratios / size
    # ---------------------------
    # Log size (clip negatives at 0 for log1p)
    df["log_sales"] = np.log1p(df["sales"].clip(lower=0))
    df["log_total_assets"] = np.log1p(df["total_asset"].clip(lower=0))

    # Profitability: NI / Sales (only when Sales > 0)
    df["profit_margin"] = np.nan
    mask_sales_pos = df["sales"] > 0
    df.loc[mask_sales_pos, "profit_margin"] = (
        df.loc[mask_sales_pos, "net_income"] / df.loc[mask_sales_pos, "sales"]
    )

    # Cash flow to assets: CFO / Total Assets (only when Assets > 0)
    df["cfo_to_assets"] = np.nan
    mask_assets_pos = df["total_asset"] > 0
    df.loc[mask_assets_pos, "cfo_to_assets"] = (
        df.loc[mask_assets_pos, "cfo"] / df.loc[mask_assets_pos, "total_asset"]
    )

    # Winsorize new ratios for stability
    df["profit_margin_w"] = winsorize_series(df["profit_margin"])
    df["cfo_to_assets_w"] = winsorize_series(df["cfo_to_assets"])

    # ---------------------------
    # 4) Target availability flags
    # ---------------------------
    df["has_prov_rate_w"] = df.get("prov_rate_w").notna().astype(int)
    df["has_writeoff_rate_w"] = df.get("writeoff_rate_w").notna().astype(int)

    # High-risk flags based on 90th percentile thresholds
    prov_nonnull = df.loc[df["prov_rate_w"].notna(), "prov_rate_w"]
    writeoff_nonnull = df.loc[df["writeoff_rate_w"].notna(), "writeoff_rate_w"]

    prov_hi = prov_nonnull.quantile(0.90) if not prov_nonnull.empty else np.nan
    wrt_hi = writeoff_nonnull.quantile(0.90) if not writeoff_nonnull.empty else np.nan

    if not np.isnan(prov_hi):
        df["high_prov_flag"] = (df["prov_rate_w"] >= prov_hi).astype(int)
    else:
        df["high_prov_flag"] = 0

    if not np.isnan(wrt_hi):
        df["high_writeoff_flag"] = (df["writeoff_rate_w"] >= wrt_hi).astype(int)
    else:
        df["high_writeoff_flag"] = 0

    # -----------------------------------------------
    # 5) Key predictor set + missing-data imputation
    # -----------------------------------------------
    predictor_cols = [
        # existing winsorized metrics from cleaning step
        "dso_approx_w",
        "ar_turnover_w",
        "liq_current_ratio_w",
        "lev_debt_to_assets_w",
        # new ageing & scale metrics
        "pct_over6m",
        "pct_under6m",
        "log_sales",
        "log_total_assets",
        "profit_margin_w",
        "cfo_to_assets_w",
    ]

    # Ensure predictors exist
    missing_predictors = [c for c in predictor_cols if c not in df.columns]
    if missing_predictors:
        raise KeyError(
            "Model-ready step expects these columns but did not find them: "
            f"{missing_predictors}"
        )

    # Missingness indicators (before imputation)
    for col in predictor_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Industry-year median imputation
    group_keys = ["industry_code", "year"]
    grouped = df.groupby(group_keys)

    for col in predictor_cols:
        med_by_group = grouped[col].transform("median")
        global_med = df[col].median()
        df[col] = np.where(df[col].isna(), med_by_group, df[col])
        df[col] = df[col].fillna(global_med)

    thresholds = {
        "prov_hi_p90": float(prov_hi) if not np.isnan(prov_hi) else None,
        "writeoff_hi_p90": float(wrt_hi) if not np.isnan(wrt_hi) else None,
    }

    return df, thresholds


# =========================
# 3) MAIN EXECUTION
# =========================
def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 80)

    print(f"Loading cleaned fact file: {FACT_FILE}")
    df_fact = load_fact(FACT_FILE)
    print(f"  Loaded {len(df_fact):,} rows from cleaned non-financial sample.")

    df_model, thresholds = build_model_ready(df_fact)

    print("\nModel-ready sample (after filters):")
    print(f"  Rows: {len(df_model):,}")
    print(f"  Companies: {df_model['company_code'].nunique():,}")
    print(f"  Years: {int(df_model['year'].min())}-{int(df_model['year'].max())}")
    print(f"  Non-missing prov_rate_w: {int(df_model['has_prov_rate_w'].sum()):,}")
    print(f"  Non-missing writeoff_rate_w: {int(df_model['has_writeoff_rate_w'].sum()):,}")
    if thresholds['prov_hi_p90'] is not None:
        print(f"  High provision (90th pct) threshold: {thresholds['prov_hi_p90']:.4f}")
    if thresholds['writeoff_hi_p90'] is not None:
        print(f"  High write-off (90th pct) threshold: {thresholds['writeoff_hi_p90']:.4f}")

    # Save full model-ready dataset
    df_model.to_csv(MODEL_READY_FILE, index=False)
    print(f"\nFull model-ready dataset written to:\n  {MODEL_READY_FILE}")

    # Convenience subsets: only rows with non-missing targets
    df_model[df_model["has_prov_rate_w"] == 1].to_csv(MODEL_READY_PROV_FILE, index=False)
    df_model[df_model["has_writeoff_rate_w"] == 1].to_csv(MODEL_READY_WRITEOFF_FILE, index=False)
    print("Provision-target subset written to:")
    print(f"  {MODEL_READY_PROV_FILE}")
    print("Write-off-target subset written to:")
    print(f"  {MODEL_READY_WRITEOFF_FILE}")


if __name__ == "__main__":
    main()

# ============================================================
# STEP 2: PREDICTIVE MODELLING (Regression + Classification)
#   - Uses model-ready CSVs from Step 1
#   - Builds:
#       * Linear regression for prov_rate_w and writeoff_rate_w
#       * Logistic regression for high_prov_flag and high_writeoff_flag
#   - Saves:
#       * Coefficients / metrics to MODEL_OUTPUT_DIR
#       * Prediction-augmented datasets back to _output_model
# ============================================================

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Directory to store model summaries / metrics
MODEL_OUTPUT_DIR = MODEL_DIR / "model_outputs"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_feature_matrix(df: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    """
    Build the X design matrix for modelling:
      - core numeric features (already winsorized / imputed in Step 1)
      - missingness indicators for each numeric feature (if present)
      - ni_pos_cfo_neg_flag as an additional risk feature
      - industry and year dummies
    """
    parts = []

    # Core numeric predictors
    X_num = df[numeric_features].copy()
    parts.append(X_num)

    # Missingness indicators (created in Step 1)
    missing_cols = []
    for col in numeric_features:
        miss_col = f"{col}_missing"
        if miss_col in df.columns:
            missing_cols.append(miss_col)
    if missing_cols:
        parts.append(df[missing_cols])

    # Earnings-quality flag
    if "ni_pos_cfo_neg_flag" in df.columns:
        parts.append(df[["ni_pos_cfo_neg_flag"]])

    # Industry dummies
    if "industry_code" in df.columns:
        ind_dummies = pd.get_dummies(
            df["industry_code"].astype("Int64"),
            prefix="ind",
            drop_first=True,
        )
        parts.append(ind_dummies)

    # Year dummies
    if "year" in df.columns:
        year_dummies = pd.get_dummies(
            df["year"].astype("Int64"),
            prefix="year",
            drop_first=True,
        )
        parts.append(year_dummies)

    X = pd.concat(parts, axis=1)
    # Just in case any residual NaNs remain
    X = X.fillna(0.0)

    return X


def run_regression_model(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: list,
    model_label: str,
):
    """
    Linear regression for a continuous target (e.g., prov_rate_w, writeoff_rate_w).
    - Train/test split for evaluation
    - Refit on full data for final predictions
    - Returns:
        df_with_preds, metrics_df (1 row), coef_df
    """
    df = df.copy()

    # X / y
    X = _prepare_feature_matrix(df, numeric_features)
    y = df[target_col].astype(float).values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fit on train
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Predictions for evaluation
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)

    # Older sklearn does not support squared=False; compute RMSE manually
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = float(np.sqrt(mse_train))
    rmse_test = float(np.sqrt(mse_test))

    metrics = {
        "model": model_label,
        "target": target_col,
        "n_obs": int(len(df)),
        "n_features": int(X.shape[1]),
        "r2_train": float(r2_score(y_train, y_train_pred)),
        "r2_test": float(r2_score(y_test, y_test_pred)),
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
    }
    metrics_df = pd.DataFrame([metrics])

    coef_df = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "coef": reg.coef_,
            }
        )
        .sort_values("coef", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )

    # Refit on full data and generate predictions for all rows
    reg_full = LinearRegression()
    reg_full.fit(X, y)
    df[f"pred_{target_col}_linreg"] = reg_full.predict(X)

    return df, metrics_df, coef_df


def run_logistic_model(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: list,
    model_label: str,
):
    """
    Logistic regression for a binary target (e.g., high_prov_flag, high_writeoff_flag).
    - Train/test split (stratified) for evaluation
    - Refit on full data for final probabilities / class predictions
    - Returns:
        df_with_preds, metrics_df (1 row), coef_df, confusion_matrix, classification_report_str
    """
    df = df.copy()

    X = _prepare_feature_matrix(df, numeric_features)
    y = df[target_col].astype(int).values

    # Stratified split to respect base rate
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # Older sklearn may not support n_jobs in LogisticRegression, keep it simple
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Train / test predictions
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_train_pred = (y_train_proba >= 0.5).astype(int)
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    # Older versions don’t support zero_division, so use defaults
    metrics = {
        "model": model_label,
        "target": target_col,
        "n_obs": int(len(df)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "accuracy_train": float(accuracy_score(y_train, y_train_pred)),
        "accuracy_test": float(accuracy_score(y_test, y_test_pred)),
        "precision_test": float(precision_score(y_test, y_test_pred)),
        "recall_test": float(recall_score(y_test, y_test_pred)),
        "f1_test": float(f1_score(y_test, y_test_pred)),
        "roc_auc_test": float(roc_auc_score(y_test, y_test_proba)),
    }
    metrics_df = pd.DataFrame([metrics])

    coef_df = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "coef": clf.coef_[0],
            }
        )
        .sort_values("coef", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )

    # Confusion matrix & classification report on test set
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)

    # Refit on full data for final probabilities / predictions on all rows
    clf_full = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
    )
    clf_full.fit(X, y)
    proba_full = clf_full.predict_proba(X)[:, 1]
    df[f"proba_{target_col}_logit"] = proba_full
    df[f"pred_{target_col}_logit"] = (proba_full >= 0.5).astype(int)

    return df, metrics_df, coef_df, cm, report


def run_predictive_models_step2():
    """
    Orchestrator for Step 2:
      - Provision models (regression + classification)
      - Write-off models (regression + classification)
      - Writes:
          *_with_preds.csv, coefficient CSVs, metric CSVs, classification reports.
    """
    # Core numeric features engineered in Step 1
    numeric_features = [
        "dso_approx_w",
        "ar_turnover_w",
        "liq_current_ratio_w",
        "lev_debt_to_assets_w",
        "pct_over6m",
        "pct_under6m",
        "log_sales",
        "log_total_assets",
        "profit_margin_w",
        "cfo_to_assets_w",
    ]

    # ===========================
    # 2A. Provision rate models
    # ===========================
    print("\n=== STEP 2A: Provision rate models ===")
    df_prov = pd.read_csv(MODEL_READY_PROV_FILE, low_memory=False)
    print(f"Loaded provision modelling dataset: {len(df_prov):,} rows.")

    # Regression: prov_rate_w ~ features
    df_prov, reg_prov_metrics, reg_prov_coefs = run_regression_model(
        df_prov,
        target_col="prov_rate_w",
        numeric_features=numeric_features,
        model_label="linreg_provision",
    )

    # Classification: high_prov_flag ~ features
    if "high_prov_flag" in df_prov.columns:
        df_prov, clf_prov_metrics, clf_prov_coefs, prov_cm, prov_report = run_logistic_model(
            df_prov,
            target_col="high_prov_flag",
            numeric_features=numeric_features,
            model_label="logit_high_provision",
        )
    else:
        clf_prov_metrics = pd.DataFrame()
        clf_prov_coefs = pd.DataFrame()
        prov_cm = None
        prov_report = ""

    # Save provision dataset with predictions
    prov_pred_file = MODEL_DIR / "_fact_model_ready_provision_with_preds.csv"
    df_prov.to_csv(prov_pred_file, index=False)

    # Save provision model outputs
    reg_prov_coefs.to_csv(
        MODEL_OUTPUT_DIR / "reg_provision_coefficients.csv", index=False
    )
    reg_prov_metrics.to_csv(
        MODEL_OUTPUT_DIR / "reg_provision_metrics.csv", index=False
    )

    if not clf_prov_metrics.empty:
        clf_prov_coefs.to_csv(
            MODEL_OUTPUT_DIR / "clf_high_provision_coefficients.csv", index=False
        )
        clf_prov_metrics.to_csv(
            MODEL_OUTPUT_DIR / "clf_high_provision_metrics.csv", index=False
        )

        with open(
            MODEL_OUTPUT_DIR / "clf_high_provision_classification_report.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write("Confusion matrix (rows=true, cols=pred) on test set:\n")
            f.write(np.array2string(prov_cm))
            f.write("\n\nClassification report (test set):\n")
            f.write(prov_report)

    print("Saved provision models and predictions.")
    print(f"  Prediction-augmented file: {prov_pred_file}")

    # ===========================
    # 2B. Write-off rate models
    # ===========================
    print("\n=== STEP 2B: Write-off rate models ===")
    df_wrt = pd.read_csv(MODEL_READY_WRITEOFF_FILE, low_memory=False)
    print(f"Loaded write-off modelling dataset: {len(df_wrt):,} rows.")

    # Regression: writeoff_rate_w ~ features
    df_wrt, reg_wrt_metrics, reg_wrt_coefs = run_regression_model(
        df_wrt,
        target_col="writeoff_rate_w",
        numeric_features=numeric_features,
        model_label="linreg_writeoff",
    )

    # Classification: high_writeoff_flag ~ features
    if "high_writeoff_flag" in df_wrt.columns:
        df_wrt, clf_wrt_metrics, clf_wrt_coefs, wrt_cm, wrt_report = run_logistic_model(
            df_wrt,
            target_col="high_writeoff_flag",
            numeric_features=numeric_features,
            model_label="logit_high_writeoff",
        )
    else:
        clf_wrt_metrics = pd.DataFrame()
        clf_wrt_coefs = pd.DataFrame()
        wrt_cm = None
        wrt_report = ""

    # Save write-off dataset with predictions
    wrt_pred_file = MODEL_DIR / "_fact_model_ready_writeoff_with_preds.csv"
    df_wrt.to_csv(wrt_pred_file, index=False)

    # Save write-off model outputs
    reg_wrt_coefs.to_csv(
        MODEL_OUTPUT_DIR / "reg_writeoff_coefficients.csv", index=False
    )
    reg_wrt_metrics.to_csv(
        MODEL_OUTPUT_DIR / "reg_writeoff_metrics.csv", index=False
    )

    if not clf_wrt_metrics.empty:
        clf_wrt_coefs.to_csv(
            MODEL_OUTPUT_DIR / "clf_high_writeoff_coefficients.csv", index=False
        )
        clf_wrt_metrics.to_csv(
            MODEL_OUTPUT_DIR / "clf_high_writeoff_metrics.csv", index=False
        )

        with open(
            MODEL_OUTPUT_DIR / "clf_high_writeoff_classification_report.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write("Confusion matrix (rows=true, cols=pred) on test set:\n")
            f.write(np.array2string(wrt_cm))
            f.write("\n\nClassification report (test set):\n")
            f.write(wrt_report)

    print("Saved write-off models and predictions.")
    print(f"  Prediction-augmented file: {wrt_pred_file}")

    print("\nStep 2 predictive modelling completed.")
    print(f"  Model outputs directory: {MODEL_OUTPUT_DIR}")


# ------------------------------------------------------------------
# Second entry point: run Step 2 after Step 1 when you run the script
# (The first if __name__ == '__main__': main() for Step 1 is above.)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1 main() has already run via the first if-block.
    run_predictive_models_step2()

# ============================================================
# STEP 3: PRESCRIPTIVE ANALYSIS
#   - Uses prediction-augmented files from Step 2
#   - Uses regression & logistic coefficients
#   - Produces:
#       * Marginal effect tables (e.g., +10 days DSO)
#       * Scenario analysis (tighten vs loosen collections)
#       * DSO sensitivity plots for provision & write-off
#       * Risk probability scenarios from logistic models
# ============================================================

import matplotlib.pyplot as plt  # safe even if already imported

PRESC_DIR = MODEL_DIR / "_prescriptive_outputs"
PRESC_DIR.mkdir(parents=True, exist_ok=True)


def _get_coef(coef_df: pd.DataFrame, feature_name: str, default: float = 0.0) -> float:
    """
    Safely fetch a coefficient for a given feature from a coefficient DataFrame.
    """
    row = coef_df[coef_df["feature"] == feature_name]
    if not row.empty:
        return float(row["coef"].iloc[0])
    return float(default)


def run_prescriptive_analysis_step3():
    # ------------------------------
    # 1) Load model outputs & data
    # ------------------------------
    prov_pred_file = MODEL_DIR / "_fact_model_ready_provision_with_preds.csv"
    wrt_pred_file = MODEL_DIR / "_fact_model_ready_writeoff_with_preds.csv"

    if not prov_pred_file.exists() or not wrt_pred_file.exists():
        raise FileNotFoundError(
            "Prediction-augmented files from Step 2 not found.\n"
            "Run Step 1 + Step 2 first to generate:\n"
            f"  {prov_pred_file}\n"
            f"  {wrt_pred_file}"
        )

    print("\n=== STEP 3: Prescriptive analysis ===")
    print("Loading prediction-augmented datasets...")

    df_prov = pd.read_csv(prov_pred_file, low_memory=False)
    df_wrt = pd.read_csv(wrt_pred_file, low_memory=False)

    print(f"  Provision dataset: {len(df_prov):,} rows")
    print(f"  Write-off dataset: {len(df_wrt):,} rows")

    # Regression coefficients
    reg_prov_coef_file = MODEL_OUTPUT_DIR / "reg_provision_coefficients.csv"
    reg_wrt_coef_file = MODEL_OUTPUT_DIR / "reg_writeoff_coefficients.csv"

    reg_prov_coefs = pd.read_csv(reg_prov_coef_file)
    reg_wrt_coefs = pd.read_csv(reg_wrt_coef_file)

    # Logistic coefficients (may be absent if classification was skipped)
    clf_prov_coef_file = MODEL_OUTPUT_DIR / "clf_high_provision_coefficients.csv"
    clf_wrt_coef_file = MODEL_OUTPUT_DIR / "clf_high_writeoff_coefficients.csv"

    clf_prov_coefs = pd.read_csv(clf_prov_coef_file) if clf_prov_coef_file.exists() else None
    clf_wrt_coefs = pd.read_csv(clf_wrt_coef_file) if clf_wrt_coef_file.exists() else None

    # Coefficients we will actually use for prescriptive work
    beta_dso_prov = _get_coef(reg_prov_coefs, "dso_approx_w", default=0.0)
    beta_pct6m_prov = _get_coef(reg_prov_coefs, "pct_over6m", default=0.0)

    beta_dso_wrt = _get_coef(reg_wrt_coefs, "dso_approx_w", default=0.0)
    beta_pct6m_wrt = _get_coef(reg_wrt_coefs, "pct_over6m", default=0.0)

    print("\nKey regression coefficients (global):")
    print(f"  Provision model: beta_dso = {beta_dso_prov:.6f}, beta_pct_over6m = {beta_pct6m_prov:.6f}")
    print(f"  Writeoff model: beta_dso = {beta_dso_wrt:.6f}, beta_pct_over6m = {beta_pct6m_wrt:.6f}")

    # For risk scenarios (logistic), if available
    if clf_prov_coefs is not None:
        beta_dso_prov_logit = _get_coef(clf_prov_coefs, "dso_approx_w", default=0.0)
        beta_pct6m_prov_logit = _get_coef(clf_prov_coefs, "pct_over6m", default=0.0)
    else:
        beta_dso_prov_logit = 0.0
        beta_pct6m_prov_logit = 0.0

    if clf_wrt_coefs is not None:
        beta_dso_wrt_logit = _get_coef(clf_wrt_coefs, "dso_approx_w", default=0.0)
        beta_pct6m_wrt_logit = _get_coef(clf_wrt_coefs, "pct_over6m", default=0.0)
    else:
        beta_dso_wrt_logit = 0.0
        beta_pct6m_wrt_logit = 0.0

    # ------------------------------
    # 2) Marginal analysis: +/−10 DSO
    # ------------------------------
    print("\nBuilding marginal DSO effect tables...")

    median_ar_prov = df_prov["accounts_receivables"].median()
    median_ar_wrt = df_wrt["accounts_receivables"].median()

    delta_dso_values = [-10.0, 10.0]  # tighten vs loosen by 10 days

    marginal_rows = []
    for delta_dso in delta_dso_values:
        label = "tighten_10_days" if delta_dso < 0 else "loosen_10_days"

        delta_prov_rate = beta_dso_prov * delta_dso
        delta_wrt_rate = beta_dso_wrt * delta_dso

        delta_prov_amt_med = delta_prov_rate * median_ar_prov
        delta_wrt_amt_med = delta_wrt_rate * median_ar_wrt

        marginal_rows.append(
            {
                "scenario": label,
                "delta_dso_days": delta_dso,
                "delta_prov_rate": delta_prov_rate,
                "delta_writeoff_rate": delta_wrt_rate,
                "median_AR_for_prov": median_ar_prov,
                "median_AR_for_writeoff": median_ar_wrt,
                "delta_expected_provision_amount_at_median_AR": delta_prov_amt_med,
                "delta_expected_writeoff_amount_at_median_AR": delta_wrt_amt_med,
            }
        )

    df_marginal_global = pd.DataFrame(marginal_rows)
    df_marginal_global.to_csv(PRESC_DIR / "marginal_dso_global.csv", index=False)

    # Industry-level marginal impact (using same global beta_dso but industry median AR)
    if "industry_name" in df_prov.columns:
        ind_median_ar_prov = df_prov.groupby("industry_name")["accounts_receivables"].median()
    else:
        ind_median_ar_prov = pd.Series(dtype=float)

    if "industry_name" in df_wrt.columns:
        ind_median_ar_wrt = df_wrt.groupby("industry_name")["accounts_receivables"].median()
    else:
        ind_median_ar_wrt = pd.Series(dtype=float)

    ind_rows = []
    for delta_dso in delta_dso_values:
        label = "tighten_10_days" if delta_dso < 0 else "loosen_10_days"
        delta_prov_rate = beta_dso_prov * delta_dso
        delta_wrt_rate = beta_dso_wrt * delta_dso

        for ind in sorted(set(ind_median_ar_prov.index) | set(ind_median_ar_wrt.index)):
            med_ar_prov = float(ind_median_ar_prov.get(ind, np.nan))
            med_ar_wrt = float(ind_median_ar_wrt.get(ind, np.nan))

            ind_rows.append(
                {
                    "industry_name": ind,
                    "scenario": label,
                    "delta_dso_days": delta_dso,
                    "delta_prov_rate": delta_prov_rate,
                    "delta_writeoff_rate": delta_wrt_rate,
                    "median_AR_for_prov": med_ar_prov,
                    "median_AR_for_writeoff": med_ar_wrt,
                    "delta_expected_provision_amount_at_median_AR": delta_prov_rate * med_ar_prov
                    if not np.isnan(med_ar_prov)
                    else np.nan,
                    "delta_expected_writeoff_amount_at_median_AR": delta_wrt_rate * med_ar_wrt
                    if not np.isnan(med_ar_wrt)
                    else np.nan,
                }
            )

    df_marginal_industry = pd.DataFrame(ind_rows)
    df_marginal_industry.to_csv(PRESC_DIR / "marginal_dso_by_industry.csv", index=False)

    # ------------------------------
    # 3) Scenario analysis: base vs tighten vs loosen
    # ------------------------------
    print("Running scenario analysis...")

    # Global scenarios: change DSO and pct_over6m uniformly for all firms
    scenarios = {
        "base": {"delta_dso": 0.0, "delta_pct_over6m": 0.0},
        "tighten_15d_5pp": {"delta_dso": -15.0, "delta_pct_over6m": -0.05},
        "loosen_15d_5pp": {"delta_dso": 15.0, "delta_pct_over6m": 0.05},
    }

    # Apply scenarios for provision predictions
    for name, cfg in scenarios.items():
        delta_dso = cfg["delta_dso"]
        delta_pct = cfg["delta_pct_over6m"]

        # Adjust predicted provision rate
        prov_new = (
            df_prov["pred_prov_rate_w_linreg"]
            + beta_dso_prov * delta_dso
            + beta_pct6m_prov * delta_pct
        )
        prov_new = prov_new.clip(lower=0.0)  # no negative rates
        df_prov[f"prov_rate_scn_{name}"] = prov_new

        # Adjust predicted writeoff rate
        wrt_new = (
            df_wrt["pred_writeoff_rate_w_linreg"]
            + beta_dso_wrt * delta_dso
            + beta_pct6m_wrt * delta_pct
        )
        wrt_new = wrt_new.clip(lower=0.0)
        df_wrt[f"writeoff_rate_scn_{name}"] = wrt_new

    # Global scenario summary
    scen_rows = []
    for name in scenarios.keys():
        scen_rows.append(
            {
                "scenario": name,
                "mean_pred_prov_rate": df_prov[f"prov_rate_scn_{name}"].mean(),
                "median_pred_prov_rate": df_prov[f"prov_rate_scn_{name}"].median(),
                "mean_pred_writeoff_rate": df_wrt[f"writeoff_rate_scn_{name}"].mean(),
                "median_pred_writeoff_rate": df_wrt[f"writeoff_rate_scn_{name}"].median(),
            }
        )
    df_scen_global = pd.DataFrame(scen_rows)
    df_scen_global.to_csv(PRESC_DIR / "scenario_global_summary.csv", index=False)

    # Industry-level scenario summary (medians)
    if "industry_name" in df_prov.columns:
        scen_ind_rows_prov = []
        for name in scenarios.keys():
            tmp = (
                df_prov.groupby("industry_name")[f"prov_rate_scn_{name}"]
                .median()
                .reset_index()
            )
            tmp["scenario"] = name
            scen_ind_rows_prov.append(tmp)
        df_scen_ind_prov = pd.concat(scen_ind_rows_prov, ignore_index=True)
        df_scen_ind_prov.to_csv(
            PRESC_DIR / "scenario_industry_median_provision.csv", index=False
        )

    if "industry_name" in df_wrt.columns:
        scen_ind_rows_wrt = []
        for name in scenarios.keys():
            tmp = (
                df_wrt.groupby("industry_name")[f"writeoff_rate_scn_{name}"]
                .median()
                .reset_index()
            )
            tmp["scenario"] = name
            scen_ind_rows_wrt.append(tmp)
        df_scen_ind_wrt = pd.concat(scen_ind_rows_wrt, ignore_index=True)
        df_scen_ind_wrt.to_csv(
            PRESC_DIR / "scenario_industry_median_writeoff.csv", index=False
        )

    # Simple global bar charts for provision & write-off
    scen_order = list(scenarios.keys())
    med_prov = [
        df_scen_global.loc[df_scen_global["scenario"] == s, "median_pred_prov_rate"].iloc[0]
        for s in scen_order
    ]
    med_wrt = [
        df_scen_global.loc[df_scen_global["scenario"] == s, "median_pred_writeoff_rate"].iloc[0]
        for s in scen_order
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(scen_order, med_prov)
    plt.ylabel("Median predicted provision rate")
    plt.title("Global median provision rate by scenario")
    plt.tight_layout()
    plt.savefig(PRESC_DIR / "scenario_global_provision_median.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(scen_order, med_wrt)
    plt.ylabel("Median predicted write-off rate")
    plt.title("Global median write-off rate by scenario")
    plt.tight_layout()
    plt.savefig(PRESC_DIR / "scenario_global_writeoff_median.png", dpi=150)
    plt.close()

    # ------------------------------
    # 4) Sensitivity: DSO vs rates
    # ------------------------------
    print("Creating DSO sensitivity plots...")

    dso_med = df_prov["dso_approx_w"].median()
    prov_med = df_prov["pred_prov_rate_w_linreg"].median()
    wrt_med = df_wrt["pred_writeoff_rate_w_linreg"].median()

    dso_min = max(0.0, dso_med - 60)
    dso_max = dso_med + 60
    dso_grid = np.linspace(dso_min, dso_max, 50)

    # Linear relationship holding other factors at "typical" levels
    prov_grid = prov_med + beta_dso_prov * (dso_grid - dso_med)
    wrt_grid = wrt_med + beta_dso_wrt * (dso_grid - dso_med)
    prov_grid = np.clip(prov_grid, 0.0, None)
    wrt_grid = np.clip(wrt_grid, 0.0, None)

    plt.figure(figsize=(8, 5))
    plt.plot(dso_grid, prov_grid, label="Provision rate")
    plt.plot(dso_grid, wrt_grid, label="Write-off rate")
    plt.axvline(dso_med, linestyle="--", alpha=0.7, label="Current median DSO")
    plt.xlabel("DSO (days)")
    plt.ylabel("Predicted rate")
    plt.title("Sensitivity of expected provision/write-off to DSO")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PRESC_DIR / "sensitivity_dso_provision_writeoff.png", dpi=150)
    plt.close()

    # ------------------------------
    # 5) Risk probability scenarios (logistic)
    # ------------------------------
    if (clf_prov_coefs is not None) and ("proba_high_prov_flag_logit" in df_prov.columns):
        print("Running logistic-based risk probability scenarios (provision)...")
        # Invert base probabilities to logits
        p_base = df_prov["proba_high_prov_flag_logit"].clip(1e-4, 1 - 1e-4)
        logit_base = np.log(p_base / (1 - p_base))

        for name, cfg in scenarios.items():
            delta_dso = cfg["delta_dso"]
            delta_pct = cfg["delta_pct_over6m"]
            delta_logit = beta_dso_prov_logit * delta_dso + beta_pct6m_prov_logit * delta_pct
            logit_new = logit_base + delta_logit
            p_new = 1.0 / (1.0 + np.exp(-logit_new))
            df_prov[f"proba_high_prov_flag_logit_scn_{name}"] = p_new

    if (clf_wrt_coefs is not None) and ("proba_high_writeoff_flag_logit" in df_wrt.columns):
        print("Running logistic-based risk probability scenarios (write-off)...")
        p_base_w = df_wrt["proba_high_writeoff_flag_logit"].clip(1e-4, 1 - 1e-4)
        logit_base_w = np.log(p_base_w / (1 - p_base_w))

        for name, cfg in scenarios.items():
            delta_dso = cfg["delta_dso"]
            delta_pct = cfg["delta_pct_over6m"]
            delta_logit = beta_dso_wrt_logit * delta_dso + beta_pct6m_wrt_logit * delta_pct
            logit_new = logit_base_w + delta_logit
            p_new = 1.0 / (1.0 + np.exp(-logit_new))
            df_wrt[f"proba_high_writeoff_flag_logit_scn_{name}"] = p_new

    # Global summary of risk probabilities (if computed)
    risk_rows = []
    if "proba_high_prov_flag_logit" in df_prov.columns:
        risk_rows.append(
            {
                "target": "high_provision",
                "scenario": "base",
                "mean_probability": df_prov["proba_high_prov_flag_logit"].mean(),
            }
        )
        for name in scenarios.keys():
            col = f"proba_high_prov_flag_logit_scn_{name}"
            if col in df_prov.columns:
                risk_rows.append(
                    {
                        "target": "high_provision",
                        "scenario": name,
                        "mean_probability": df_prov[col].mean(),
                    }
                )

    if "proba_high_writeoff_flag_logit" in df_wrt.columns:
        risk_rows.append(
            {
                "target": "high_writeoff",
                "scenario": "base",
                "mean_probability": df_wrt["proba_high_writeoff_flag_logit"].mean(),
            }
        )
        for name in scenarios.keys():
            col = f"proba_high_writeoff_flag_logit_scn_{name}"
            if col in df_wrt.columns:
                risk_rows.append(
                    {
                        "target": "high_writeoff",
                        "scenario": name,
                        "mean_probability": df_wrt[col].mean(),
                    }
                )

    if risk_rows:
        df_risk_global = pd.DataFrame(risk_rows)
        df_risk_global.to_csv(PRESC_DIR / "risk_probability_global_summary.csv", index=False)

        # Simple bar chart for high-provision risk probabilities by scenario
        sub_prov = df_risk_global[df_risk_global["target"] == "high_provision"]
        if not sub_prov.empty:
            plt.figure(figsize=(8, 5))
            plt.bar(sub_prov["scenario"], sub_prov["mean_probability"])
            plt.ylabel("Mean probability of high provision risk")
            plt.title("Average high-provision risk by scenario")
            plt.tight_layout()
            plt.savefig(PRESC_DIR / "risk_prob_high_provision_by_scenario.png", dpi=150)
            plt.close()

        # Simple bar chart for high-writeoff risk probabilities by scenario
        sub_wrt = df_risk_global[df_risk_global["target"] == "high_writeoff"]
        if not sub_wrt.empty:
            plt.figure(figsize=(8, 5))
            plt.bar(sub_wrt["scenario"], sub_wrt["mean_probability"])
            plt.ylabel("Mean probability of high write-off risk")
            plt.title("Average high-writeoff risk by scenario")
            plt.tight_layout()
            plt.savefig(PRESC_DIR / "risk_prob_high_writeoff_by_scenario.png", dpi=150)
            plt.close()

    # Save augmented datasets (with scenario columns added)
    df_prov.to_csv(PRESC_DIR / "_fact_model_ready_provision_prescriptive.csv", index=False)
    df_wrt.to_csv(PRESC_DIR / "_fact_model_ready_writeoff_prescriptive.csv", index=False)

    print("\nStep 3 prescriptive analysis completed.")
    print(f"  Outputs written to directory:\n    {PRESC_DIR}")


# ------------------------------------------------------------------
# Third entry point: run Step 3 after Step 1 and Step 2 when you run
# the script. (Step 1 main() and Step 2 run_predictive_models_step2()
# are defined above with their own if __name__ == '__main__' blocks.)
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_prescriptive_analysis_step3()

# ============================================================
# STEP 3B: Additional prescriptive visuals
#   Extra charts on top of Step 3 outputs to improve readability:
#     1) Scenario deltas vs base (provision)
#     2) Scenario deltas vs base (write-off)
#     3) Top 10 industries by DSO-tightening impact (provision)
#     4) Distribution shift in high-provision risk (base vs tighten)
# ============================================================

def run_additional_prescriptive_visuals():
    print("\n=== STEP 3B: Additional prescriptive visuals ===")

    # 1 & 2) Scenario deltas vs base (provision & write-off)
    scen_file = PRESC_DIR / "scenario_global_summary.csv"
    if scen_file.exists():
        df_scen = pd.read_csv(scen_file)

        base_rows = df_scen[df_scen["scenario"] == "base"]
        if not base_rows.empty:
            base_row = base_rows.iloc[0]
            base_prov_med = base_row["median_pred_prov_rate"]
            base_wrt_med = base_row["median_pred_writeoff_rate"]

            df_delta = df_scen.copy()
            df_delta["delta_median_prov_vs_base"] = (
                df_delta["median_pred_prov_rate"] - base_prov_med
            )
            df_delta["delta_median_wrt_vs_base"] = (
                df_delta["median_pred_writeoff_rate"] - base_wrt_med
            )

            df_delta_no_base = df_delta[df_delta["scenario"] != "base"]

            # 1) Provision delta vs base
            plt.figure(figsize=(8, 5))
            plt.bar(
                df_delta_no_base["scenario"],
                df_delta_no_base["delta_median_prov_vs_base"],
            )
            plt.axhline(0.0, linestyle="--")
            plt.ylabel("Δ median predicted provision rate vs base")
            plt.title("Change in median provision rate by scenario")
            plt.tight_layout()
            plt.savefig(
                PRESC_DIR / "scenario_global_delta_provision_median.png", dpi=150
            )
            plt.close()

            # 2) Write-off delta vs base
            plt.figure(figsize=(8, 5))
            plt.bar(
                df_delta_no_base["scenario"],
                df_delta_no_base["delta_median_wrt_vs_base"],
            )
            plt.axhline(0.0, linestyle="--")
            plt.ylabel("Δ median predicted write-off rate vs base")
            plt.title("Change in median write-off rate by scenario")
            plt.tight_layout()
            plt.savefig(
                PRESC_DIR / "scenario_global_delta_writeoff_median.png", dpi=150
            )
            plt.close()
        else:
            print("  [WARN] No 'base' scenario row found in scenario_global_summary.csv")
    else:
        print("  [INFO] scenario_global_summary.csv not found; skipping scenario delta charts.")

    # 3) Top 10 industries by DSO-tightening impact on provision
    marginal_ind_file = PRESC_DIR / "marginal_dso_by_industry.csv"
    if marginal_ind_file.exists():
        df_marg_ind = pd.read_csv(marginal_ind_file)

        df_tight = df_marg_ind[df_marg_ind["scenario"] == "tighten_10_days"].copy()
        if not df_tight.empty:
            df_tight["abs_delta_prov_amt"] = df_tight[
                "delta_expected_provision_amount_at_median_AR"
            ].abs()

            # Top 10 by absolute dollar change at median AR
            df_top10 = df_tight.nlargest(10, "abs_delta_prov_amt")

            # Sort for nice horizontal plotting
            df_top10 = df_top10.sort_values(
                "delta_expected_provision_amount_at_median_AR"
            )

            plt.figure(figsize=(10, 6))
            plt.barh(
                df_top10["industry_name"],
                df_top10["delta_expected_provision_amount_at_median_AR"],
            )
            plt.axvline(0.0, linestyle="--")
            plt.xlabel("Δ expected provision amount at median AR (tighten 10 days)")
            plt.title(
                "Top 10 industries by provision impact from tightening DSO by 10 days"
            )
            plt.tight_layout()
            plt.savefig(
                PRESC_DIR / "marginal_dso_top10_industries_provision.png", dpi=150
            )
            plt.close()
        else:
            print("  [WARN] No 'tighten_10_days' rows found in marginal_dso_by_industry.csv")
    else:
        print("  [INFO] marginal_dso_by_industry.csv not found; skipping industry impact chart.")

    # 4) Distribution shift in high-provision risk (base vs tighten)
    prov_pres_file = PRESC_DIR / "_fact_model_ready_provision_prescriptive.csv"
    if prov_pres_file.exists():
        df_prov_pres = pd.read_csv(prov_pres_file)

        base_col = "proba_high_prov_flag_logit"
        tight_col = "proba_high_prov_flag_logit_scn_tighten_15d_5pp"

        if base_col in df_prov_pres.columns and tight_col in df_prov_pres.columns:
            base_probs = df_prov_pres[base_col].clip(0.0, 1.0).values
            tight_probs = df_prov_pres[tight_col].clip(0.0, 1.0).values

            plt.figure(figsize=(8, 5))
            plt.hist(
                base_probs,
                bins=30,
                alpha=0.5,
                density=True,
                label="base",
            )
            plt.hist(
                tight_probs,
                bins=30,
                alpha=0.5,
                density=True,
                label="tighten_15d_5pp",
            )
            plt.xlabel("Predicted probability of high provision risk")
            plt.ylabel("Density")
            plt.title("High-provision risk distribution: base vs tighten scenario")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                PRESC_DIR / "risk_prob_high_provision_hist_base_vs_tighten.png",
                dpi=150,
            )
            plt.close()
        else:
            print(
                "  [WARN] Needed columns for base/tighten high-provision probabilities "
                "not found in _fact_model_ready_provision_prescriptive.csv"
            )
    else:
        print(
            "  [INFO] _fact_model_ready_provision_prescriptive.csv not found; "
            "skipping risk probability histogram."
        )

    print("Additional prescriptive visuals saved in:")
    print(f"  {PRESC_DIR}")


# ------------------------------------------------------------------
# Final extra entry point: run the additional visuals after Step 3
# (Step 1 main(), Step 2 run_predictive_models_step2(), and
#  Step 3 run_prescriptive_analysis_step3() are defined above.)
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_additional_prescriptive_visuals()
