# Presentation Visuals.py
# Extra visuals grouped into:
#   1) "My choice of visuals"
#   2) "Carlos's choice of visuals"
#
# Assumes Step 1–3 + 3B of Predictive & Prescriptive script already ran.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# ===================
# PATHS (same as main project)
# ===================

BASE_DIR = Path(
    r"F:\Master Resources\ACCT.745.01 - Acctg Info. & Analytics\Project 1\(1) Accounts Receivables Provision Data"
)
CLEAN_DIR = BASE_DIR / "_output_clean"
MODEL_DIR = BASE_DIR / "_output_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FACT_FILE = CLEAN_DIR / "_fact_cleaned_with_metrics.csv"          # cleaned, non-financial sample
MODEL_READY_FILE = MODEL_DIR / "_fact_model_ready.csv"
MODEL_READY_PROV_FILE = MODEL_DIR / "_fact_model_ready_provision.csv"
MODEL_READY_WRITEOFF_FILE = MODEL_DIR / "_fact_model_ready_writeoff.csv"

MODEL_OUTPUT_DIR = MODEL_DIR / "model_outputs"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prescriptive outputs (from Step 3)
PRESC_DIR = MODEL_DIR / "_prescriptive_outputs"
PRESC_DIR.mkdir(parents=True, exist_ok=True)

# Core numeric features used in modelling (same as Step 2)
NUMERIC_FEATURES = [
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


# ============================
# HELPER FUNCTIONS
# ============================

def _prepare_feature_matrix(df: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    """
    Same logic as in Predictive script:
      - core numeric features
      - missingness indicators
      - ni_pos_cfo_neg_flag
      - industry & year dummies
    """
    df = df.copy()
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
    X = X.fillna(0.0)
    return X


def _get_coef_from_csv(csv_path: Path, feature_name: str, default: float = 0.0) -> float:
    if not csv_path.exists():
        return float(default)
    coef_df = pd.read_csv(csv_path)
    row = coef_df[coef_df["feature"] == feature_name]
    if not row.empty:
        return float(row["coef"].iloc[0])
    return float(default)


def _scatter_pred_vs_actual(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    title: str,
    filename: Path,
    sample_n: int = 5000,
):
    # Drop missing, sample if needed
    sub = df[[actual_col, pred_col]].dropna()
    if sub.empty:
        print(f"  [WARN] No non-missing data for {actual_col} vs {pred_col}; skipping.")
        return

    if len(sub) > sample_n:
        sub = sub.sample(n=sample_n, random_state=42)

    x = sub[actual_col].values
    y = sub[pred_col].values

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    line_min = min(x_min, y_min)
    line_max = max(x_max, y_max)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.25, edgecolor="none")
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--")
    plt.xlabel(f"Actual {actual_col}")
    plt.ylabel(f"Predicted {pred_col}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def _plot_top_coefficients(
    coef_file: Path,
    title: str,
    filename: Path,
    top_n: int = 10,
):
    if not coef_file.exists():
        print(f"  [INFO] {coef_file.name} not found; skipping '{title}'.")
        return

    coef_df = pd.read_csv(coef_file)
    if "coef" not in coef_df.columns or "feature" not in coef_df.columns:
        print(f"  [WARN] Unexpected structure in {coef_file.name}; skipping.")
        return

    # If not already sorted by |coef|, sort it
    coef_df = coef_df.sort_values("coef", key=lambda s: s.abs(), ascending=False)
    top = coef_df.head(top_n).copy()
    top = top.iloc[::-1]  # reverse for nicer horizontal plotting

    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["coef"])
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Coefficient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def _choose_scale(max_val: float):
    """
    Choose a nice scale for dollar-ish amounts.
    Returns (scale_factor, label_suffix).
    """
    if max_val >= 1e9:
        return 1e9, " (billions)"
    elif max_val >= 1e6:
        return 1e6, " (millions)"
    elif max_val >= 1e3:
        return 1e3, " (thousands)"
    else:
        return 1.0, ""


# ==========================================
# SECTION 1 — MY CHOICE OF VISUALS
# ==========================================

def my_visual_1_feature_importance_logit_high_provision():
    """
    Horizontal bar chart: logistic high_prov_flag top coefficients.
    Uses: MODEL_OUTPUT_DIR / 'clf_high_provision_coefficients.csv'
    """
    print("My Visual 1: Logistic feature importance (high_provision).")
    coef_file = MODEL_OUTPUT_DIR / "clf_high_provision_coefficients.csv"
    out_file = PRESC_DIR / "my_feat_importance_logit_high_provision_top10.png"
    _plot_top_coefficients(
        coef_file,
        "Top logistic coefficients – High provision risk",
        out_file,
        top_n=10,
    )


def my_visual_2_predicted_risk_vs_dso_decile():
    """
    Line chart: average predicted high_prov risk vs DSO decile.
    Uses: _fact_model_ready_provision_with_preds.csv
    """
    print("My Visual 2: Predicted high-provision risk vs DSO decile.")

    prov_pred_file = MODEL_DIR / "_fact_model_ready_provision_with_preds.csv"
    if not prov_pred_file.exists():
        print(f"  [INFO] {prov_pred_file} not found; run Step 2 first.")
        return

    df = pd.read_csv(prov_pred_file, low_memory=False)

    if "dso_approx_w" not in df.columns or "proba_high_prov_flag_logit" not in df.columns:
        print("  [WARN] Needed columns not found; skipping DSO decile plot.")
        return

    sub = df[["dso_approx_w", "proba_high_prov_flag_logit"]].dropna()
    if sub.empty:
        print("  [WARN] No non-missing DSO/probabilities; skipping.")
        return

    # Build deciles of DSO
    try:
        sub["dso_decile"] = pd.qcut(
            sub["dso_approx_w"],
            10,
            labels=False,
            duplicates="drop",
        ) + 1
    except ValueError:
        # fallback to 5 bins if qcut fails
        sub["dso_decile"] = pd.qcut(
            sub["dso_approx_w"],
            5,
            labels=False,
            duplicates="drop",
        ) + 1

    dec_group = (
        sub.groupby("dso_decile")["proba_high_prov_flag_logit"]
        .mean()
        .reset_index()
        .sort_values("dso_decile")
    )
    dec_group["dso_decile"] = dec_group["dso_decile"].astype(int)

    plt.figure(figsize=(7, 4))
    plt.plot(dec_group["dso_decile"], dec_group["proba_high_prov_flag_logit"], marker="o")
    plt.xticks(dec_group["dso_decile"])
    plt.xlabel("DSO decile (1 = fastest, higher = slower)")
    plt.ylabel("Avg predicted probability of high provision")
    plt.title("Predicted high-provision risk vs DSO decile")
    plt.tight_layout()
    plt.savefig(PRESC_DIR / "my_predicted_high_prov_vs_dso_decile.png", dpi=150)
    plt.close()


def my_visual_3_roc_curve_high_provision():
    """
    ROC curve for logistic high_prov_flag using the model-ready file.
    Re-runs logistic regression only for this diagnostic.
    Uses: MODEL_READY_PROV_FILE
    """
    print("My Visual 3: ROC curve for high_provision classifier.")

    if not MODEL_READY_PROV_FILE.exists():
        print(f"  [INFO] {MODEL_READY_PROV_FILE} not found; run Step 1 first.")
        return

    df = pd.read_csv(MODEL_READY_PROV_FILE, low_memory=False)
    if "high_prov_flag" not in df.columns:
        print("  [WARN] 'high_prov_flag' not found; skipping ROC.")
        return

    X = _prepare_feature_matrix(df, NUMERIC_FEATURES)
    y = df["high_prov_flag"].astype(int).values

    # Stratified split as in Step 2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_test_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc = roc_auc_score(y_test, y_test_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC – High provision classifier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PRESC_DIR / "my_roc_curve_high_provision.png", dpi=150)
    plt.close()


def my_visual_4_total_expected_provision_dollars_by_scenario():
    """
    Bar chart: total expected provision amount by scenario
    (Base vs Tighten vs Loosen) using prescriptive rates.
    Uses: _fact_model_ready_provision_prescriptive.csv
    """
    print("My Visual 4: Total expected provision dollars by scenario.")

    prov_pres_file = PRESC_DIR / "_fact_model_ready_provision_prescriptive.csv"
    if not prov_pres_file.exists():
        print(f"  [INFO] {prov_pres_file} not found; run Step 3 first.")
        return

    df = pd.read_csv(prov_pres_file, low_memory=False)
    required_cols = {"accounts_receivables",
                     "prov_rate_scn_base",
                     "prov_rate_scn_tighten_15d_5pp",
                     "prov_rate_scn_loosen_15d_5pp"}
    if not required_cols.issubset(df.columns):
        print("  [WARN] Needed scenario provision columns not found; skipping.")
        return

    scenarios = {
        "base": "prov_rate_scn_base",
        "tighten_15d_5pp": "prov_rate_scn_tighten_15d_5pp",
        "loosen_15d_5pp": "prov_rate_scn_loosen_15d_5pp",
    }

    rows = []
    for name, col in scenarios.items():
        amt = (df["accounts_receivables"] * df[col]).sum(skipna=True)
        rows.append({"scenario": name, "total_expected_provision": amt})

    res = pd.DataFrame(rows)
    max_val = res["total_expected_provision"].max()
    scale, suffix = _choose_scale(max_val)
    res["scaled_total"] = res["total_expected_provision"] / scale

    plt.figure(figsize=(7, 4))
    plt.bar(res["scenario"], res["scaled_total"])
    for idx, row in res.iterrows():
        plt.text(
            idx,
            row["scaled_total"],
            f"{row['scaled_total']:.2f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel("Total expected provision" + suffix)
    plt.title("Total expected provision amount by scenario")
    plt.tight_layout()
    plt.savefig(PRESC_DIR / "my_total_expected_provision_by_scenario.png", dpi=150)
    plt.close()


def my_visual_5_cfo_to_assets_vs_dso_deciles():
    """
    Line chart: median CFO/Assets vs DSO deciles (core model-ready sample).
    Uses: MODEL_READY_FILE
    """
    print("My Visual 5: CFO/Assets vs DSO deciles.")

    if not MODEL_READY_FILE.exists():
        print(f"  [INFO] {MODEL_READY_FILE} not found; run Step 1 first.")
        return

    df = pd.read_csv(MODEL_READY_FILE, low_memory=False)
    required = {"dso_approx_w", "cfo_to_assets_w"}
    if not required.issubset(df.columns):
        print("  [WARN] Needed columns not found; skipping CFO/Assets vs DSO chart.")
        return

    sub = df[["dso_approx_w", "cfo_to_assets_w"]].dropna()
    if sub.empty:
        print("  [WARN] No non-missing DSO/CFO; skipping.")
        return

    try:
        sub["dso_decile"] = pd.qcut(
            sub["dso_approx_w"],
            10,
            labels=False,
            duplicates="drop",
        ) + 1
    except ValueError:
        sub["dso_decile"] = pd.qcut(
            sub["dso_approx_w"],
            5,
            labels=False,
            duplicates="drop",
        ) + 1

    grp = (
        sub.groupby("dso_decile")["cfo_to_assets_w"]
        .median()
        .reset_index()
        .sort_values("dso_decile")
    )
    grp["dso_decile"] = grp["dso_decile"].astype(int)

    plt.figure(figsize=(7, 4))
    plt.plot(grp["dso_decile"], grp["cfo_to_assets_w"], marker="o")
    plt.xticks(grp["dso_decile"])
    plt.xlabel("DSO decile (1 = fastest, higher = slower)")
    plt.ylabel("Median CFO / Total assets")
    plt.title("Operating cash flow quality vs collection speed")
    plt.tight_layout()
    plt.savefig(PRESC_DIR / "my_cfo_to_assets_vs_dso_deciles.png", dpi=150)
    plt.close()


def generate_my_choice_visuals():
    print("\n==============================")
    print("SECTION: My choice of visuals")
    print("==============================")
    my_visual_1_feature_importance_logit_high_provision()
    my_visual_2_predicted_risk_vs_dso_decile()
    my_visual_3_roc_curve_high_provision()
    my_visual_4_total_expected_provision_dollars_by_scenario()
    my_visual_5_cfo_to_assets_vs_dso_deciles()


# ==========================================
# SECTION 2 — CARLOS'S CHOICE OF VISUALS
# ==========================================

def carlos_visual_1_pred_vs_actual_scatterplots():
    """
    Predicted vs actual scatterplots for provision and write-off.
    Uses: _fact_model_ready_provision_with_preds.csv,
           _fact_model_ready_writeoff_with_preds.csv
    """
    print("Carlos Visual 1: Predicted vs Actual scatterplots (provision & write-off).")

    prov_pred_file = MODEL_DIR / "_fact_model_ready_provision_with_preds.csv"
    wrt_pred_file = MODEL_DIR / "_fact_model_ready_writeoff_with_preds.csv"

    if prov_pred_file.exists():
        df_prov = pd.read_csv(prov_pred_file, low_memory=False)
        if {"prov_rate_w", "pred_prov_rate_w_linreg"}.issubset(df_prov.columns):
            _scatter_pred_vs_actual(
                df_prov,
                actual_col="prov_rate_w",
                pred_col="pred_prov_rate_w_linreg",
                title="Provision rate – Actual vs Predicted",
                filename=PRESC_DIR / "carlos_scatter_provision_actual_vs_pred.png",
            )
        else:
            print("  [WARN] Provision scatter: needed columns not found.")
    else:
        print(f"  [INFO] {prov_pred_file} not found; skipping provision scatter.")

    if wrt_pred_file.exists():
        df_wrt = pd.read_csv(wrt_pred_file, low_memory=False)
        if {"writeoff_rate_w", "pred_writeoff_rate_w_linreg"}.issubset(df_wrt.columns):
            _scatter_pred_vs_actual(
                df_wrt,
                actual_col="writeoff_rate_w",
                pred_col="pred_writeoff_rate_w_linreg",
                title="Write-off rate – Actual vs Predicted",
                filename=PRESC_DIR / "carlos_scatter_writeoff_actual_vs_pred.png",
            )
        else:
            print("  [WARN] Write-off scatter: needed columns not found.")
    else:
        print(f"  [INFO] {wrt_pred_file} not found; skipping write-off scatter.")


def carlos_visual_2_feature_importance_reg_and_logit():
    """
    Feature-importance style bar charts from both regression & logistic coefficients.
    Uses:
      - reg_provision_coefficients.csv
      - reg_writeoff_coefficients.csv
      - clf_high_provision_coefficients.csv
      - clf_high_writeoff_coefficients.csv
    """
    print("Carlos Visual 2: Feature-importance bar charts (regression & logistic).")

    # Regression coefficients (provision & write-off)
    _plot_top_coefficients(
        MODEL_OUTPUT_DIR / "reg_provision_coefficients.csv",
        "Top regression coefficients – Provision rate",
        PRESC_DIR / "carlos_feat_importance_reg_provision_top10.png",
        top_n=10,
    )
    _plot_top_coefficients(
        MODEL_OUTPUT_DIR / "reg_writeoff_coefficients.csv",
        "Top regression coefficients – Write-off rate",
        PRESC_DIR / "carlos_feat_importance_reg_writeoff_top10.png",
        top_n=10,
    )

    # Logistic coefficients (high_prov & high_writeoff)
    _plot_top_coefficients(
        MODEL_OUTPUT_DIR / "clf_high_provision_coefficients.csv",
        "Top logistic coefficients – High provision risk",
        PRESC_DIR / "carlos_feat_importance_logit_high_provision_top10.png",
        top_n=10,
    )
    _plot_top_coefficients(
        MODEL_OUTPUT_DIR / "clf_high_writeoff_coefficients.csv",
        "Top logistic coefficients – High write-off risk",
        PRESC_DIR / "carlos_feat_importance_logit_high_writeoff_top10.png",
        top_n=10,
    )


def carlos_visual_3_dso_sensitivity_line_chart():
    """
    DSO sensitivity line chart for provision & write-off rates.
    Recreates the Step 3 sensitivity plot.
    Uses:
      - reg_provision_coefficients.csv (beta_dso)
      - reg_writeoff_coefficients.csv (beta_dso)
      - _fact_model_ready_provision_with_preds.csv
      - _fact_model_ready_writeoff_with_preds.csv
    """
    print("Carlos Visual 3: DSO sensitivity line chart.")

    prov_pred_file = MODEL_DIR / "_fact_model_ready_provision_with_preds.csv"
    wrt_pred_file = MODEL_DIR / "_fact_model_ready_writeoff_with_preds.csv"
    if not (prov_pred_file.exists() and wrt_pred_file.exists()):
        print("  [INFO] Prediction-augmented files not found; skipping DSO sensitivity.")
        return

    df_prov = pd.read_csv(prov_pred_file, low_memory=False)
    df_wrt = pd.read_csv(wrt_pred_file, low_memory=False)

    # Load regression coefficients for DSO
    beta_dso_prov = _get_coef_from_csv(
        MODEL_OUTPUT_DIR / "reg_provision_coefficients.csv", "dso_approx_w", default=0.0
    )
    beta_dso_wrt = _get_coef_from_csv(
        MODEL_OUTPUT_DIR / "reg_writeoff_coefficients.csv", "dso_approx_w", default=0.0
    )

    if "dso_approx_w" not in df_prov.columns:
        print("  [WARN] dso_approx_w not in provision file; skipping.")
        return
    if "pred_prov_rate_w_linreg" not in df_prov.columns:
        print("  [WARN] pred_prov_rate_w_linreg not found; skipping.")
        return
    if "pred_writeoff_rate_w_linreg" not in df_wrt.columns:
        print("  [WARN] pred_writeoff_rate_w_linreg not found; skipping.")
        return

    dso_med = df_prov["dso_approx_w"].median()
    prov_med = df_prov["pred_prov_rate_w_linreg"].median()
    wrt_med = df_wrt["pred_writeoff_rate_w_linreg"].median()

    dso_min = max(0.0, dso_med - 60)
    dso_max = dso_med + 60
    dso_grid = np.linspace(dso_min, dso_max, 50)

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
    plt.savefig(PRESC_DIR / "carlos_sensitivity_dso_provision_writeoff.png", dpi=150)
    plt.close()


def carlos_visual_4_scenario_comparison_bar_charts():
    """
    Scenario comparison bar charts (Base vs Tighten vs Loosen)
    for median predicted provision and write-off rates.
    Uses: scenario_global_summary.csv
    """
    print("Carlos Visual 4: Scenario comparison bar charts.")

    scen_file = PRESC_DIR / "scenario_global_summary.csv"
    if not scen_file.exists():
        print("  [INFO] scenario_global_summary.csv not found; run Step 3 first.")
        return

    df_scen = pd.read_csv(scen_file)
    required = {
        "scenario",
        "median_pred_prov_rate",
        "median_pred_writeoff_rate",
    }
    if not required.issubset(df_scen.columns):
        print("  [WARN] scenario_global_summary missing needed columns; skipping.")
        return

    scen_order = ["base", "tighten_15d_5pp", "loosen_15d_5pp"]
    # fallback to whatever order exists if needed
    scen_available = [s for s in scen_order if s in set(df_scen["scenario"])]
    if not scen_available:
        scen_available = list(df_scen["scenario"].unique())

    med_prov = [
        df_scen.loc[df_scen["scenario"] == s, "median_pred_prov_rate"].iloc[0]
        for s in scen_available
    ]
    med_wrt = [
        df_scen.loc[df_scen["scenario"] == s, "median_pred_writeoff_rate"].iloc[0]
        for s in scen_available
    ]

    plt.figure(figsize=(7, 4))
    plt.bar(scen_available, med_prov)
    plt.ylabel("Median predicted provision rate")
    plt.title("Global median provision rate by scenario")
    plt.tight_layout()
    plt.savefig(PRESC_DIR / "carlos_scenario_global_provision_median.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(scen_available, med_wrt)
    plt.ylabel("Median predicted write-off rate")
    plt.title("Global median write-off rate by scenario")
    plt.tight_layout()
    plt.savefig(PRESC_DIR / "carlos_scenario_global_writeoff_median.png", dpi=150)
    plt.close()


def carlos_visual_5_industry_level_dso_tighten_impact():
    """
    Industry-level impact bars: change in expected provision amount
    when DSO tightens by 10 days (top 10 industries).
    Uses: marginal_dso_by_industry.csv
    """
    print("Carlos Visual 5: Industry-level impact from tightening DSO 10 days.")

    marginal_ind_file = PRESC_DIR / "marginal_dso_by_industry.csv"
    if not marginal_ind_file.exists():
        print("  [INFO] marginal_dso_by_industry.csv not found; run Step 3 first.")
        return

    df_marg = pd.read_csv(marginal_ind_file)
    if "scenario" not in df_marg.columns or \
       "delta_expected_provision_amount_at_median_AR" not in df_marg.columns:
        print("  [WARN] marginal_dso_by_industry missing needed columns; skipping.")
        return

    df_tight = df_marg[df_marg["scenario"] == "tighten_10_days"].copy()
    if df_tight.empty:
        print("  [WARN] No tighten_10_days rows; skipping.")
        return

    df_tight["abs_delta_prov_amt"] = df_tight[
        "delta_expected_provision_amount_at_median_AR"
    ].abs()

    df_top10 = df_tight.nlargest(10, "abs_delta_prov_amt")
    if df_top10.empty:
        print("  [WARN] No rows after top-10 filter; skipping.")
        return

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
    plt.title("Top 10 industries by provision impact from tightening DSO by 10 days")
    plt.tight_layout()
    plt.savefig(
        PRESC_DIR / "carlos_marginal_dso_top10_industries_provision.png",
        dpi=150,
    )
    plt.close()


def carlos_visual_6_high_risk_probability_distribution_shift():
    """
    High-risk probability distribution shift (base vs tighten scenario)
    for high_provision risk.
    Uses: _fact_model_ready_provision_prescriptive.csv
    """
    print("Carlos Visual 6: High-risk probability distribution shift (base vs tighten).")

    prov_pres_file = PRESC_DIR / "_fact_model_ready_provision_prescriptive.csv"
    if not prov_pres_file.exists():
        print("  [INFO] _fact_model_ready_provision_prescriptive.csv not found; run Step 3 first.")
        return

    df = pd.read_csv(prov_pres_file, low_memory=False)
    base_col = "proba_high_prov_flag_logit"
    tight_col = "proba_high_prov_flag_logit_scn_tighten_15d_5pp"

    if base_col not in df.columns or tight_col not in df.columns:
        print("  [WARN] Needed probability columns not found; skipping distribution shift.")
        return

    base_probs = df[base_col].clip(0.0, 1.0).values
    tight_probs = df[tight_col].clip(0.0, 1.0).values

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
        PRESC_DIR / "carlos_risk_prob_high_provision_hist_base_vs_tighten.png",
        dpi=150,
    )
    plt.close()


def generate_carlos_choice_visuals():
    print("\n==================================")
    print("SECTION: Carlos's choice of visuals")
    print("==================================")
    carlos_visual_1_pred_vs_actual_scatterplots()
    carlos_visual_2_feature_importance_reg_and_logit()
    carlos_visual_3_dso_sensitivity_line_chart()
    carlos_visual_4_scenario_comparison_bar_charts()
    carlos_visual_5_industry_level_dso_tighten_impact()
    carlos_visual_6_high_risk_probability_distribution_shift()


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    generate_my_choice_visuals()
    generate_carlos_choice_visuals()
    print("\nAll presentation visuals saved under:")
    print(f"  {PRESC_DIR}")
