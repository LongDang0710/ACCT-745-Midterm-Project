# Dashboard_Data_Exports.py
# ACCT 745 — Export dashboard-ready CSV tables for Power BI
# Author: Long Dang (with help)
#
# This script builds a small "data mart" for Power BI from your existing outputs:
#   - Dimension tables  (dim_*)
#   - Core fact tables  (fact_*)
#   - Descriptive/diagnostic aggregates (dash_q*)
#   - Predictive & prescriptive aggregates (dash_* scenario / marginal / risk)
#
# It assumes you have already run:
#   1) Data Cleaning, Transformation, and Summary.py
#   2) Descriptive & Diagnostic Analytics.py
#   3) Predictive Modeling & Prescriptive Analysis.py (Step 2 + Step 3)

from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# 1) PATHS & OUTPUT FOLDER
# =========================

BASE_DIR = Path(
    r"F:\Master Resources\ACCT.745.01 - Acctg Info. & Analytics\Project 1\(1) Accounts Receivables Provision Data"
)

CLEAN_DIR = BASE_DIR / "_output_clean"
ANALYSIS_DIR = BASE_DIR / "_output_analysis"
MODEL_DIR = BASE_DIR / "_output_model"
PRESC_DIR = MODEL_DIR / "_prescriptive_outputs"

DASH_DIR = BASE_DIR / "_output_dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)

FACT_CLEAN_FILE = CLEAN_DIR / "_fact_cleaned_with_metrics.csv"


# =========================
# 2) HELPER FUNCTIONS
# =========================

def load_clean_fact() -> pd.DataFrame:
    """Load the main cleaned fact table."""
    if not FACT_CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned fact file not found: {FACT_CLEAN_FILE}\n"
            "Run 'Data Cleaning, Transformation, and Summary.py' first."
        )
    df = pd.read_csv(FACT_CLEAN_FILE, low_memory=False)
    # Basic type coercions
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV if it exists, else return None."""
    if not path.exists():
        print(f"[INFO] File not found, skipping: {path}")
        return None
    return pd.read_csv(path, low_memory=False)


def write_csv(df: pd.DataFrame, path: Path):
    """Write a DataFrame to CSV with a simple log."""
    df.to_csv(path, index=False)
    print(f"[OK] Wrote: {path}")


# ===================================
# 3) DIMENSION TABLE BUILDERS
# ===================================

def build_dim_company(df: pd.DataFrame) -> pd.DataFrame:
    required = {"company_code", "industry_code", "industry_name", "year", "sales", "accounts_receivables"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for dim_company: {missing}")

    dim = (
        df.groupby("company_code", dropna=False)
          .agg(
              industry_code=("industry_code", "first"),
              industry_name=("industry_name", "first"),
              first_year=("year", "min"),
              last_year=("year", "max"),
              n_years=("year", "nunique"),
              avg_sales=("sales", "mean"),
              avg_accounts_receivables=("accounts_receivables", "mean"),
          )
          .reset_index()
    )
    return dim


def build_dim_industry(df: pd.DataFrame) -> pd.DataFrame:
    required = {"industry_code", "industry_name"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for dim_industry: {missing}")

    dim = (
        df[["industry_code", "industry_name"]]
        .drop_duplicates()
        .sort_values(["industry_code", "industry_name"])
        .reset_index(drop=True)
    )
    return dim


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
    elif "year" in df.columns:
        # Fallback: assume 31 Dec for each year
        dates = pd.to_datetime(df["year"].astype("Int64").astype(str) + "-12-31", errors="coerce")
    else:
        raise KeyError("No 'date' or 'year' column available to build dim_date.")

    dim = pd.DataFrame({"date": dates}).dropna().drop_duplicates().reset_index(drop=True)
    dim["year"] = dim["date"].dt.year
    dim["quarter"] = dim["date"].dt.quarter
    dim["month"] = dim["date"].dt.month
    dim["month_name"] = dim["date"].dt.month_name()
    return dim


def build_dim_scenario() -> pd.DataFrame:
    rows = [
        {
            "scenario_id": "base",
            "scenario_label": "Base (no change)",
            "delta_dso_days": 0.0,
            "delta_pct_over6m": 0.0,
        },
        {
            "scenario_id": "tighten_15d_5pp",
            "scenario_label": "Tighten: -15 days DSO, -5pp >6m AR",
            "delta_dso_days": -15.0,
            "delta_pct_over6m": -0.05,
        },
        {
            "scenario_id": "loosen_15d_5pp",
            "scenario_label": "Loosen: +15 days DSO, +5pp >6m AR",
            "delta_dso_days": 15.0,
            "delta_pct_over6m": 0.05,
        },
        {
            "scenario_id": "tighten_10_days",
            "scenario_label": "Marginal: -10 days DSO",
            "delta_dso_days": -10.0,
            "delta_pct_over6m": 0.0,
        },
        {
            "scenario_id": "loosen_10_days",
            "scenario_label": "Marginal: +10 days DSO",
            "delta_dso_days": 10.0,
            "delta_pct_over6m": 0.0,
        },
    ]
    return pd.DataFrame(rows)


# ===================================
# 4) DESCRIPTIVE / DIAGNOSTIC TABLES
# ===================================

def build_q1_industry_year_aging(df: pd.DataFrame) -> pd.DataFrame:
    required = {"industry_code", "industry_name", "year", "dso_approx_w", "pct_over6m", "company_code"}
    missing = required.difference(df.columns)
    if missing:
        print(f"[WARN] Missing columns for Q1; skipping. Missing: {missing}")
        return pd.DataFrame()

    q1 = (
        df.groupby(["industry_code", "industry_name", "year"], dropna=False)
          .agg(
              med_dso=("dso_approx_w", "median"),
              med_pct_over6m=("pct_over6m", "median"),
              n_obs=("company_code", "size"),
              n_companies=("company_code", "nunique"),
          )
          .reset_index()
    )
    q1["rank_dso_in_year"] = q1.groupby("year")["med_dso"].rank(
        method="min", ascending=False
    )
    return q1


def build_q2_dso_deciles_rates(df: pd.DataFrame) -> pd.DataFrame:
    required = {"dso_approx_w", "prov_rate_w", "writeoff_rate_w"}
    missing = required.difference(df.columns)
    if missing:
        print(f"[WARN] Missing columns for Q2; skipping. Missing: {missing}")
        return pd.DataFrame()

    valid = df["dso_approx_w"].between(1, 400)
    df_q2 = df.loc[valid, ["dso_approx_w", "prov_rate_w", "writeoff_rate_w"]].dropna()
    if df_q2.empty:
        print("[INFO] Q2: not enough valid observations after filtering; skipping.")
        return pd.DataFrame()

    df_q2["dso_decile"] = pd.qcut(
        df_q2["dso_approx_w"], 10, labels=[f"D{i}" for i in range(1, 11)]
    )

    q2 = (
        df_q2.groupby("dso_decile")
             .agg(
                 n=("dso_approx_w", "size"),
                 med_dso=("dso_approx_w", "median"),
                 med_prov_rate=("prov_rate_w", "median"),
                 med_writeoff_rate=("writeoff_rate_w", "median"),
             )
             .reset_index()
             .sort_values("med_dso")
    )
    return q2


def build_q3_dso_deciles_cashflow(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"dso_approx_w", "ni_pos_cfo_neg_flag", "industry_name"}
    missing = required.difference(df.columns)
    if missing:
        print(f"[WARN] Missing columns for Q3; skipping. Missing: {missing}")
        return pd.DataFrame(), pd.DataFrame()

    valid = df["dso_approx_w"].between(1, 400)
    df_q3 = df.loc[valid, ["dso_approx_w", "ni_pos_cfo_neg_flag", "industry_name"]].dropna()
    if df_q3.empty:
        print("[INFO] Q3: not enough valid observations for DSO deciles; skipping.")
        return pd.DataFrame(), pd.DataFrame()

    df_q3["dso_decile"] = pd.qcut(
        df_q3["dso_approx_w"], 10, labels=[f"D{i}" for i in range(1, 11)]
    )

    q3a = (
        df_q3.groupby("dso_decile")["ni_pos_cfo_neg_flag"]
             .mean()
             .rename("share_nipos_cfoneg")
             .reset_index()
    )

    q3b = (
        df.groupby("industry_name")["ni_pos_cfo_neg_flag"]
          .mean()
          .rename("share_nipos_cfoneg")
          .reset_index()
          .sort_values("share_nipos_cfoneg", ascending=False)
    )
    return q3a, q3b


def build_q4_ar_composition(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "ar_comp_ok_num" in df.columns:
        ar_comp_ok_num = pd.to_numeric(df["ar_comp_ok_num"], errors="coerce")
    elif "ar_comp_ok" in df.columns:
        ar_comp_ok_num = pd.to_numeric(df["ar_comp_ok"], errors="coerce")
    else:
        print("[WARN] No AR composition flag found for Q4; skipping.")
        return pd.DataFrame(), pd.DataFrame()

    df_local = df.copy()
    df_local["ar_comp_ok_num"] = ar_comp_ok_num

    # Industry-year integrity
    q4a = (
        df_local.groupby(["industry_name", "year"], dropna=False)["ar_comp_ok_num"]
               .mean()
               .rename("ar_comp_ok_rate")
               .reset_index()
    )

    # Company-level failure rate
    comp_fail = (
        df_local.assign(ar_fail=1 - pd.to_numeric(df_local["ar_comp_ok_num"], errors="coerce"))
                 .groupby("company_code")["ar_fail"]
                 .mean()
                 .rename("ar_fail_rate")
                 .reset_index()
                 .sort_values("ar_fail_rate", ascending=False)
    )
    return q4a, comp_fail


def build_q5_year_trends(df: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "dso_approx_w", "ar_turnover_w", "company_code"}
    missing = required.difference(df.columns)
    if missing:
        print(f"[WARN] Missing columns for Q5; skipping. Missing: {missing}")
        return pd.DataFrame()

    q5 = (
        df.groupby("year", dropna=False)
          .agg(
              med_dso=("dso_approx_w", "median"),
              med_ar_turnover=("ar_turnover_w", "median"),
              n_obs=("company_code", "size"),
          )
          .reset_index()
          .sort_values("year")
    )
    return q5


def build_q6_liquidity_leverage(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"dso_approx_w", "liq_current_ratio_w", "lev_debt_to_assets_w"}
    missing = required.difference(df.columns)
    if missing:
        print(f"[WARN] Missing columns for Q6; skipping. Missing: {missing}")
        return pd.DataFrame(), pd.DataFrame()

    df_f = (
        df[["dso_approx_w", "liq_current_ratio_w", "lev_debt_to_assets_w"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if df_f.empty:
        print("[INFO] Q6: not enough valid observations; skipping.")
        return pd.DataFrame(), pd.DataFrame()

    def quartile_bins(s, name):
        return pd.qcut(
            s,
            4,
            labels=[
                f"{name} Q1 (low)",
                f"{name} Q2",
                f"{name} Q3",
                f"{name} Q4 (high)",
            ],
        )

    df_f["liq_bin"] = quartile_bins(df_f["liq_current_ratio_w"], "CR")
    df_f["lev_bin"] = quartile_bins(df_f["lev_debt_to_assets_w"], "Lev")

    q6a = (
        df_f.groupby("liq_bin")
             .agg(
                 median_dso=("dso_approx_w", "median"),
                 n=("dso_approx_w", "size"),
             )
             .reset_index()
    )

    q6b = (
        df_f.groupby("lev_bin")
             .agg(
                 median_dso=("dso_approx_w", "median"),
                 n=("dso_approx_w", "size"),
             )
             .reset_index()
    )

    return q6a, q6b


def build_q7_coverage(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"industry_name", "year", "provision_bad_receivables", "writeoff_bad_receivables", "company_code"}
    missing = required.difference(df.columns)
    if missing:
        print(f"[WARN] Missing columns for Q7; skipping. Missing: {missing}")
        return pd.DataFrame(), pd.DataFrame()

    q7 = (
        df.groupby(["industry_name", "year"], dropna=False)
          .agg(
              prov_cov=("provision_bad_receivables", lambda s: s.notna().mean()),
              writeoff_cov=("writeoff_bad_receivables", lambda s: s.notna().mean()),
              n_obs=("company_code", "size"),
          )
          .reset_index()
    )

    q7b = (
        q7.groupby("industry_name")
          .agg(
              avg_prov_cov=("prov_cov", "mean"),
              avg_writeoff_cov=("writeoff_cov", "mean"),
              years=("year", "nunique"),
          )
          .reset_index()
          .sort_values("avg_prov_cov")
    )

    return q7, q7b


def build_q8_fast_slow(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"year", "dso_approx_w", "industry_name", "company_code"}
    missing = required.difference(df.columns)
    if missing:
        print(f"[WARN] Missing columns for Q8; skipping. Missing: {missing}")
        return pd.DataFrame(), pd.DataFrame()

    latest_year = int(df["year"].max())
    df_latest = df[df["year"] == latest_year].copy()
    mask = (df_latest["dso_approx_w"] > 0) & (df_latest["dso_approx_w"] < 400)

    if not mask.any():
        print("[INFO] Q8: no valid DSO observations in latest year; skipping.")
        return pd.DataFrame(), pd.DataFrame()

    fast = (
        df_latest.loc[mask]
                 .groupby(["industry_name", "company_code"])["dso_approx_w"]
                 .median()
                 .reset_index()
                 .rename(columns={"dso_approx_w": "median_dso_w"})
                 .sort_values("median_dso_w")
                 .head(20)
    )
    slow = (
        df_latest.loc[mask]
                 .groupby(["industry_name", "company_code"])["dso_approx_w"]
                 .median()
                 .reset_index()
                 .rename(columns={"dso_approx_w": "median_dso_w"})
                 .sort_values("median_dso_w", ascending=False)
                 .head(20)
    )

    fast["year"] = latest_year
    slow["year"] = latest_year

    return fast, slow


# ===================================
# 5) MAIN ORCHESTRATOR
# ===================================

def main():
    print("\n=== Building dashboard data exports ===")
    df = load_clean_fact()

    # 5.1 Dimensions
    print("\n[STEP] Building dimension tables...")
    dim_company = build_dim_company(df)
    write_csv(dim_company, DASH_DIR / "dim_company.csv")

    dim_industry = build_dim_industry(df)
    write_csv(dim_industry, DASH_DIR / "dim_industry.csv")

    dim_date = build_dim_date(df)
    write_csv(dim_date, DASH_DIR / "dim_date.csv")

    dim_scenario = build_dim_scenario()
    write_csv(dim_scenario, DASH_DIR / "dim_scenario.csv")

    # 5.2 Core fact tables
    print("\n[STEP] Exporting core fact tables...")
    # Full cleaned fact table
    write_csv(df, DASH_DIR / "fact_ar_core.csv")

    # Model-ready with predictions (provision & write-off)
    prov_with_preds = safe_read_csv(MODEL_DIR / "_fact_model_ready_provision_with_preds.csv")
    if prov_with_preds is not None:
        write_csv(prov_with_preds, DASH_DIR / "fact_model_ready_provision.csv")

    wrt_with_preds = safe_read_csv(MODEL_DIR / "_fact_model_ready_writeoff_with_preds.csv")
    if wrt_with_preds is not None:
        write_csv(wrt_with_preds, DASH_DIR / "fact_model_ready_writeoff.csv")

    # Prescriptive augmented fact tables
    prov_pres = safe_read_csv(PRESC_DIR / "_fact_model_ready_provision_prescriptive.csv")
    if prov_pres is not None:
        write_csv(prov_pres, DASH_DIR / "fact_prescriptive_provision.csv")

    wrt_pres = safe_read_csv(PRESC_DIR / "_fact_model_ready_writeoff_prescriptive.csv")
    if wrt_pres is not None:
        write_csv(wrt_pres, DASH_DIR / "fact_prescriptive_writeoff.csv")

    # 5.3 Copy existing summary outputs (overall + industry-year)
    print("\n[STEP] Copying existing summary outputs...")
    overall_kpis = safe_read_csv(ANALYSIS_DIR / "_overall_kpis.csv")
    if overall_kpis is not None:
        write_csv(overall_kpis, DASH_DIR / "dash_overall_kpis.csv")

    indyr_medians = safe_read_csv(ANALYSIS_DIR / "_indyear_medians_coverage.csv")
    if indyr_medians is not None:
        write_csv(indyr_medians, DASH_DIR / "dash_industry_year_medians_coverage.csv")

    overall_numeric = safe_read_csv(CLEAN_DIR / "_overall_numeric_summary.csv")
    if overall_numeric is not None:
        write_csv(overall_numeric.reset_index(), DASH_DIR / "dash_overall_numeric_summary.csv")

    indyr_profile = safe_read_csv(CLEAN_DIR / "_industry_year_profile.csv")
    if indyr_profile is not None:
        write_csv(indyr_profile, DASH_DIR / "dash_industry_year_profile.csv")

    # 5.4 Q&A style aggregates (recomputed)
    print("\n[STEP] Building Q&A-style aggregate tables...")

    q1 = build_q1_industry_year_aging(df)
    if not q1.empty:
        write_csv(q1, DASH_DIR / "dash_q1_industry_year_aging.csv")

    q2 = build_q2_dso_deciles_rates(df)
    if not q2.empty:
        write_csv(q2, DASH_DIR / "dash_q2_dso_deciles_rates.csv")

    q3a, q3b = build_q3_dso_deciles_cashflow(df)
    if not q3a.empty:
        write_csv(q3a, DASH_DIR / "dash_q3_dso_deciles_cashflow.csv")
    if not q3b.empty:
        write_csv(q3b, DASH_DIR / "dash_q3_industry_cashflow_quality.csv")

    q4a, comp_fail = build_q4_ar_composition(df)
    if not q4a.empty:
        write_csv(q4a, DASH_DIR / "dash_q4_industry_year_ar_composition.csv")
    if not comp_fail.empty:
        write_csv(comp_fail, DASH_DIR / "dash_q4_company_ar_fail_rate.csv")

    q5 = build_q5_year_trends(df)
    if not q5.empty:
        write_csv(q5, DASH_DIR / "dash_q5_year_trends_dso_turnover.csv")

    q6a, q6b = build_q6_liquidity_leverage(df)
    if not q6a.empty:
        write_csv(q6a, DASH_DIR / "dash_q6_liquidity_quartiles_dso.csv")
    if not q6b.empty:
        write_csv(q6b, DASH_DIR / "dash_q6_leverage_quartiles_dso.csv")

    q7, q7b = build_q7_coverage(df)
    if not q7.empty:
        write_csv(q7, DASH_DIR / "dash_q7_industry_year_coverage.csv")
    if not q7b.empty:
        write_csv(q7b, DASH_DIR / "dash_q7_industry_avg_coverage.csv")

    fast, slow = build_q8_fast_slow(df)
    if not fast.empty:
        write_csv(fast, DASH_DIR / "dash_q8_fast_collectors_latest.csv")
    if not slow.empty:
        write_csv(slow, DASH_DIR / "dash_q8_slow_collectors_latest.csv")

    # 5.5 Predictive & prescriptive aggregate outputs
    print("\n[STEP] Copying predictive & prescriptive aggregate outputs...")

    # Scenario summaries
    scen_global = safe_read_csv(PRESC_DIR / "scenario_global_summary.csv")
    if scen_global is not None:
        write_csv(scen_global, DASH_DIR / "dash_scenario_global_summary.csv")

    scen_ind_prov = safe_read_csv(PRESC_DIR / "scenario_industry_median_provision.csv")
    if scen_ind_prov is not None:
        write_csv(scen_ind_prov, DASH_DIR / "dash_scenario_industry_median_provision.csv")

    scen_ind_wrt = safe_read_csv(PRESC_DIR / "scenario_industry_median_writeoff.csv")
    if scen_ind_wrt is not None:
        write_csv(scen_ind_wrt, DASH_DIR / "dash_scenario_industry_median_writeoff.csv")

    # Marginal DSO effects
    marginal_global = safe_read_csv(PRESC_DIR / "marginal_dso_global.csv")
    if marginal_global is not None:
        write_csv(marginal_global, DASH_DIR / "dash_marginal_dso_global.csv")

    marginal_ind = safe_read_csv(PRESC_DIR / "marginal_dso_by_industry.csv")
    if marginal_ind is not None:
        write_csv(marginal_ind, DASH_DIR / "dash_marginal_dso_by_industry.csv")

    # Risk probability global summary
    risk_global = safe_read_csv(PRESC_DIR / "risk_probability_global_summary.csv")
    if risk_global is not None:
        write_csv(risk_global, DASH_DIR / "dash_risk_probability_global_summary.csv")

    print("\n=== Dashboard data export complete ===")
    print(f"All tables written under: {DASH_DIR}")


if __name__ == "__main__":
    main()
