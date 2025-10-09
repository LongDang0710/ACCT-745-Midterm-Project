# clean_and_summarize.py
# ACCT 745 — Data cleaning & statistics summary
# Author: Long Dang
# Purpose: Parse dates, de-duplicate at grain, standardize industry tables,
#          merge, compute metrics/flags, and export cleaned tables + summaries.

from pathlib import Path
import pandas as pd
import numpy as np
import re

# ============
# 1) SETTINGS (Remember to change the datasets path if you plan to use the code yourself)
# ============
BASE_DIR = Path(r"F:\Master Resources\ACCT.745.01 - Acctg Info. & Analytics\Project 1\(1) Accounts Receivables Provision Data")

FILES = {
    "provisions": "provisions for acct receivables.csv",
    "industry_data": "industry_data.csv",
    "industry_names": "industry names.csv",
}

OUT_DIR = BASE_DIR / "_output_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================
# 2) LOAD HELPERS
# ================
def safe_read_csv(path: Path) -> pd.DataFrame:
    """Robust CSV reader with common encodings."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)

def parse_stata_like_date(val):
    """Parse '31dec2019' → Timestamp('2019-12-31'); else NaT."""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip().lower()
    m = re.match(r"^(\d{1,2})([a-z]{3})(\d{4})$", s)
    if not m:
        return pd.NaT
    day, mon, year = m.groups()
    try:
        return pd.to_datetime(f"{day}-{mon}-{year}", format="%d-%b-%Y")
    except Exception:
        return pd.NaT

# =====================================
# 3) CLEANING / MERGE UTILS & CONSTANTS
# =====================================
AR_FIELDS = [
    "accounts_receivables",
    "receivables_morethan6m",
    "secured_receivables_morethan6m",
    "unsec_receivables_morethan6m",
    "receivables_lessthan6m",
    "secured_receivables_lessthan6m",
    "unsec_receivables_lessthan6m",
    "provision_bad_receivables",
    "writeoff_bad_receivables",
]

def dedup_at_company_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows at the (company_code, date) grain using:
      1) more non-null AR fields,
      2) higher sales,
      3) first occurrence as final tiebreaker.
    """
    df = df.copy()
    # Efficient non-null count across AR_FIELDS (missing columns are treated as NA)
    for c in AR_FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    df["_nn_ar"] = df[AR_FIELDS].notna().sum(axis=1)
    df["_sales_tie"] = df["sales"].astype(float).fillna(-np.inf)
    df = df.sort_values(["company_code", "date", "_nn_ar", "_sales_tie"],
                        ascending=[True, True, False, False])
    out = df.drop_duplicates(subset=["company_code", "date"], keep="first").copy()
    out.drop(columns=["_nn_ar", "_sales_tie"], inplace=True, errors="ignore")
    return out

def compute_metrics_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive/diagnostic metrics used in AR/provisioning analysis."""
    df = df.copy()
    # Core
    df["ar_turnover"] = df["sales"] / df["accounts_receivables"]
    df["dso_approx"] = 365 * df["accounts_receivables"] / df["sales"]
    df["prov_rate"] = df["provision_bad_receivables"] / df["accounts_receivables"]
    df["writeoff_rate"] = df["writeoff_bad_receivables"] / df["accounts_receivables"]
    df["liq_current_ratio"] = df["current_asset"] / df["current_liabilities"]
    df["lev_debt_to_assets"] = df["debt_total"] / df["total_asset"]
    df["ni_pos_cfo_neg_flag"] = np.where((df["net_income"] > 0) & (df["cfo"] < 0), 1, 0)

    # Composition integrity: (>6m + <6m) ≈ total AR (within 1% tolerance)
    a = df["receivables_morethan6m"].fillna(0)
    b = df["receivables_lessthan6m"].fillna(0)
    df["ar_bucket_sum"] = a + b
    # Only evaluate when total AR is present
    mask_eval = df["accounts_receivables"].notna() & df["ar_bucket_sum"].notna()
    df["ar_comp_ok"] = np.nan
    df.loc[mask_eval, "ar_comp_ok"] = np.isclose(
        df.loc[mask_eval, "accounts_receivables"],
        df.loc[mask_eval, "ar_bucket_sum"],
        rtol=0.01, atol=1e-6
    )
    return df

def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall numeric summary (count, mean, std, percentiles) for quick QA.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return pd.DataFrame()
    desc = df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    return desc

def summarize_by_industry_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Industry × Year profile: medians and coverage rates for key variables.
    """
    grouped = (df.groupby(["industry_code", "industry_name", "year"], dropna=False)
                 .agg(
                     n_obs=("company_code", "size"),
                     n_companies=("company_code", "nunique"),
                     med_sales=("sales", "median"),
                     med_ar=("accounts_receivables", "median"),
                     med_ar_turnover=("ar_turnover", "median"),
                     med_dso=("dso_approx", "median"),
                     prov_cov=("provision_bad_receivables", lambda s: s.notna().mean()),
                     writeoff_cov=("writeoff_bad_receivables", lambda s: s.notna().mean()),
                     ar_comp_ok_rate=("ar_comp_ok", lambda s: pd.to_numeric(s, errors="coerce").mean()),
                     ni_pos_cfo_neg_rate=("ni_pos_cfo_neg_flag", "mean"),
                 )
                 .reset_index())
    return grouped

# =====================
# 4) MAIN CLEANING RUN
# =====================
def main():
    # ---- Load
    provisions = safe_read_csv(BASE_DIR / FILES["provisions"])
    industry_data = safe_read_csv(BASE_DIR / FILES["industry_data"])
    industry_names = safe_read_csv(BASE_DIR / FILES["industry_names"])

    print("Loaded:")
    print("  provisions:", provisions.shape)
    print("  industry_data:", industry_data.shape)
    print("  industry_names:", industry_names.shape)

    # ---- Parse date fields
    if "date_var" in provisions.columns:
        provisions["date"] = provisions["date_var"].map(parse_stata_like_date)
        provisions["year"] = provisions["date"].dt.year
        provisions["quarter"] = provisions["date"].dt.quarter

    # ---- Ensure required keys exist
    required_keys = {"company_code", "date"}
    if not required_keys.issubset(provisions.columns):
        raise KeyError(f"Provisions missing required keys {required_keys} — found: {set(provisions.columns)}")

    # ---- De-duplicate at grain (company_code, date)
    dupes = provisions.duplicated(subset=["company_code", "date"], keep=False).sum()
    print(f"Duplicate rows at (company_code, date) BEFORE dedup: {int(dupes)}")
    provisions = dedup_at_company_date(provisions)
    dupes_after = provisions.duplicated(subset=["company_code", "date"], keep=False).sum()
    print(f"Duplicate rows AFTER dedup: {int(dupes_after)}")
    print("Rows after dedup:", provisions.shape[0])

    # ---- Standardize industry tables
    ind_data = industry_data.rename(columns={"new_industry_code": "industry_code",
                                             "company_id": "company_code"}).copy()
    ind_names = industry_names.rename(columns={"Industry code": "industry_code",
                                               "Industry Name": "industry_name"}).copy()

    # dtypes
    ind_data["company_code"] = ind_data["company_code"].astype("Int64")
    ind_data["industry_code"] = ind_data["industry_code"].astype("Int64")
    ind_names["industry_code"] = ind_names["industry_code"].astype("Int64")

    # Handle duplicate industry_code labels (keep first as canonical name)
    before = ind_names.shape[0]
    ind_names = ind_names.drop_duplicates(subset=["industry_code"], keep="first").copy()
    after = ind_names.shape[0]
    if after < before:
        print(f"Deduped industry_names: kept {after} unique codes (from {before}).")

    # ---- Merge: provisions → industry code → industry name
    fact = (provisions.merge(ind_data, on="company_code", how="left")
                      .merge(ind_names, on="industry_code", how="left"))

    # ---- Compute metrics & flags
    fact = compute_metrics_flags(fact)

    # ---- Final sample info
    n_rows = fact.shape[0]
    n_companies = fact["company_code"].nunique()
    years_min = int(fact["year"].min())
    years_max = int(fact["year"].max())
    pct_dec = float((fact["date"].dt.month.eq(12)).mean() * 100)
    pct_jun = float((fact["date"].dt.month.eq(6)).mean() * 100)
    prov_cov = float(fact["provision_bad_receivables"].notna().mean() * 100)
    writeoff_cov = float(fact["writeoff_bad_receivables"].notna().mean() * 100)

    print("\nFinal sample (post-clean):")
    print(f"  Rows: {n_rows:,}")
    print(f"  Companies: {n_companies:,}")
    print(f"  Years: {years_min}–{years_max}")
    print(f"  Date distribution: {pct_dec:.2f}% Dec 31, {pct_jun:.2f}% Jun 30")
    print(f"  Provision coverage: {prov_cov:.2f}% | Write-off coverage: {writeoff_cov:.2f}%")

    # ---- Summaries
    overall_numeric = summarize_overall(fact)
    by_industry_year = summarize_by_industry_year(fact)

    # ---- Save outputs
    fact_out = OUT_DIR / "_fact_cleaned_with_metrics.csv"
    overall_out = OUT_DIR / "_overall_numeric_summary.csv"
    prof_out = OUT_DIR / "_industry_year_profile.csv"

    fact.to_csv(fact_out, index=False)
    overall_numeric.to_csv(overall_out)
    by_industry_year.to_csv(prof_out, index=False)

    print("\nFiles written:")
    print(f"  Cleaned fact table: {fact_out}")
    print(f"  Overall numeric summary: {overall_out}")
    print(f"  Industry×Year profile: {prof_out}")

    # ---- Optional sanity snapshots
    fact.head(50).to_csv(OUT_DIR / "_sample_head_fact.csv", index=False)
    fact.tail(50).to_csv(OUT_DIR / "_sample_tail_fact.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 60)
    main()
