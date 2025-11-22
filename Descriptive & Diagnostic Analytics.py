# describe_and_diagnose.py
# ACCT 745 — Descriptive & Diagnostic analytics (figures + summary tables)
# Author: Long Dang (refined)
#
# Inputs:
#   Uses the CLEANED / TRANSFORMED core dataset (non-financial industries)
#   from step 2: _output_clean/_fact_cleaned_with_metrics.csv
#
# Key refinements vs earlier version:
#   - Uses winsorized metrics (dso_approx_w, ar_turnover_w, prov_rate_w,
#     writeoff_rate_w, liq_current_ratio_w, lev_debt_to_assets_w) to avoid
#     distortion from extreme outliers.
#   - The underlying cleaned file already excludes financial services.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =============
# 1) SETTINGS
# =============
BASE_DIR = Path(r"F:\Master Resources\ACCT.745.01 - Acctg Info. & Analytics\Project 1\(1) Accounts Receivables Provision Data")
CLEAN_DIR = BASE_DIR / "_output_clean"
ANALYSIS_DIR = BASE_DIR / "_output_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Core (non-financial) sample from the refined cleaning step
FACT_FILE = CLEAN_DIR / "_fact_cleaned_with_metrics.csv"

# ----------------------
# PLOT STYLE
# ----------------------
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =====================
# 2) LOAD CLEANED DATA
# =====================
def load_fact(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned file not found: {path}\n"
            f"Run the Cleaning, Transformation, and Summary script first."
        )
    return pd.read_csv(path, low_memory=False)

df = load_fact(FACT_FILE)

# Ensure expected columns exist (created in the cleaning step)
required_cols = {
    "company_code", "date", "year", "industry_code", "industry_name",
    "sales", "accounts_receivables", "receivables_morethan6m", "receivables_lessthan6m",
    "provision_bad_receivables", "writeoff_bad_receivables", "cfo", "net_income",
    # raw ratios
    "ar_turnover", "dso_approx", "prov_rate", "writeoff_rate",
    "liq_current_ratio", "lev_debt_to_assets",
    # winsorized ratios
    "dso_approx_w", "ar_turnover_w", "prov_rate_w", "writeoff_rate_w",
    "liq_current_ratio_w", "lev_debt_to_assets_w",
    # flags
    "ar_comp_ok", "ni_pos_cfo_neg_flag"
}
missing = required_cols.difference(df.columns)
if missing:
    raise KeyError(f"Missing expected columns in cleaned file: {missing}")

# Ensure types
if not np.issubdtype(df["year"].dtype, np.number):
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

# If date was written as string, we don't strictly need to parse it here,
# but it doesn't hurt for potential extensions:
# df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Helper AR composition shares (only when AR > 0 to avoid weird ratios)
df["accounts_receivables"] = pd.to_numeric(df["accounts_receivables"], errors="coerce")
mask_ar_pos = df["accounts_receivables"] > 0

df["pct_over6m"] = np.nan
df.loc[mask_ar_pos, "pct_over6m"] = (
    df.loc[mask_ar_pos, "receivables_morethan6m"] /
    df.loc[mask_ar_pos, "accounts_receivables"]
)

df["pct_under6m"] = np.nan
df.loc[mask_ar_pos, "pct_under6m"] = (
    df.loc[mask_ar_pos, "receivables_lessthan6m"] /
    df.loc[mask_ar_pos, "accounts_receivables"]
)

# Filter helper for plotting (avoid +/- inf)
def finite_series(s: pd.Series):
    return s.replace([np.inf, -np.inf], np.nan)

# ==================================
# 3) DESCRIPTIVE SUMMARY ARTIFACTS
# ==================================
# Overall KPIs — use winsorized DSO & AR turnover for stability
overall = pd.Series({
    "rows": len(df),
    "companies": df["company_code"].nunique(),
    "years_min": int(df["year"].min()),
    "years_max": int(df["year"].max()),
    "median_dso_w": finite_series(df["dso_approx_w"]).median(skipna=True),
    "median_ar_turnover_w": finite_series(df["ar_turnover_w"]).median(skipna=True),
    "prov_coverage_share": df["provision_bad_receivables"].notna().mean(),
    "writeoff_coverage_share": df["writeoff_bad_receivables"].notna().mean(),
    "ar_comp_ok_rate": pd.to_numeric(df["ar_comp_ok"], errors="coerce").mean(),
    "ni_pos_cfo_neg_rate": df["ni_pos_cfo_neg_flag"].mean(),
}).to_frame("value")
overall.to_csv(ANALYSIS_DIR / "_overall_kpis.csv")

# Industry × Year medians / coverage (using winsorized DSO & turnover)
indyr_medians = (
    df.groupby(["industry_code", "industry_name", "year"], dropna=False)
      .agg(
          med_sales=("sales", "median"),
          med_ar=("accounts_receivables", "median"),
          med_dso_w=("dso_approx_w", "median"),
          med_ar_turnover_w=("ar_turnover_w", "median"),
          med_pct_over6m=("pct_over6m", "median"),
          prov_cov=("provision_bad_receivables", lambda s: s.notna().mean()),
          writeoff_cov=("writeoff_bad_receivables", lambda s: s.notna().mean()),
      )
      .reset_index()
)
indyr_medians.to_csv(ANALYSIS_DIR / "_indyear_medians_coverage.csv", index=False)

# ==================================
# 4) FIGURES — DESCRIPTIVE ANALYTICS
# ==================================
def savefig(name: str):
    out = ANALYSIS_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out}")

# 4.1 DSO distribution (using winsorized DSO, trimmed at 500 days)
dso_w = finite_series(df["dso_approx_w"])
dso_w = dso_w[(dso_w > 0) & (dso_w < 500)]
plt.figure()
plt.hist(dso_w.dropna(), bins=60)
plt.title("Distribution of DSO (winsorized; truncated at 500 days)")
plt.xlabel("Days")
plt.ylabel("Frequency")
savefig("fig_dso_hist.png")

# 4.2 AR Turnover by Year (winsorized, boxplot, filtered)
turn_w = finite_series(df["ar_turnover_w"])
sub = df.loc[turn_w.between(0, 50, inclusive="neither"), ["year", "ar_turnover_w"]].dropna()
plt.figure()
sub.boxplot(column="ar_turnover_w", by="year")
plt.suptitle("")
plt.title("AR Turnover by Year (winsorized, 0 < x < 50)")
plt.xlabel("Year")
plt.ylabel("Turnover (times)")
savefig("fig_ar_turnover_box_by_year.png")

# 4.3 Provisioning coverage by industry (top 15 industries by companies)
top_ind = (
    df.groupby("industry_name")["company_code"]
      .nunique()
      .sort_values(ascending=False)
      .head(15)
      .index
      .tolist()
)

cov = (
    df[df["industry_name"].isin(top_ind)]
      .groupby("industry_name")
      .agg(
          prov_cov=("provision_bad_receivables", lambda s: s.notna().mean()),
          writeoff_cov=("writeoff_bad_receivables", lambda s: s.notna().mean()),
      )
      .sort_values("prov_cov", ascending=False)
)

plt.figure()
plt.barh(cov.index, cov["prov_cov"])
plt.title("Provision Field Coverage by Industry (Top 15 by companies)")
plt.xlabel("Coverage (share of rows with non-null)")
savefig("fig_provision_coverage_by_industry.png")

plt.figure()
plt.barh(cov.index, cov["writeoff_cov"])
plt.title("Write-off Field Coverage by Industry (Top 15 by companies)")
plt.xlabel("Coverage (share of rows with non-null)")
savefig("fig_writeoff_coverage_by_industry.png")

# 4.4 Composition: Median % AR > 6 months by industry (top 15)
comp = (
    df[df["industry_name"].isin(top_ind)]
      .groupby("industry_name")
      .agg(med_pct_over6m=("pct_over6m", "median"))
      .sort_values("med_pct_over6m", ascending=False)
)
plt.figure()
plt.barh(comp.index, comp["med_pct_over6m"])
plt.title("Median % of AR > 6 months by Industry (Top 15 by companies)")
plt.xlabel("Median Share of AR > 6m")
savefig("fig_pct_over6m_by_industry.png")

# ==================================
# 5) FIGURES — DIAGNOSTIC ANALYTICS
# ==================================
# 5.1 Scatter: DSO vs Provision Rate (winsorized, trimmed to sensible range)
tmp = df[["dso_approx_w", "prov_rate_w"]].copy()
tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
tmp = tmp[(tmp["dso_approx_w"] > 0) & (tmp["dso_approx_w"] < 400)]
# For clarity in the scatter, focus on provision rates in [0, 1)
tmp = tmp[(tmp["prov_rate_w"] >= 0) & (tmp["prov_rate_w"] < 1)]

plt.figure()
plt.scatter(tmp["dso_approx_w"], tmp["prov_rate_w"], s=8, alpha=0.3)
# Simple trendline
if len(tmp) > 50:
    m, b = np.polyfit(tmp["dso_approx_w"], tmp["prov_rate_w"], 1)
    x = np.linspace(tmp["dso_approx_w"].min(), tmp["dso_approx_w"].max(), 200)
    plt.plot(x, m * x + b)
plt.title("Relationship: DSO (winsorized) vs Provision Rate (winsorized)")
plt.xlabel("DSO (approx, winsorized)")
plt.ylabel("Provision Rate (winsorized)")
savefig("fig_scatter_dso_vs_provrate.png")

# 5.2 Scatter: DSO vs Write-off Rate (winsorized)
tmp2 = df[["dso_approx_w", "writeoff_rate_w"]].copy()
tmp2 = tmp2.replace([np.inf, -np.inf], np.nan).dropna()
tmp2 = tmp2[(tmp2["dso_approx_w"] > 0) & (tmp2["dso_approx_w"] < 400)]
tmp2 = tmp2[(tmp2["writeoff_rate_w"] >= 0) & (tmp2["writeoff_rate_w"] < 1)]

plt.figure()
plt.scatter(tmp2["dso_approx_w"], tmp2["writeoff_rate_w"], s=8, alpha=0.3)
if len(tmp2) > 50:
    m, b = np.polyfit(tmp2["dso_approx_w"], tmp2["writeoff_rate_w"], 1)
    x = np.linspace(tmp2["dso_approx_w"].min(), tmp2["dso_approx_w"].max(), 200)
    plt.plot(x, m * x + b)
plt.title("Relationship: DSO (winsorized) vs Write-off Rate (winsorized)")
plt.xlabel("DSO (approx, winsorized)")
plt.ylabel("Write-off Rate (winsorized)")
savefig("fig_scatter_dso_vs_writeoffrate.png")

# 5.3 Diagnostic: NI>0 & CFO<0 rate by industry (top 15)
neg_cash = (
    df[df["industry_name"].isin(top_ind)]
      .groupby("industry_name")["ni_pos_cfo_neg_flag"]
      .mean()
      .sort_values(ascending=False)
)
plt.figure()
plt.barh(neg_cash.index, neg_cash.values)
plt.title("Share of NI>0 & CFO<0 by Industry (Top 15 by companies)")
plt.xlabel("Share of observations")
savefig("fig_nipos_cfoneg_by_industry.png")

# ==================================
# 6) CORRELATION & EXPORTS
# ==================================
# Correlation matrix across key metrics (winsorized)
corr_cols = [
    "dso_approx_w", "ar_turnover_w", "prov_rate_w", "writeoff_rate_w",
    "liq_current_ratio_w", "lev_debt_to_assets_w", "pct_over6m"
]

corr_df = (
    df[corr_cols]
      .replace([np.inf, -np.inf], np.nan)
      .corr(method="pearson")
)
corr_df.to_csv(ANALYSIS_DIR / "_corr_metrics.csv")

# Heatmap with blue↔white↔red gradient centered at 0
plt.figure(figsize=(10, 8))
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = plt.imshow(corr_df.values, cmap="coolwarm", norm=norm)

# Ticks & labels
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)

# Annotate each cell with r (2 decimals)
n = len(corr_cols)
for i in range(n):
    for j in range(n):
        val = float(corr_df.values[i, j])
        text_color = "white" if abs(val) >= 0.6 else "black"
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)

# Colorbar
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r")

plt.title("Correlation Heatmap — Key Metrics (winsorized)")
plt.tight_layout()
savefig("fig_corr_heatmap.png")

# ==================================
# 7) OPTIONAL READ-ME STYLE SUMMARY
# ==================================
summary_lines = [
    "ACCT-745 Descriptive & Diagnostic Analytics — Summary",
    "----------------------------------------------------",
    f"Input cleaned file (core non-financial sample): {FACT_FILE.name}",
    f"Rows: {len(df):,} | Companies: {df['company_code'].nunique():,} | Years: {int(df['year'].min())}-{int(df['year'].max())}",
    f"Median DSO (winsorized): {finite_series(df['dso_approx_w']).median(skipna=True):.2f}",
    f"Median AR Turnover (winsorized): {finite_series(df['ar_turnover_w']).median(skipna=True):.2f}",
    f"Provision coverage: {df['provision_bad_receivables'].notna().mean():.2%} | Write-off coverage: {df['writeoff_bad_receivables'].notna().mean():.2%}",
    f"AR composition ok rate: {pd.to_numeric(df['ar_comp_ok'], errors='coerce').mean():.2%}",
    f"NI>0 & CFO<0 rate: {df['ni_pos_cfo_neg_flag'].mean():.2%}",
    "",
    "Notes:",
    "  - DSO, AR Turnover, Provision Rate, Write-off Rate, Current Ratio, and Leverage",
    "    are analyzed using winsorized versions to reduce the impact of extreme outliers.",
    "  - Financial services industry observations have already been excluded in the",
    "    cleaned input file, so this analysis focuses on non-financial industries.",
    "",
    "Artifacts created:",
    "  - _overall_kpis.csv",
    "  - _indyear_medians_coverage.csv",
    "  - _corr_metrics.csv",
    "  - Figures: PNGs for DSO distribution, AR Turnover by Year, coverage charts,",
    "    composition, scatter diagnostics, NI>0 & CFO<0 rates, and correlation heatmap.",
]
with open(ANALYSIS_DIR / "_README_analysis.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\nAnalysis complete. Outputs written to:", ANALYSIS_DIR)
