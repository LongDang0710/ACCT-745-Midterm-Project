# questions_and_answers_display.py
# ACCT 745 — Broad analytics questions + on-screen answers (tables & charts only)
# Input : _output_clean/_fact_cleaned_with_metrics.csv (from clean_and_summarize.py)
# Output: Printed tables in console + matplotlib figures (no files saved)

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# ============
# 1) SETTINGS (Remember to change the datasets path if you plan to use the code yourself)
# ============
BASE_DIR  = Path(r"F:\Master Resources\ACCT.745.01 - Acctg Info. & Analytics\Project 1\(1) Accounts Receivables Provision Data")
CLEAN_DIR = BASE_DIR / "_output_clean"
FACT_FILE = CLEAN_DIR / "_fact_cleaned_with_metrics.csv"

# Console display prefs
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 25)

# Style
plt.rcParams.update({
    "figure.figsize": (11, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================
# 2) LOAD CLEANED DATA
# =====================
def load_fact(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned file not found: {path}\nRun clean_and_summarize.py first.")
    df = pd.read_csv(path, low_memory=False)

    # Basic coercions / helpers
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    # Composition shares
    if "pct_over6m" not in df.columns and {"receivables_morethan6m","accounts_receivables"}.issubset(df.columns):
        df["pct_over6m"]  = df["receivables_morethan6m"]  / df["accounts_receivables"]
        df["pct_under6m"] = df["receivables_lessthan6m"] / df["accounts_receivables"]
    # Flags to numeric
    df["ar_comp_ok_num"] = pd.to_numeric(df.get("ar_comp_ok"), errors="coerce")
    df["ni_pos_cfo_neg_flag"] = pd.to_numeric(df.get("ni_pos_cfo_neg_flag"), errors="coerce").fillna(0).astype(int)

    # Replace infs
    for c in ["dso_approx","ar_turnover","prov_rate","writeoff_rate","pct_over6m","pct_under6m",
              "liq_current_ratio","lev_debt_to_assets"]:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df

df = load_fact(FACT_FILE)

def finite(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan)

def display_table(df_table: pd.DataFrame, title: str, rows: int = 15):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))
    print(df_table.head(rows).to_string(index=False))

# ==========================
# 3) Q&A — TABLES + FIGURES
# ==========================

# Q1 — Which industries/years show atypical AR aging?
q1 = (df.groupby(["industry_code","industry_name","year"], dropna=False)
        .agg(med_dso=("dso_approx","median"),
             med_pct_over6m=("pct_over6m","median"),
             n_obs=("company_code","size"),
             n_companies=("company_code","nunique"))
        .reset_index())
q1["rank_dso_in_year"] = q1.groupby("year")["med_dso"].rank(method="min", ascending=False)

# Show: latest year top/bottom 10 industries by median DSO
latest_year = int(q1["year"].max())
q1_latest = q1[q1["year"] == latest_year].sort_values("med_dso", ascending=False)
display_table(q1_latest[["industry_name","med_dso","med_pct_over6m","n_obs","n_companies","rank_dso_in_year"]],
              f"Q1 — {latest_year}: Industries ranked by median DSO (higher = older)")

# Figure: bar of top 15 oldest industries (latest year)
top15 = q1_latest.head(15)
plt.figure()
plt.barh(top15["industry_name"], top15["med_dso"])
plt.gca().invert_yaxis()
plt.title(f"Median DSO by Industry — {latest_year} (Top 15 oldest)")
plt.xlabel("Median DSO (days)")
plt.tight_layout()
plt.show()

# Q2 — Do slower collections correlate with higher provisioning/write-offs?
valid = df["dso_approx"].between(1, 400)
df_q2 = df.loc[valid, ["dso_approx","prov_rate","writeoff_rate"]].dropna()
df_q2["dso_decile"] = pd.qcut(df_q2["dso_approx"], 10, labels=[f"D{i}" for i in range(1,11)])

q2 = (df_q2.groupby("dso_decile")
          .agg(n=("dso_approx","size"),
               med_dso=("dso_approx","median"),
               med_prov_rate=("prov_rate","median"),
               med_writeoff_rate=("writeoff_rate","median"))
          .reset_index()
          .sort_values("med_dso"))
display_table(q2, "Q2 — DSO deciles vs median provision & write-off rates", rows=10)

# Figure: lines of provision & write-off vs DSO deciles
plt.figure()
plt.plot(q2["dso_decile"].astype(str), q2["med_prov_rate"], marker="o", label="Provision rate (median)")
plt.plot(q2["dso_decile"].astype(str), q2["med_writeoff_rate"], marker="o", label="Write-off rate (median)")
plt.title("Provision / Write-off vs DSO Deciles")
plt.xlabel("DSO decile (D1=fastest … D10=slowest)")
plt.ylabel("Rate")
plt.legend()
plt.tight_layout()
plt.show()

# Q3 — Is earnings quality weaker where collections lag?
df_q3 = df.loc[valid, ["dso_approx","ni_pos_cfo_neg_flag","industry_name"]].dropna()
df_q3["dso_decile"] = pd.qcut(df_q3["dso_approx"], 10, labels=[f"D{i}" for i in range(1,11)])

q3a = (df_q3.groupby("dso_decile")["ni_pos_cfo_neg_flag"]
            .mean().rename("share_nipos_cfoneg").reset_index())
display_table(q3a, "Q3a — Share of NI>0 & CFO<0 by DSO decile", rows=10)

# Figure: NI>0 & CFO<0 by DSO decile
plt.figure()
plt.plot(q3a["dso_decile"].astype(str), q3a["share_nipos_cfoneg"], marker="o")
plt.title("Share of NI>0 & CFO<0 by DSO decile")
plt.xlabel("DSO decile")
plt.ylabel("Share of observations")
plt.tight_layout()
plt.show()

q3b = (df.groupby("industry_name")["ni_pos_cfo_neg_flag"]
          .mean().rename("share_nipos_cfoneg").reset_index()
          .sort_values("share_nipos_cfoneg", ascending=False))
display_table(q3b, "Q3b — Industries ranked by NI>0 & CFO<0 share", rows=15)

# Figure: top 15 industries by NI>0 & CFO<0 share
plt.figure()
plt.barh(q3b.head(15)["industry_name"], q3b.head(15)["share_nipos_cfoneg"])
plt.gca().invert_yaxis()
plt.title("Top 15 industries — NI>0 & CFO<0 share")
plt.xlabel("Share")
plt.tight_layout()
plt.show()

# Q4 — Where are AR composition integrity issues?
q4a = (df.groupby(["industry_name","year"], dropna=False)["ar_comp_ok_num"]
         .mean().rename("ar_comp_ok_rate").reset_index())
# Bottom 15 industry-years by integrity
q4a_sorted = q4a.sort_values("ar_comp_ok_rate", ascending=True)
display_table(q4a_sorted, "Q4a — Lowest AR composition integrity (industry-year)", rows=15)

# Figure: bar of bottom 10
plt.figure()
tmp = q4a_sorted.head(10)
plt.barh(tmp["industry_name"] + " " + tmp["year"].astype(int).astype(str), tmp["ar_comp_ok_rate"])
plt.gca().invert_yaxis()
plt.title("Lowest AR composition integrity — bottom 10 industry-years")
plt.xlabel("Integrity pass rate")
plt.tight_layout()
plt.show()

# Company-level failure rate (top 20)
comp_fail = (df.assign(ar_fail = 1 - pd.to_numeric(df["ar_comp_ok_num"], errors="coerce"))
               .groupby("company_code")["ar_fail"].mean().rename("ar_fail_rate").reset_index()
               .sort_values("ar_fail_rate", ascending=False))
display_table(comp_fail, "Q4b — Companies with highest AR composition failure rate", rows=20)

# Q5 — What are the time trends in collection efficiency?
q5 = (df.groupby("year", dropna=False)
        .agg(med_dso=("dso_approx","median"),
             med_ar_turnover=("ar_turnover","median"),
             n_obs=("company_code","size"))
        .reset_index()
        .sort_values("year"))
display_table(q5, "Q5 — Year-over-year trends: median DSO & AR turnover", rows=20)

# Figures: DSO and AR turnover trend lines
plt.figure()
plt.plot(q5["year"], q5["med_dso"], marker="o")
plt.title("Median DSO over time")
plt.xlabel("Year"); plt.ylabel("Days")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(q5["year"], q5["med_ar_turnover"], marker="o")
plt.title("Median AR Turnover over time")
plt.xlabel("Year"); plt.ylabel("Times")
plt.tight_layout()
plt.show()

# Q6 — How do liquidity and leverage relate to DSO?
df_f = df[["dso_approx","liq_current_ratio","lev_debt_to_assets"]].replace([np.inf, -np.inf], np.nan).dropna()

def quartile_bins(s, name):
    return pd.qcut(s, 4, labels=[f"{name} Q1 (low)",f"{name} Q2",f"{name} Q3",f"{name} Q4 (high)"])

df_f["liq_bin"] = quartile_bins(df_f["liq_current_ratio"], "CR")
df_f["lev_bin"] = quartile_bins(df_f["lev_debt_to_assets"], "Lev")

q6a = (df_f.groupby("liq_bin")
          .agg(median_dso=("dso_approx","median"), n=("dso_approx","size"))
          .reset_index())
display_table(q6a, "Q6a — Median DSO by Current Ratio quartile", rows=10)

plt.figure()
plt.bar(q6a["liq_bin"].astype(str), q6a["median_dso"])
plt.title("Median DSO by Current Ratio quartile")
plt.xlabel("Current Ratio quartile"); plt.ylabel("Median DSO")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

q6b = (df_f.groupby("lev_bin")
          .agg(median_dso=("dso_approx","median"), n=("dso_approx","size"))
          .reset_index())
display_table(q6b, "Q6b — Median DSO by Leverage (Debt/Assets) quartile", rows=10)

plt.figure()
plt.bar(q6b["lev_bin"].astype(str), q6b["median_dso"])
plt.title("Median DSO by Leverage quartile")
plt.xlabel("Leverage quartile"); plt.ylabel("Median DSO")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# Q7 — Which industries under-report provisioning/write-offs?
q7 = (df.groupby(["industry_name","year"], dropna=False)
        .agg(prov_cov=("provision_bad_receivables", lambda s: s.notna().mean()),
             writeoff_cov=("writeoff_bad_receivables", lambda s: s.notna().mean()),
             n_obs=("company_code","size"))
        .reset_index())
# Worst average coverage across years
q7b = (q7.groupby("industry_name")
         .agg(avg_prov_cov=("prov_cov","mean"),
              avg_writeoff_cov=("writeoff_cov","mean"),
              years=("year","nunique"))
         .reset_index()
         .sort_values("avg_prov_cov"))
display_table(q7b, "Q7 — Industries with lowest average provision coverage", rows=15)

plt.figure()
plt.barh(q7b.head(12)["industry_name"], q7b.head(12)["avg_prov_cov"])
plt.gca().invert_yaxis()
plt.title("Lowest average provision coverage — top 12 industries")
plt.xlabel("Average coverage (share)")
plt.tight_layout()
plt.show()

# Q8 — Who are standout fast/slow collectors in latest year?
df_latest = df[df["year"] == latest_year].copy()
mask = (df_latest["dso_approx"] > 0) & (df_latest["dso_approx"] < 400)

fast = (df_latest.loc[mask]
        .groupby(["industry_name","company_code"])["dso_approx"]
        .median().reset_index().sort_values("dso_approx").head(20))
slow = (df_latest.loc[mask]
        .groupby(["industry_name","company_code"])["dso_approx"]
        .median().reset_index().sort_values("dso_approx", ascending=False).head(20))

display_table(fast, f"Q8 — {latest_year}: Fastest collectors (lowest median DSO)", rows=20)
display_table(slow, f"Q8 — {latest_year}: Slowest collectors (highest median DSO)", rows=20)

# Figure: DSO histogram (latest year)
plt.figure()
dso_latest = df_latest.loc[mask, "dso_approx"].dropna()
plt.hist(dso_latest, bins=50)
plt.title(f"DSO distribution — {latest_year} (trimmed 0–400)")
plt.xlabel("DSO (days)"); plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\nDone. Tables printed above, and figures displayed.")
