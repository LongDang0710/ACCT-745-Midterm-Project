# inspect_and_dictionary.py
# ACCT 745 — Data inspection & data dictionary creation
# Author: Long Dang
# Purpose: Load the 3 CSVs, sanity-check, and write data dictionaries to CSV.

from pathlib import Path
import pandas as pd
import numpy as np
import re

# ============
# 1) SETTINGS
# ============
# Use a raw string (r"...") for Windows paths so backslashes don't need escaping.
# Remember to change the datasets path if you plan to use the code yourself
BASE_DIR = Path(r"F:\Master Resources\ACCT.745.01 - Acctg Info. & Analytics\Project 1\(1) Accounts Receivables Provision Data")

FILES = {
    "provisions": "provisions for acct receivables.csv",
    "industry_data": "industry_data.csv",
    "industry_names": "industry names.csv",
}

# Output folder for dictionaries & quick snapshots
OUT_DIR = BASE_DIR / "_output_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================
# 2) LOAD HELPERS
# ================
def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader with a couple of common encodings.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    # Last attempt with default
    return pd.read_csv(path, low_memory=False)

# ==========================
# 3) DATA DICTIONARY BUILDER
# ==========================
def build_data_dictionary(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Create a compact data dictionary:
      - dtype
      - non-null count, null count, % missing
      - # unique (for small-cardinality hints)
      - example values (first 3 non-null)
    """
    d = pd.DataFrame({
        "table": table_name,
        "column": df.columns,
        "dtype_inferred": [str(t) for t in df.dtypes],
        "n_non_null": df.notna().sum().values,
        "n_null": df.isna().sum().values,
        "pct_missing": (df.isna().mean().values * 100).round(2),
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
    examples = []
    for c in df.columns:
        vals = df[c].dropna().head(3).tolist()
        # Convert to strings for safety in CSV
        examples.append(", ".join(map(lambda x: str(x)[:60], vals)))
    d["examples"] = examples
    return d

# =====================
# 4) RUN THE INSPECTION
# =====================
def parse_stata_like_date(val: str):
    """
    Parse strings like '31dec2019' → Timestamp('2019-12-31').
    Returns pd.NaT if the pattern doesn't match.
    """
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

def main():
    # ---- Load
    provisions = safe_read_csv(BASE_DIR / FILES["provisions"])
    industry_data = safe_read_csv(BASE_DIR / FILES["industry_data"])
    industry_names = safe_read_csv(BASE_DIR / FILES["industry_names"])

    # ---- Quick shapes
    print("Shapes:")
    print("  provisions:", provisions.shape)
    print("  industry_data:", industry_data.shape)
    print("  industry_names:", industry_names.shape)

    # ---- Provisions: add parsed date fields if present
    if "date_var" in provisions.columns:
        provisions["date"] = provisions["date_var"].map(parse_stata_like_date)
        provisions["year"] = provisions["date"].dt.year
        provisions["quarter"] = provisions["date"].dt.quarter

    # ---- Optional: check duplicate (company_code, date) pairs (just report here)
    if {"company_code", "date"}.issubset(provisions.columns):
        dupes = provisions.duplicated(subset=["company_code", "date"], keep=False)
        n_dupe_rows = int(dupes.sum())
        n_dupe_pairs = int(provisions.loc[dupes, ["company_code", "date"]].drop_duplicates().shape[0])
        print(f"Duplicate rows at (company_code, date): {n_dupe_rows} rows across {n_dupe_pairs} pairs.")

    # ---- Build data dictionaries
    dict_prov = build_data_dictionary(provisions, "provisions")
    dict_ind_data = build_data_dictionary(industry_data, "industry_data")
    dict_ind_names = build_data_dictionary(industry_names, "industry_names")

    # ---- Save dictionaries
    out_prov = OUT_DIR / "_data_dictionary_provisions.csv"
    out_indd = OUT_DIR / "_data_dictionary_industry_data.csv"
    out_indn = OUT_DIR / "_data_dictionary_industry_names.csv"

    dict_prov.to_csv(out_prov, index=False)
    dict_ind_data.to_csv(out_indd, index=False)
    dict_ind_names.to_csv(out_indn, index=False)

    print("\nData dictionaries written to:")
    print(f"  {out_prov}")
    print(f"  {out_indd}")
    print(f"  {out_indn}")

    # ---- Light sanity snapshots
    # Head/tail samples saved so you can eyeball structure without opening full files
    provisions.head(50).to_csv(OUT_DIR / "_sample_head_provisions.csv", index=False)
    industry_data.head(50).to_csv(OUT_DIR / "_sample_head_industry_data.csv", index=False)
    industry_names.head(50).to_csv(OUT_DIR / "_sample_head_industry_names.csv", index=False)

    # ---- Simple EDA snippets printed to console
    print("\nProvisions — date distribution (month %), if date parsed:")
    if "date" in provisions.columns:
        month_share = provisions["date"].dt.month.value_counts(normalize=True).sort_index() * 100
        print(month_share.round(2).to_string())

    print("\nProvisions — AR-related columns present:")
    ar_cols = [c for c in provisions.columns if "receivable" in c.lower() or c.lower().startswith(("ar_", "accounts_receivables"))]
    print(ar_cols)

    # Save a one-pager of core numeric columns with basic stats
    numeric_cols = provisions.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        prov_describe = provisions[numeric_cols].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).T
        prov_describe.to_csv(OUT_DIR / "_provisions_numeric_summary.csv")
        print("\nSaved numeric summary for provisions to _provisions_numeric_summary.csv")

if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 40)
    main()
