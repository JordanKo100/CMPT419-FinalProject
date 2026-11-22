"""
build_datasets.py — Clean, leak-free data pipeline for BoC & Fed rate prediction.

Place this script in your project root (fproj/) and run:
    python build_datasets.py

What this script produces inside fproj/data/processed/:
    1. factor_<macro>.csv               -> 6 separate CSVs for independent macro models
    2. factor_communications_fed.csv    -> Fed FOMC statements only (input for RoBERTa)
    3. factor_communications_boc.csv    -> BoC CBS speeches (input for RoBERTa)
    4. composite_dataset.csv            -> All 36 macro trend features + comms_count
                                           (left-join aggregated RoBERTa scores here later)
"""

import csv
import re
from pathlib import Path
import pandas as pd
import requests

# =============================================================================
# 1. CONFIGURATION
#    Adjust dates, rolling windows, or FRED series IDs here without touching
#    the pipeline logic below.
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent        # Resolves to fproj/
RAW_DIR = ROOT_DIR / "data" / "raw"               # Cached/manual raw inputs
PROCESSED_DIR = ROOT_DIR / "data" / "processed"   # Clean model-ready CSVs

# Date window where all datasets overlap cleanly:
# - Hugging Face FOMC statements begin in 2000
# - CBS speeches dataset runs through Dec 2023 (set END_DATE = None to extend)
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"

# Threshold rule for "maintain_high" vs "maintain_low":
# Compare the held rate against the bank's rolling 60-month (~5-year) past median.
ROLLING_MEDIAN_WINDOW = 60

# Small tolerance so floating-point noise (e.g., 0.0000001) isn't misread as a rate hike/cut.
RATE_CHANGE_EPS = 1e-4

# External dataset identifiers
KAGGLE_RATES_SLUG = "kanchana1990/central-bank-rates-intelligence-19452026"
HF_FOMC_DATASET = "vtasca/fomc-statements-minutes"

# Symmetrical FRED / BIS series IDs for Bank of Canada (BoC) and Federal Reserve (Fed).
# Each key becomes one independent factor CSV (factor_<key>.csv) and 6 columns in composite.
FRED_SERIES = {
    "inflation":    {"BoC": "CPALTT01CAM659N", "Fed": "CPALTT01USM659N"},  # Monthly CPI YoY %
    "unemployment": {"BoC": "LRUNTTTTCAM156S", "Fed": "UNRATE"},           # Monthly Unemployment %
    "gdp":          {"BoC": "NAEXKP01CAQ657S", "Fed": "NAEXKP01USQ657S"},  # Quarterly Real GDP QoQ %
    "housing":      {"BoC": "QCAR628BIS",      "Fed": "QUSR628BIS"},       # Quarterly BIS Property Prices
    "stock_market": {"BoC": "SPASTT01CAM661N", "Fed": "SPASTT01USM661N"},  # Monthly Share Prices Index
    "bond_yield":   {"BoC": "IRLTLT01CAM156N", "Fed": "IRLTLT01USM156N"},  # Monthly 10Y Gov Bond Yield %
}

# Quarterly series use a +3-month lag (previous quarter); all others use +1-month (previous month).
QUARTERLY_FACTORS = {"gdp", "housing"}

# Index-level series measure price indices (e.g., 2015=100), so % change is more meaningful
# than raw point differences. Rate/growth series (CPI %, UNRATE %, GDP %, yields) use raw diffs.
INDEX_FACTORS = {"housing", "stock_market"}

# Ordinal integer encoding for Quadratic Weighted Kappa (QWK):
# Ordered from most dovish (0 = cut) to most hawkish (3 = hike) so QWK penalizes
# predicting a hike during a cut much more heavily than predicting maintain_high.
TARGET_MAP = {
    "decrease": 0,
    "maintain_low": 1,
    "maintain_high": 2,
    "increase": 3,
}


# =============================================================================
# 2. HELPERS & DATA LOADERS
#    Each loader returns a standardized DataFrame so downstream alignment code
#    stays identical regardless of where the raw data came from.
# =============================================================================
def clean_text(text: object) -> str:
    """
    Sanitize speech/statement text for CSV storage and RoBERTa tokenization.
    Replaces all line breaks (\r, \n), tabs (\t), and repeated spaces with a single space
    so CSV rows never split across lines.
    """
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _pick_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """
    Case-insensitive column finder used for external datasets (like Hugging Face)
    whose capitalization or spacing may vary. Raises a clear error listing actual
    columns if nothing matches, making schema debugging immediate.
    """
    lookup = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise KeyError(f"Could not find '{label}' column. Available: {list(df.columns)}")


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """
    Fetch a single FRED/BIS time series directly from FRED's public CSV endpoint
    (no API key required) and cache it in data/raw/fred/<series_id>.csv.
    Delete the cached CSV if you ever want to pull fresh numbers from FRED.
    """
    cache_path = RAW_DIR / "fred" / f"{series_id}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)

    df = pd.read_csv(cache_path)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # FRED marks missing observations as "." — errors="coerce" converts "." cleanly to NaN
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def load_policy_rates() -> pd.DataFrame:
    """
    Load monthly policy rates for BoC and Fed from Kaggle's Central Bank Rates Intelligence.
    Checks data/raw/kaggle/ first (manual download); otherwise downloads via kagglehub.
    Only keeps date, central_bank, and policy_rate_pct to avoid leaking Kaggle's
    precomputed cycle/regime analytics columns.
    """
    local_dir = RAW_DIR / "kaggle"
    csv_files = list(local_dir.glob("*.csv")) if local_dir.exists() else []

    if not csv_files:
        import kagglehub
        downloaded = Path(kagglehub.dataset_download(KAGGLE_RATES_SLUG))
        csv_files = list(downloaded.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV found for Kaggle dataset {KAGGLE_RATES_SLUG}.")

    # If the Kaggle folder has multiple CSVs, pick the largest (the main rates table)
    raw = pd.read_csv(max(csv_files, key=lambda p: p.stat().st_size))
    df = raw[["date", "central_bank", "policy_rate_pct"]].rename(
        columns={"policy_rate_pct": "policy_rate"}
    )

    # Keep only Bank of Canada and Federal Reserve, mapped to short flags ("BoC", "Fed")
    df["central_bank"] = df["central_bank"].map({
        "Bank of Canada": "BoC",
        "Federal Reserve": "Fed",
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["policy_rate"] = pd.to_numeric(df["policy_rate"], errors="coerce")

    return (
        df.dropna(subset=["date", "central_bank", "policy_rate"])
        .drop_duplicates(["central_bank", "date"])
        .sort_values(["central_bank", "date"])
        .reset_index(drop=True)
    )


def load_fed_statements() -> pd.DataFrame:
    """
    Load Fed FOMC statements from Hugging Face (auto-downloaded & cached in ~/.cache/huggingface).
    Filters strictly to Type == 'statement' (dropping FOMC minutes) and uses Release Date
    so timestamps match when the public/markets actually saw the statement.
    """
    from datasets import load_dataset
    fomc = load_dataset(HF_FOMC_DATASET, split="train").to_pandas()
    rel_col = _pick_col(fomc, ["release date", "release_date", "date"], "FOMC release date")
    type_col = _pick_col(fomc, ["type", "doc_type"], "FOMC type")
    txt_col = _pick_col(fomc, ["text", "content"], "FOMC text")

    # Keep only formal FOMC policy statements (exclude minutes)
    sub = fomc[fomc[type_col].astype(str).str.lower().str.strip() == "statement"]
    df = pd.DataFrame({
        "doc_date": pd.to_datetime(sub[rel_col], errors="coerce"),
        "central_bank": "Fed",
        "doc_type": "fomc_statement",
        "text": sub[txt_col].map(clean_text),
    }).dropna(subset=["doc_date"])
    return df[df["text"].str.len() > 0].sort_values("doc_date").reset_index(drop=True)


def load_boc_speeches() -> pd.DataFrame:
    """
    Load Bank of Canada speeches from the manually downloaded CBS CSV (data/raw/cbs/).
    Reads only the 3 needed columns (Date, CentralBank, text) for speed and outputs the
    exact same 4-column schema as load_fed_statements().
    Note: CBS 'text' is already English-translated ('text_original' holds non-English text).
    """
    cbs_files = list((RAW_DIR / "cbs").glob("*.csv"))
    if not cbs_files:
        raise FileNotFoundError("Place CBS_dataset_v1.0.csv inside fproj/data/raw/cbs/.")

    cbs = pd.read_csv(cbs_files[0], usecols=["Date", "CentralBank", "text"], low_memory=False)
    sub = cbs[cbs["CentralBank"] == "Bank of Canada"]
    df = pd.DataFrame({
        "doc_date": pd.to_datetime(sub["Date"], errors="coerce"),
        "central_bank": "BoC",
        "doc_type": "boc_speech",
        "text": sub["text"].map(clean_text),
    }).dropna(subset=["doc_date"])
    return df[df["text"].str.len() > 0].sort_values("doc_date").reset_index(drop=True)


# =============================================================================
# 3. TARGET LABELING, TREND FEATURES & LEAK-FREE ALIGNMENT
# =============================================================================
def build_targets(rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 4-class target label for each (central_bank, date) decision row:
      - increase      (rate_t > rate_{t-1})
      - decrease      (rate_t < rate_{t-1})
      - maintain_high (rate_t == rate_{t-1} and rate_t >= past 60-month median)
      - maintain_low  (rate_t == rate_{t-1} and rate_t <  past 60-month median)

    Crucial anti-leakage detail: rolling_median uses .shift(1) BEFORE .rolling() so
    the threshold only looks at past rates strictly before month t.
    """
    parts = []
    for _, grp in rates_df.groupby("central_bank"):
        g = grp.sort_values("date").copy()
        g["prev_date"] = g["date"].shift(1)               # Previous decision date (t - 1)
        g["prev_rate"] = g["policy_rate"].shift(1)        # Previous policy rate
        g["rate_diff"] = g["policy_rate"] - g["prev_rate"]

        # Rolling historical median computed strictly over past observations [t-60, t-1]
        g["rolling_median"] = (
            g["policy_rate"]
            .shift(1)
            .rolling(window=ROLLING_MEDIAN_WINDOW, min_periods=12)
            .median()
        )

        def classify(row):
            if pd.isna(row["rate_diff"]) or pd.isna(row["rolling_median"]):
                return None
            if row["rate_diff"] > RATE_CHANGE_EPS:
                return "increase"
            if row["rate_diff"] < -RATE_CHANGE_EPS:
                return "decrease"
            return "maintain_high" if row["policy_rate"] >= row["rolling_median"] else "maintain_low"

        g["target"] = g.apply(classify, axis=1)
        g["target_id"] = g["target"].map(TARGET_MAP)
        parts.append(g)

    # Filter to [START_DATE, END_DATE] AFTER computing rolling medians so early 2000 rows
    # still have a full 5-year historical median computed from 1995-1999 rates.
    out = pd.concat(parts, ignore_index=True).dropna(subset=["target"])
    out = out[out["date"] >= START_DATE]
    if END_DATE:
        out = out[out["date"] <= END_DATE]
    return out.sort_values(["date", "central_bank"]).reset_index(drop=True)


def add_trend_features(series_df: pd.DataFrame, factor: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute 6 central-bank trend features on a raw FRED series BEFORE merging:
      1. Level          ({f})           -> latest available reading
      2. 1-period chg   (_chg_1m/1q)    -> immediate momentum
      3. Short-term chg (_chg_3m/2q)    -> quarterly / half-year acceleration
      4. 1-year chg     (_chg_12m/4q)   -> year-over-year direction
      5. Rolling mean   (_roll6m/4q)    -> smoothed underlying trend (filters noise)
      6. Trend gap      (_vs_roll6m/4q) -> how far the latest reading sits above/below trend
    """
    s = series_df.copy()
    is_quarterly = factor in QUARTERLY_FACTORS
    is_index = factor in INDEX_FACTORS

    # Pick matching horizons: monthly (1m, 3m, 12m, 6m mean) vs quarterly (1q, 2q, 4q, 4q mean)
    p_short, p_year, p_roll = (2, 4, 4) if is_quarterly else (3, 12, 6)
    unit = "q" if is_quarterly else "m"

    def change(col: pd.Series, periods: int) -> pd.Series:
        # Index series (stocks, housing) use % return; rate/growth series use arithmetic diff
        return col.pct_change(periods) * 100.0 if is_index else col.diff(periods)

    c1 = f"{factor}_chg_1{unit}"
    c_short = f"{factor}_chg_{p_short}{unit}"
    c_year = f"{factor}_chg_{p_year}{unit}"
    c_roll = f"{factor}_roll{p_roll}{unit}_mean"
    c_gap = f"{factor}_vs_roll{p_roll}{unit}"

    s[c1] = change(s[factor], 1)
    s[c_short] = change(s[factor], p_short)
    s[c_year] = change(s[factor], p_year)
    s[c_roll] = s[factor].rolling(window=p_roll, min_periods=2).mean()
    s[c_gap] = s[factor] - s[c_roll]

    return s, [factor, c1, c_short, c_year, c_roll, c_gap]


def attach_macro_factor(base_df: pd.DataFrame, factor: str, series_map: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Align one macro factor to every decision date with uniform, leak-free calendar lags:
      - Monthly factors (+1 month): e.g., Feb obs (2023-02-01) becomes available on 2023-03-01.
      - Quarterly factors (+3 months): e.g., Q1 obs (2023-01-01) becomes available on 2023-04-01.

    merge_asof(..., direction="backward") picks the most recent available_date <= decision date.
    """
    lag_months = 3 if factor in QUARTERLY_FACTORS else 1
    parts, feat_cols = [], []

    for bank, grp in base_df.groupby("central_bank"):
        s = fetch_fred_series(series_map[bank]).rename(columns={"date": "obs_date", "value": factor})
        s, feat_cols = add_trend_features(s, factor)

        # Shift observation date forward by exact calendar lag to form available_date
        s["available_date"] = s["obs_date"].apply(lambda d: d + pd.DateOffset(months=lag_months))

        merged = pd.merge_asof(
            grp.sort_values("date"),
            s[["obs_date", "available_date"] + feat_cols].sort_values("available_date"),
            left_on="date",
            right_on="available_date",
            direction="backward",
        )

        # Hard runtime guardrail: fail immediately if any macro obs_date is not strictly earlier
        valid = merged.dropna(subset=["obs_date"])
        assert (valid["obs_date"] < valid["date"]).all(), f"Look-ahead bias detected in {factor} ({bank})"
        parts.append(merged.drop(columns=["obs_date", "available_date"]))

    # Sort identically to base_df so columns align 1-to-1 when copied into composite_dataset
    aligned = pd.concat(parts, ignore_index=True).sort_values(["date", "central_bank"]).reset_index(drop=True)
    return aligned, feat_cols


def align_communications(base_df: pd.DataFrame, docs_df: pd.DataFrame, bank: str) -> pd.DataFrame:
    """
    Map each speech/statement to the NEXT upcoming rate decision so NLP predicts future moves:
      - Window: prev_date <= doc_date < decision_date
      - Example: a statement released on 2023-02-01 or 2023-02-15 is mapped to predict 2023-03-01.
      - allow_exact_matches=False ensures a doc released ON 2023-03-01 predicts 2023-04-01, not 2023-03-01.
    """
    bank_dec = base_df[base_df["central_bank"] == bank].sort_values("date").rename(columns={"date": "decision_date"})
    merged = pd.merge_asof(
        docs_df.sort_values("doc_date"),
        bank_dec[["decision_date", "prev_date", "target", "target_id"]],
        left_on="doc_date",
        right_on="decision_date",
        direction="forward",
        allow_exact_matches=False,
    )

    # Drop documents outside [START_DATE, END_DATE] or before the first prev_date
    merged = merged.dropna(subset=["decision_date"])
    merged = merged[merged["doc_date"] >= merged["prev_date"]]

    # Hard runtime guardrail: every document must strictly precede the decision it predicts
    assert (merged["doc_date"] < merged["decision_date"]).all(), f"Look-ahead bias in {bank} comms"

    cols = ["decision_date", "central_bank", "doc_date", "doc_type", "target", "target_id", "text"]
    return merged[cols].sort_values(["decision_date", "doc_date"]).reset_index(drop=True)


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    core_cols = ["date", "central_bank", "policy_rate", "prev_rate", "rolling_median", "target", "target_id"]

    # Step 1: Build master target table (576 monthly rows: 288 BoC + 288 Fed)
    print("1/4 Loading policy rates & building 4-class targets...")
    base = build_targets(load_policy_rates())
    print(f"    Base rows: {len(base)} | Banks: {base['central_bank'].value_counts().to_dict()}")
    print(f"    Class balance: {base['target'].value_counts().to_dict()}")

    # Step 2: Pull each FRED/BIS indicator, compute 6 trend features, save independent CSV,
    #         and copy those 6 columns into the composite table
    print("2/4 Fetching & aligning FRED/BIS macro factors (uniform 1m/1q lags + trends)...")
    composite = base.copy()
    all_macro_cols = []
    for factor, series_map in FRED_SERIES.items():
        factor_df, feat_cols = attach_macro_factor(base, factor, series_map)
        out_path = PROCESSED_DIR / f"factor_{factor}.csv"
        factor_df[core_cols + feat_cols].dropna().to_csv(out_path, index=False)

        composite[feat_cols] = factor_df[feat_cols].values
        all_macro_cols.extend(feat_cols)

        lag_label = "prev quarter" if factor in QUARTERLY_FACTORS else "prev month"
        print(f"    Saved {out_path.name} ({len(factor_df.dropna(subset=feat_cols))} rows, {len(feat_cols)} features, lag={lag_label})")

    # Step 3: Align Fed statements & BoC speeches to the NEXT decision date.
    #         Written with QUOTE_ALL so commas/quotes inside text never shift CSV columns.
    print("3/4 Aligning Fed statements & BoC speeches (identical 7-column schema)...")
    fed_aligned = align_communications(base, load_fed_statements(), "Fed")
    boc_aligned = align_communications(base, load_boc_speeches(), "BoC")

    fed_path = PROCESSED_DIR / "factor_communications_fed.csv"
    boc_path = PROCESSED_DIR / "factor_communications_boc.csv"
    fed_aligned.to_csv(fed_path, index=False, quoting=csv.QUOTE_ALL)
    boc_aligned.to_csv(boc_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"    Saved {fed_path.name} ({len(fed_aligned)} statements)")
    print(f"    Saved {boc_path.name} ({len(boc_aligned)} speeches)")

    # Step 4: Save the numeric composite base (36 macro features + comms_count).
    #         After RoBERTa scores the two comms CSVs, aggregate by (decision_date, central_bank)
    #         and left-join onto composite_dataset.csv on (date, central_bank).
    print("4/4 Saving composite macro base (ready to merge RoBERTa scores)...")
    all_comms = pd.concat([fed_aligned, boc_aligned], ignore_index=True)
    counts = all_comms.groupby(["decision_date", "central_bank"]).size().rename("comms_count").reset_index()
    composite = composite.merge(
        counts, left_on=["date", "central_bank"], right_on=["decision_date", "central_bank"], how="left"
    ).drop(columns=["decision_date"])
    composite["comms_count"] = composite["comms_count"].fillna(0).astype(int)

    comp_path = PROCESSED_DIR / "composite_dataset.csv"
    composite_clean = composite[core_cols + all_macro_cols + ["comms_count"]].dropna(subset=all_macro_cols)
    composite_clean.to_csv(comp_path, index=False)
    print(f"    Saved {comp_path.name} ({len(composite_clean)} rows, {len(all_macro_cols)} macro features)")


if __name__ == "__main__":
    main()