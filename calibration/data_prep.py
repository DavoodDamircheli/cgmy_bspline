"""
OptionMetrics SPX snapshot loader for Section 7 calibration.

load_raw_chain reads one CSV file, constructs derived columns, and applies
the liquidity / data-quality filters from Section 2 of the calibration
roadmap.  Moneyness / maturity selection (Step 5) is NOT done here.
"""

import numpy as np
import pandas as pd


# Columns returned by load_raw_chain (in this order)
_OUTPUT_COLS = [
    "date", "exdate", "T_years", "cp_flag",
    "K", "mid", "best_bid", "best_offer",
    "spread", "rel_spread",
    "open_interest", "impl_volatility", "optionid",
]

# Expected issuer string; we print unique values before filtering so a
# mismatch is immediately visible rather than silently zeroing all rows.
_EXPECTED_ISSUER = "CBOE S&P 500 INDEX"


def load_raw_chain(csv_path: str) -> pd.DataFrame:
    """Read one OptionMetrics SPX snapshot and apply Section 2 filters.

    Parameters
    ----------
    csv_path : str
        Path to an OptionMetrics CSV with columns:
        date, exdate, cp_flag, strike_price, best_bid, best_offer,
        open_interest, impl_volatility, optionid, index_flag, issuer,
        exercise_style

    Returns
    -------
    pd.DataFrame with columns defined in _OUTPUT_COLS.
    Row count after each filter step is printed to stdout.
    """
    # ── 1. Read CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, parse_dates=["date", "exdate"])

    # ── 2-6. Derived columns ─────────────────────────────────────────────────
    df["K"]       = df["strike_price"] / 1000.0
    df["mid"]     = (df["best_bid"] + df["best_offer"]) / 2.0
    df["spread"]  = df["best_offer"] - df["best_bid"]

    # Guard against mid <= 0 to avoid divide-by-zero; those rows are filtered
    # out at step (f) anyway, but inf makes the intent explicit.
    df["rel_spread"] = np.where(
        df["mid"] > 0,
        df["spread"] / df["mid"],
        np.inf,
    )

    df["T_years"] = (df["exdate"] - df["date"]).dt.days / 365.0

    # ── Issuer diagnostic ─────────────────────────────────────────────────────
    unique_issuers = df["issuer"].unique()
    print(f"  unique issuer values before filter: {unique_issuers.tolist()}")

    issuer_filter = _EXPECTED_ISSUER
    if _EXPECTED_ISSUER not in unique_issuers:
        # Fall back to the closest match by character overlap so at least one
        # filter value is chosen, and warn loudly.
        from difflib import get_close_matches
        matches = get_close_matches(_EXPECTED_ISSUER, unique_issuers, n=1, cutoff=0.0)
        issuer_filter = matches[0] if matches else unique_issuers[0]
        print(
            f"  WARNING: expected issuer '{_EXPECTED_ISSUER}' not found; "
            f"using closest match '{issuer_filter}'"
        )
    else:
        print(f"  using issuer filter: '{issuer_filter}'  (exact match)")

    # ── 7. Filters (a)-(h) ───────────────────────────────────────────────────
    n_raw = len(df)
    print(f"  raw rows: {n_raw:,}")

    # (a) European exercise
    df = df[df["exercise_style"] == "E"]
    print(f"  after (a) exercise_style=='E':             {len(df):,}")

    # (b) Index option flag
    df = df[df["index_flag"] == 1]
    print(f"  after (b) index_flag==1:                   {len(df):,}")

    # (c) CBOE SPX issuer
    df = df[df["issuer"] == issuer_filter]
    print(f"  after (c) issuer=='{issuer_filter}':  {len(df):,}")

    # (d) Non-negative bid
    df = df[df["best_bid"] >= 0]
    print(f"  after (d) best_bid>=0:                     {len(df):,}")

    # (e) Positive spread (offer strictly above bid)
    df = df[df["best_offer"] > df["best_bid"]]
    print(f"  after (e) best_offer>best_bid:             {len(df):,}")

    # (f) Positive mid-price
    df = df[df["mid"] > 0]
    print(f"  after (f) mid>0:                           {len(df):,}")

    # (g) Implied volatility must be available
    df = df[df["impl_volatility"].notna()]
    print(f"  after (g) impl_volatility.notna():         {len(df):,}")

    # (h) Open interest threshold (liquid strikes only)
    df = df[df["open_interest"] > 100]
    print(f"  after (h) open_interest>100:               {len(df):,}")

    # ── 8. Select and return output columns ──────────────────────────────────
    return df[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Expiration inspection and target-maturity selection
# ---------------------------------------------------------------------------

def list_expirations(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-exdate summary table for a cleaned chain.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_raw_chain — must contain columns
        date, exdate, cp_flag.

    Returns
    -------
    pd.DataFrame with columns:
        exdate, days_to_expiry, n_calls, n_puts, n_total
    sorted ascending by days_to_expiry.
    """
    # All rows share the same trade date; pull it from the first row.
    trade_date = df["date"].iloc[0]

    # Compute days_to_expiry per row then aggregate — avoids groupby.apply
    # and the pandas 2.x FutureWarning about operating on grouping columns.
    df2 = df.assign(
        days_to_expiry=((df["exdate"] - trade_date).dt.days).astype(int),
        is_call=(df["cp_flag"] == "C").astype(int),
        is_put=(df["cp_flag"] == "P").astype(int),
    )

    agg = (
        df2.groupby("exdate", sort=True)
        .agg(
            days_to_expiry=("days_to_expiry", "first"),
            n_calls=("is_call", "sum"),
            n_puts=("is_put", "sum"),
            n_total=("is_call", "count"),
        )
        .reset_index()
    )

    return agg.sort_values("days_to_expiry").reset_index(drop=True)


# Roadmap reference values for the 5-slot comparison table.
# Used only for the ">5 day" warning; the code never hardcodes these
# as the actual selection targets.
_ROADMAP_ACTUAL_DAYS = {
    # date_str -> {target_days: expected_actual_days}
    "2020-01-15": {7: 7,   30: 30,  93: 93,  182: 167, 366: 366},
    "2020-03-18": {7: 7,   30: 30,  93: 93,  182: 184, 366: 366},
    "2021-09-15": {7: 7,   30: 30,  93: 93,  182: 184, 366: 366},
}


def select_target_maturities(
    df: pd.DataFrame,
    target_days: tuple = (7, 30, 93, 182, 366),
) -> dict:
    """For each target horizon, pick the single closest available exdate.

    Parameters
    ----------
    df          : cleaned chain (output of load_raw_chain)
    target_days : tuple of integer day targets; default (7, 30, 93, 182, 366)

    Returns
    -------
    dict mapping each target_days value -> sub-DataFrame for the chosen exdate.

    Also prints a comparison table and any collision / roadmap-deviation warnings.
    """
    expirations = list_expirations(df)
    trade_date  = df["date"].iloc[0]
    date_str    = str(trade_date.date())

    roadmap_ref = _ROADMAP_ACTUAL_DAYS.get(date_str, {})

    result: dict = {}
    chosen: dict = {}   # target_days -> actual_days (for collision detection)

    header = f"{'target_d':>10}  {'chosen_exdate':>14}  {'actual_d':>10}  {'n_calls':>8}  {'n_puts':>7}"
    print(header)
    print("-" * len(header))

    for tgt in target_days:
        # Closest exdate by |actual_days - target|; first match wins on ties.
        idx     = (expirations["days_to_expiry"] - tgt).abs().idxmin()
        row     = expirations.loc[idx]
        actual  = int(row["days_to_expiry"])
        exdate  = row["exdate"]

        sub_df  = df[df["exdate"] == exdate].copy()
        result[tgt] = sub_df

        chosen[tgt] = actual

        # Roadmap deviation warning
        ref = roadmap_ref.get(tgt)
        deviation_warn = ""
        if ref is not None and abs(actual - ref) > 5:
            deviation_warn = f"  <-- WARNING: roadmap expected {ref}d"

        print(
            f"{tgt:>10}  {str(exdate.date()):>14}  {actual:>10}  "
            f"{int(row['n_calls']):>8}  {int(row['n_puts']):>7}"
            f"{deviation_warn}"
        )

    print()

    # Collision check: two targets landing on the same actual_days
    seen_days: dict = {}   # actual_days -> first target that claimed it
    for tgt, actual in chosen.items():
        if actual in seen_days:
            print(
                f"  WARNING: target {tgt}d and target {seen_days[actual]}d "
                f"both mapped to the same exdate ({actual}d)"
            )
        else:
            seen_days[actual] = tgt

    return result


# ---------------------------------------------------------------------------
# Self-test: run on all three data files
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pathlib

    data_dir = pathlib.Path(__file__).parents[1] / "data"
    files = [
        data_dir / "2020-01-15.csv",
        data_dir / "2020-03-18.csv",
        data_dir / "2021-09-15.csv",
    ]

    TARGET_DAYS = (7, 30, 93, 182, 366)
    all_passed  = True

    for f in files:
        print()
        print("=" * 65)
        print(f"File: {f.name}")
        print("=" * 65)

        # ── Step 2 self-test (load_raw_chain) ────────────────────────────────
        df = load_raw_chain(str(f))

        n_final = len(df)
        print()
        print(f"  FINAL row count:        {n_final:,}")

        if n_final == 0:
            all_passed = False
            raw = pd.read_csv(str(f))
            print("  FAIL — zero rows after filtering.  Raw diagnostics:")
            print("    unique issuer:", raw["issuer"].unique().tolist())
            print("    unique exercise_style:", raw["exercise_style"].unique().tolist())
            print("    unique index_flag:", raw["index_flag"].unique().tolist())
            continue

        t_min, t_max = df["T_years"].min(), df["T_years"].max()
        k_min, k_max = df["K"].min(), df["K"].max()
        n_exp = df["exdate"].nunique()
        print(f"  T_years range:          {t_min:.4f}  –  {t_max:.4f}")
        print(f"  K range:                {k_min:.1f}  –  {k_max:.1f}")
        print(f"  distinct exdates:       {n_exp}")

        # ── Step 3 self-test (select_target_maturities) ──────────────────────
        print()
        print(f"  Expiration list (first 10 rows of {n_exp} exdates):")
        exp_tbl = list_expirations(df)
        print(exp_tbl.head(10).to_string(index=False))

        print()
        print(f"  Target-maturity selection (targets={TARGET_DAYS}):")
        buckets = select_target_maturities(df, target_days=TARGET_DAYS)

        # Validate: must have exactly 5 buckets, each non-empty, no duplicates
        n_buckets = len(buckets)
        if n_buckets != len(TARGET_DAYS):
            print(f"  FAIL — expected {len(TARGET_DAYS)} buckets, got {n_buckets}")
            all_passed = False
            continue

        bucket_sizes = {t: len(sub) for t, sub in buckets.items()}
        empty = [t for t, n in bucket_sizes.items() if n == 0]
        if empty:
            print(f"  FAIL — empty buckets for targets: {empty}")
            all_passed = False
            continue

        # Check no two targets collapsed onto the same exdate
        chosen_exdates = [sub["exdate"].iloc[0] for sub in buckets.values()]
        if len(set(pd.Timestamp(e) for e in chosen_exdates)) < len(TARGET_DAYS):
            print("  FAIL — two or more targets share the same exdate")
            all_passed = False
            continue

        print(f"  rows per bucket: { {t: n for t, n in bucket_sizes.items()} }")
        print("  PASS")

    print()
    print("=" * 65)
    print("OVERALL:", "PASS" if all_passed else "FAIL")
    print("=" * 65)
    sys.exit(0 if all_passed else 1)
