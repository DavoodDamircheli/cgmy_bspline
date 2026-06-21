"""
Diagnostic funnel for the Section 7 data pipeline (Steps 2-5).

For a given snapshot date, tracks how many quotes survive at each filtering
stage, broken out per nominal target maturity {1w, 1m, 3m, 6m, 1y}. Does not
modify any filter in data_prep.py / build_surface.py -- it duplicates the
filter logic so each stage boundary can be inspected independently of the
others (in particular, Stage 0 here uses the closest-expiry match computed
*before* the open-interest filter, whereas the production pipeline applies
the open-interest filter first; this lets us tell apart a "no listed expiry
nearby" miss from a "expiry exists but gets filtered out downstream" miss).

Run:
    python calibration/debug_data_survival.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from calibration.parity import fit_forward_discount

_TARGET_DAYS = (7, 30, 93, 182, 366)
_TARGET_LABELS = {7: "1w", 30: "1m", 93: "3m", 182: "6m", 366: "1y"}
_KAPPA_LO, _KAPPA_HI = 0.85, 1.15
_SPREAD_THRESHOLDS = (0.05, 0.15)


# ---------------------------------------------------------------------------
# Stage 0/1: quality filters (a)-(g) [no OI], then OI filter, then closest-
# expiry match
# ---------------------------------------------------------------------------

def _quality_filter_preOI(csv_path: str) -> pd.DataFrame:
    """Reproduce data_prep.load_raw_chain filters (a)-(g), skipping (h) OI."""
    df = pd.read_csv(csv_path, parse_dates=["date", "exdate"])

    df["K"] = df["strike_price"] / 1000.0
    df["mid"] = (df["best_bid"] + df["best_offer"]) / 2.0
    df["spread"] = df["best_offer"] - df["best_bid"]
    df["rel_spread"] = np.where(df["mid"] > 0, df["spread"] / df["mid"], np.inf)
    df["T_years"] = (df["exdate"] - df["date"]).dt.days / 365.0

    unique_issuers = df["issuer"].unique()
    issuer_filter = "CBOE S&P 500 INDEX"
    if issuer_filter not in unique_issuers:
        from difflib import get_close_matches
        matches = get_close_matches(issuer_filter, unique_issuers, n=1, cutoff=0.0)
        issuer_filter = matches[0] if matches else unique_issuers[0]

    df = df[df["exercise_style"] == "E"]
    df = df[df["index_flag"] == 1]
    df = df[df["issuer"] == issuer_filter]
    df = df[df["best_bid"] >= 0]
    df = df[df["best_offer"] > df["best_bid"]]
    df = df[df["mid"] > 0]
    df = df[df["impl_volatility"].notna()]
    return df.reset_index(drop=True)


def _closest_expiry_buckets(df: pd.DataFrame, target_days=_TARGET_DAYS) -> dict:
    """Map each nominal target to the sub-DataFrame of its closest listed exdate."""
    trade_date = df["date"].iloc[0]
    df2 = df.assign(days_to_expiry=((df["exdate"] - trade_date).dt.days).astype(int))
    expirations = (
        df2.groupby("exdate", sort=True)["days_to_expiry"].first().reset_index()
    )

    out = {}
    for tgt in target_days:
        if expirations.empty:
            out[tgt] = df2.iloc[0:0]
            continue
        idx = (expirations["days_to_expiry"] - tgt).abs().idxmin()
        exdate = expirations.loc[idx, "exdate"]
        out[tgt] = df2[df2["exdate"] == exdate].copy()
    return out


# ---------------------------------------------------------------------------
# Full per-date funnel
# ---------------------------------------------------------------------------

def run_funnel(date_str: str, csv_path: str) -> dict:
    """Return {tgt_days: {stage_name: count, ...}} for one date."""
    df_preOI = _quality_filter_preOI(csv_path)
    stage0_buckets = _closest_expiry_buckets(df_preOI, _TARGET_DAYS)

    results = {}
    for tgt in _TARGET_DAYS:
        bucket0 = stage0_buckets[tgt]
        n_stage0 = len(bucket0)

        bucket1 = bucket0[bucket0["open_interest"] > 100]
        n_stage1 = len(bucket1)

        rec = {
            "stage0_raw": n_stage0,
            "stage1_oi": n_stage1,
            "rel_spread_min": float(bucket0["rel_spread"].min()) if n_stage0 else float("nan"),
            "rel_spread_median": float(bucket0["rel_spread"].median()) if n_stage0 else float("nan"),
            "rel_spread_max": float(bucket0["rel_spread"].max()) if n_stage0 else float("nan"),
        }

        # Parity fit (Step 4) happens on the post-OI bucket, BEFORE the
        # spread/moneyness filters are applied -- matching build_surface_date's
        # actual order (it fits F,D on `sub`, then filters `work` derived from
        # that same `sub`). Using F from a spread-filtered subset would not
        # match production behaviour.
        F = None
        if n_stage1 > 0:
            calls = bucket1[bucket1["cp_flag"] == "C"]
            puts = bucket1[bucket1["cp_flag"] == "P"]
            T = float(bucket1["T_years"].iloc[0])
            try:
                F, _, _ = fit_forward_discount(
                    calls, puts, T, context=f"{date_str}/{tgt}d"
                )
            except ValueError:
                F = None

        for thresh in _SPREAD_THRESHOLDS:
            key = f"{int(thresh * 100):02d}"
            bucket2 = bucket1[bucket1["rel_spread"] < thresh]
            n_stage2 = len(bucket2)
            rec[f"stage2_spread{key}"] = n_stage2

            if n_stage2 == 0 or F is None:
                rec[f"stage3_mny{key}"] = 0 if n_stage2 == 0 else float("nan")
                rec[f"stage4_otm{key}"] = 0 if n_stage2 == 0 else float("nan")
                continue

            kappa = bucket2["K"] / F
            mny_mask = (kappa >= _KAPPA_LO) & (kappa <= _KAPPA_HI)
            bucket3 = bucket2[mny_mask]
            n_stage3 = len(bucket3)
            rec[f"stage3_mny{key}"] = n_stage3

            otm_mask = (
                ((bucket3["K"] < F) & (bucket3["cp_flag"] == "P"))
                | ((bucket3["K"] >= F) & (bucket3["cp_flag"] == "C"))
            )
            bucket4 = bucket3[otm_mask]
            rec[f"stage4_otm{key}"] = len(bucket4)

        results[tgt] = rec

    return results


def _stage5_from_surface(date_str: str, spread_label: str) -> dict:
    """Stage 5: rows per nominal maturity actually in the saved surface CSV."""
    path = ROOT / "data" / "processed" / f"surface_{date_str}_{spread_label}.csv"
    if not path.exists():
        return {tgt: 0 for tgt in _TARGET_DAYS}

    df = pd.read_csv(path)
    if df.empty:
        return {tgt: 0 for tgt in _TARGET_DAYS}

    # Map each row's actual T to the nearest nominal target (same convention
    # as select_target_maturities: closest by |actual_days - target|).
    actual_days_per_row = (df["T"] * 365).round().astype(int)
    out = {tgt: 0 for tgt in _TARGET_DAYS}
    for exdate, grp in df.groupby("exdate"):
        actual_d = int(round(float(grp["T"].iloc[0]) * 365))
        nearest_tgt = min(_TARGET_DAYS, key=lambda t: abs(t - actual_d))
        out[nearest_tgt] += len(grp)
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_funnel_table(date_str: str, funnel: dict, stage5_05: dict, stage5_15: dict):
    print(f"\n{'=' * 100}")
    print(f"Funnel: {date_str}")
    print("=" * 100)
    hdr = (
        f"{'tgt':>5} {'stage0_raw':>11} {'stage1_oi':>10} "
        f"{'stage2_sp05':>11} {'stage3_mny05':>13} {'stage4_otm05':>13} {'stage5_05':>10}   |  "
        f"{'stage2_sp15':>11} {'stage3_mny15':>13} {'stage4_otm15':>13} {'stage5_15':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tgt in _TARGET_DAYS:
        rec = funnel[tgt]
        label = _TARGET_LABELS[tgt]
        print(
            f"{label:>5} {rec['stage0_raw']:>11} {rec['stage1_oi']:>10} "
            f"{rec['stage2_spread05']:>11} {rec['stage3_mny05']:>13} {rec['stage4_otm05']:>13} "
            f"{stage5_05[tgt]:>10}   |  "
            f"{rec['stage2_spread15']:>11} {rec['stage3_mny15']:>13} {rec['stage4_otm15']:>13} "
            f"{stage5_15[tgt]:>10}"
        )


def print_spread_distribution(date_str: str, funnel: dict):
    print(f"\nRaw bid-ask rel_spread (= spread/mid) distribution BEFORE any filtering, {date_str}:")
    hdr = f"{'tgt':>5} {'n':>6} {'min':>10} {'median':>10} {'max':>10}"
    print(hdr)
    print("-" * len(hdr))
    for tgt in _TARGET_DAYS:
        rec = funnel[tgt]
        label = _TARGET_LABELS[tgt]
        print(
            f"{label:>5} {rec['stage0_raw']:>6} "
            f"{rec['rel_spread_min']:>10.4f} {rec['rel_spread_median']:>10.4f} "
            f"{rec['rel_spread_max']:>10.4f}"
        )


def classify(date_str: str, funnel: dict, stage5_05: dict):
    print(f"\nSELF-TEST classification, {date_str}:")
    for tgt in (7, 182, 366):
        label = _TARGET_LABELS[tgt]
        rec = funnel[tgt]
        s0, s1 = rec["stage0_raw"], rec["stage1_oi"]
        s2 = rec["stage2_spread05"]
        s3 = rec["stage3_mny05"]
        s4 = rec["stage4_otm05"]
        s5 = stage5_05[tgt]

        if s0 == 0:
            verdict = "(a) CLOSEST-EXPIRY MISS"
        elif s1 == 0:
            verdict = "(b) OI FILTER"
        elif s2 == 0:
            verdict = "(c) SPREAD FILTER"
        elif s4 == 0 or s5 == 0:
            verdict = "(d) MONEYNESS/OTM FILTER"
        else:
            verdict = "SURVIVES (nonzero at stage5)"

        print(
            f"  {label:>3} (target={tgt}d): "
            f"stage0={s0} stage1_oi={s1} stage2_spread05={s2} "
            f"stage3_mny05={s3} stage4_otm05={s4} stage5={s5}  ->  {verdict}"
        )


def main():
    dates = [
        ("2020-03-18", ROOT / "data" / "2020-03-18.csv"),
        ("2020-01-15", ROOT / "data" / "2020-01-15.csv"),
    ]

    funnels = {}
    for date_str, csv_path in dates:
        funnels[date_str] = run_funnel(date_str, str(csv_path))

    for date_str, _ in dates:
        funnel = funnels[date_str]
        stage5_05 = _stage5_from_surface(date_str, "spread05")
        stage5_15 = _stage5_from_surface(date_str, "spread15")
        print_funnel_table(date_str, funnel, stage5_05, stage5_15)

    print_spread_distribution("2020-03-18", funnels["2020-03-18"])

    classify("2020-03-18", funnels["2020-03-18"], _stage5_from_surface("2020-03-18", "spread05"))


if __name__ == "__main__":
    main()
