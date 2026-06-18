"""
Build cleaned OTM implied-volatility surfaces for Section 7 calibration.

For each snapshot date this script:
  1. Loads and quality-filters the raw chain (Step 2).
  2. Selects the 5 target maturities (Step 3).
  3. Fits F and D per maturity via put-call parity (Step 4).
  4. Computes log_moneyness, forward_moneyness (kappa), and BS vega.
  5. Applies moneyness filter  kappa ∈ [0.85, 1.15].
  6. Applies the OTM selection rule (puts for K<F, calls for K>=F).
  7. Applies a relative-spread filter and writes a CSV per date.

Usage:
    python calibration/build_surface.py        # from project root
    python -m calibration.build_surface        # same, module form
"""

import io
import contextlib
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Black-Scholes forward vega (dependency-free implementation)
# ---------------------------------------------------------------------------

def bs_vega_forward(
    F: np.ndarray,
    K: np.ndarray,
    T: float,
    D: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Vega of a European option in the Black forward framework.

    vega = D * F * phi(d1) * sqrt(T)
    d1   = (ln(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))

    Vectorised over (F, K, sigma); returns nan where T<=0 or sigma<=0.
    """
    F     = np.asarray(F,     dtype=float)
    K     = np.asarray(K,     dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    out  = np.full_like(sigma, np.nan)
    good = (sigma > 0.0) & (F > 0.0) & (K > 0.0) & (T > 0.0)
    if good.any():
        sqrtT      = np.sqrt(T)
        d1         = (np.log(F[good] / K[good]) + 0.5 * sigma[good] ** 2 * T) \
                     / (sigma[good] * sqrtT)
        out[good]  = D * F[good] * norm.pdf(d1) * sqrtT
    return out


# ---------------------------------------------------------------------------
# Per-date surface builder
# ---------------------------------------------------------------------------

# Output column order
_OUT_COLS = [
    "date", "exdate", "T", "cp_flag",
    "K", "F", "D", "log_moneyness",
    "mid", "bid", "ask",
    "spread", "rel_spread", "open_interest",
    "market_iv", "bs_vega",
]

# Moneyness window
_KAPPA_LO, _KAPPA_HI = 0.85, 1.15

# Roadmap reference counts for ±30% PASS check: (n_ref, spread_thresh, date_str)
_ROADMAP_REFS = [
    ("2020-01-15", 0.05, 224),
    ("2020-03-18", 0.05, 113),
    ("2021-09-15", 0.05, 575),
    ("2020-03-18", 0.15, 243),
]


def build_surface_date(
    csv_path: str,
    spread_thresh: float,
    target_days: tuple = (7, 30, 93, 182, 366),
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the OTM surface for one snapshot date and one spread threshold.

    Parameters
    ----------
    csv_path      : path to the raw OptionMetrics CSV
    spread_thresh : relative-spread cap (e.g. 0.05 or 0.15)
    target_days   : tuple of target maturities in calendar days
    verbose       : if True, print per-maturity row counts

    Returns
    -------
    pd.DataFrame with columns _OUT_COLS, one row per surviving OTM quote.
    """
    from calibration.data_prep import load_raw_chain, select_target_maturities
    from calibration.parity import fit_forward_discount

    # ── 1. Load and filter chain (suppress Step-2 chatter) ───────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_raw_chain(csv_path)

    # ── 2. Select target maturities (suppress Step-3 chatter) ────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        buckets = select_target_maturities(df, target_days=target_days)

    maturity_frames = []

    for tgt, sub in buckets.items():
        T          = float(sub["T_years"].iloc[0])
        actual_d   = int(round(T * 365))
        exdate_val = sub["exdate"].iloc[0]

        calls = sub[sub["cp_flag"] == "C"]
        puts  = sub[sub["cp_flag"] == "P"]

        # ── 3. Fit F, D via put-call parity ──────────────────────────────────
        ctx = f"{pathlib.Path(csv_path).stem} / {tgt}d (actual {actual_d}d)"
        F, D, _ = fit_forward_discount(calls, puts, T, context=ctx)

        # ── 4. Derived columns ────────────────────────────────────────────────
        work = sub.copy()
        work["F"]             = F
        work["D"]             = D
        work["log_moneyness"] = np.log(work["K"] / F)
        work["kappa"]         = work["K"] / F
        work["bs_vega"]       = bs_vega_forward(
            F   = np.full(len(work), F),
            K   = work["K"].to_numpy(),
            T   = T,
            D   = D,
            sigma = work["impl_volatility"].to_numpy(),
        )

        n0 = len(work)

        # ── 5. Spread filter ──────────────────────────────────────────────────
        work = work[work["rel_spread"] < spread_thresh]
        n_spread = len(work)

        # ── 6. Moneyness filter ───────────────────────────────────────────────
        work = work[(work["kappa"] >= _KAPPA_LO) & (work["kappa"] <= _KAPPA_HI)]
        n_mono = len(work)

        # ── 7. OTM selection: puts for K < F, calls for K >= F ───────────────
        otm_mask = (
            ((work["K"] <  F) & (work["cp_flag"] == "P")) |
            ((work["K"] >= F) & (work["cp_flag"] == "C"))
        )
        work = work[otm_mask]
        n_otm = len(work)

        if verbose:
            print(
                f"    tgt={tgt:3d}d  act={actual_d:3d}d  F={F:7.1f}  "
                f"n_start={n0:4d}  "
                f"→spread={n_spread:4d}  →mono={n_mono:4d}  →OTM={n_otm:4d}"
            )

        if n_otm == 0:
            continue

        # ── 8. Rename and select output columns ──────────────────────────────
        out = work.rename(columns={
            "T_years":        "T",
            "best_bid":       "bid",
            "best_offer":     "ask",
            "impl_volatility":"market_iv",
        })[_OUT_COLS].copy()

        maturity_frames.append(out)

    if not maturity_frames:
        return pd.DataFrame(columns=_OUT_COLS)

    return pd.concat(maturity_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main driver: all 3 dates × 2 spread thresholds
# ---------------------------------------------------------------------------

def _stem(date_str: str) -> str:
    return date_str  # keep as YYYY-MM-DD for filenames


def main() -> None:
    import sys

    project_root = pathlib.Path(__file__).parents[1]
    data_dir     = project_root / "data"
    out_dir      = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    DATES = [
        (data_dir / "2020-01-15.csv", "2020-01-15"),
        (data_dir / "2020-03-18.csv", "2020-03-18"),
        (data_dir / "2021-09-15.csv", "2021-09-15"),
    ]
    SPREAD_CONFIGS = [
        (0.05, "spread05"),
        (0.15, "spread15"),
    ]

    # Build reference lookup for PASS check
    ref_lookup = {
        (date_str, thresh): n_ref
        for date_str, thresh, n_ref in _ROADMAP_REFS
    }

    all_pass = True
    surfaces: dict[tuple, pd.DataFrame] = {}   # (date_str, label) -> df

    for fpath, date_str in DATES:
        print()
        print("=" * 65)
        print(f"Date: {date_str}")
        print("=" * 65)

        for spread_thresh, suffix in SPREAD_CONFIGS:
            print(f"\n  Spread filter: rel_spread < {spread_thresh:.0%}  [{suffix}]")
            surf = build_surface_date(str(fpath), spread_thresh=spread_thresh)
            surfaces[(date_str, suffix)] = surf

            n_total = len(surf)
            # Per-maturity breakdown
            if not surf.empty:
                breakdown = (
                    surf.groupby("exdate", sort=True)
                    .agg(n_quotes=("K", "count"), T=("T", "first"))
                    .reset_index()
                )
                breakdown["actual_days"] = (breakdown["T"] * 365).round().astype(int)
                print("    breakdown by exdate:")
                for _, row in breakdown.iterrows():
                    print(f"      {str(row['exdate'].date()):>12}  "
                          f"({row['actual_days']:3d}d)  n={int(row['n_quotes']):4d}")

            print(f"    TOTAL OTM quotes: {n_total}")

            # PASS check vs roadmap reference (within ±30%)
            n_ref = ref_lookup.get((date_str, spread_thresh))
            if n_ref is not None:
                lo, hi = int(0.70 * n_ref), int(1.30 * n_ref)
                ok = lo <= n_total <= hi
                status = "PASS" if ok else "FAIL"
                if not ok:
                    all_pass = False
                print(f"    Roadmap ref ~{n_ref:4d}, ±30% band [{lo:4d},{hi:4d}]: "
                      f"{status} (got {n_total})")
                if not ok:
                    print(f"    NOTE: rerun build_surface_date with verbose=True "
                          f"to see per-step counts.")

            # Save CSV
            label = f"spread{'05' if spread_thresh == 0.05 else '15'}"
            fname = out_dir / f"surface_{date_str}_{label}.csv"
            surf.to_csv(fname, index=False)
            print(f"    Saved → {fname.relative_to(project_root)}")

    # ── Visual sanity check: head(5) of each primary (5%) surface ────────────
    print()
    print("=" * 65)
    print("Head(5) of each primary surface (spread < 5%)")
    print("=" * 65)
    for _, date_str in DATES:
        df_show = surfaces.get((date_str, "spread05"), pd.DataFrame())
        print(f"\n  {date_str}  (n={len(df_show):,})")
        if df_show.empty:
            print("  (empty)")
        else:
            pd.set_option("display.max_columns", 16)
            pd.set_option("display.width", 200)
            pd.set_option("display.float_format", "{:.4f}".format)
            print(df_show[_OUT_COLS].head(5).to_string(index=False))

    print()
    print("=" * 65)
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    print("=" * 65)


if __name__ == "__main__":
    main()
