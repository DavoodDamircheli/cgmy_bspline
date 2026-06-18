"""
Multi-date CGMY calibration driver (Section 7, Step 8 — all dates).

Runs calibrate_one_date on the four target surfaces and saves results to
cgmy_bspline/results/calibration_results.csv.

Fixed optimizer settings (reported here for reproducibility; these exact
values must appear in the paper's Section 7 as "fixed optimizer settings"):
    seed      = 0
    de_maxiter = 300
    de_popsize = 15

Surfaces calibrated:
    surface_2020-01-15_spread05.csv   regime: low-vol
    surface_2020-03-18_spread05.csv   regime: crisis
    surface_2020-03-18_spread15.csv   regime: crisis (robustness check)
    surface_2021-09-15_spread05.csv   regime: recovery

The 2020-03-18 pair (spread05 vs spread15) is the robustness check
required by the data-roadmap's Section 5.
"""

import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from calibration.calibrate import calibrate_one_date

# ---------------------------------------------------------------------------
# Fixed optimizer settings — do NOT change without updating the paper text
# ---------------------------------------------------------------------------

_SEED       = 0
_DE_MAXITER = 300
_DE_POPSIZE = 15

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

_DATA_DIR = pathlib.Path(__file__).parents[1] / "data" / "processed"
_OUT_DIR  = pathlib.Path(__file__).parents[1] / "cgmy_bspline" / "results"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_RUNS = [
    # (date_str, spread_filter_label, regime, csv_filename)
    ("2020-01-15", "05", "low-vol",  "surface_2020-01-15_spread05.csv"),
    ("2020-03-18", "05", "crisis",   "surface_2020-03-18_spread05.csv"),
    ("2020-03-18", "15", "crisis",   "surface_2020-03-18_spread15.csv"),
    ("2021-09-15", "05", "recovery", "surface_2021-09-15_spread05.csv"),
]

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    rows = []
    t_total_start = time.time()

    for date_str, spread_label, regime, csv_name in _RUNS:
        csv_path = str(_DATA_DIR / csv_name)

        print()
        print("=" * 70)
        print(f"Calibrating: {date_str}  spread={spread_label}%  regime={regime}")
        print(f"  seed={_SEED}  de_maxiter={_DE_MAXITER}  de_popsize={_DE_POPSIZE}")
        print("=" * 70)

        try:
            result = calibrate_one_date(
                surface_csv_path=csv_path,
                r=0.0,
                q=0.0,
                seed=_SEED,
                de_maxiter=_DE_MAXITER,
                de_popsize=_DE_POPSIZE,
            )
            theta = result["theta_cgmy"]
            rows.append({
                "date":          date_str,
                "spread_filter": spread_label,
                "regime":        regime,
                # Calibrated parameters
                "C":             theta.C,
                "G":             theta.G,
                "M":             theta.M,
                "Y":             theta.Y,
                # Fit quality
                "iv_rmse":       result["iv_rmse"],
                "max_iv_err":    result["max_iv_err"],
                "de_loss":       result["de_loss"],
                "lbfgs_loss":    result["lbfgs_loss"],
                # Optimizer metadata
                "de_nfev":       result["de_nfev"],
                "lbfgs_nfev":    result["lbfgs_nfev"],
                "total_time_sec": result["total_time_sec"],
                # Surface metadata
                "n_quotes":      result["n_quotes"],
                "S0":            result["S0"],
                # Fixed optimizer settings (for traceability)
                "seed":          _SEED,
                "de_maxiter":    _DE_MAXITER,
                "de_popsize":    _DE_POPSIZE,
            })
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
            rows.append({
                "date": date_str, "spread_filter": spread_label,
                "regime": regime, "C": None, "G": None, "M": None, "Y": None,
                "iv_rmse": None, "max_iv_err": None, "de_loss": None,
                "lbfgs_loss": None, "de_nfev": None, "lbfgs_nfev": None,
                "total_time_sec": None, "n_quotes": None, "S0": None,
                "seed": _SEED, "de_maxiter": _DE_MAXITER, "de_popsize": _DE_POPSIZE,
            })

    t_total = time.time() - t_total_start

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_path = _OUT_DIR / "calibration_results.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    print()
    print(f"Saved → {out_path}")
    print(f"Total wall time for all runs: {t_total:.0f}s ({t_total/60:.1f} min)")

    # ── Self-test 1: Full table ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SELF-TEST 1: Full calibration_results.csv")
    print("=" * 70)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.5f}".format)
    display_cols = [
        "date", "spread_filter", "regime",
        "C", "G", "M", "Y",
        "iv_rmse", "max_iv_err",
        "de_loss", "lbfgs_loss",
        "n_quotes", "total_time_sec",
    ]
    print(df[display_cols].to_string(index=False))

    # ── Self-test 2: 2020-03-18 robustness check (05 vs 15 spread) ────────────
    print()
    print("=" * 70)
    print("SELF-TEST 2: Crisis-date robustness (2020-03-18 spread05 vs spread15)")
    print("=" * 70)
    r05 = df[(df["date"] == "2020-03-18") & (df["spread_filter"] == "05")].iloc[0]
    r15 = df[(df["date"] == "2020-03-18") & (df["spread_filter"] == "15")].iloc[0]

    for param in ["C", "G", "M", "Y", "iv_rmse"]:
        v05 = r05[param]
        v15 = r15[param]
        if v05 is not None and v15 is not None and v05 != 0:
            abs_diff  = abs(v15 - v05)
            pct_diff  = abs_diff / abs(v05) * 100
            print(f"  {param:8s}: spread05={v05:.4f}  spread15={v15:.4f}  "
                  f"|Δ|={abs_diff:.4f}  ({pct_diff:.1f}%)")
        else:
            print(f"  {param:8s}: missing data — cannot compare")

    print()
    # Narrative finding
    params_stable = all(
        abs(r15[p] - r05[p]) / max(abs(r05[p]), 1e-10) < 0.10
        for p in ["C", "G", "M", "Y"]
        if r05[p] is not None and r15[p] is not None
    )
    if params_stable:
        print("  Finding: CGMY parameters are stable (< 10% relative change) "
              "under the\n  looser 15%-spread filter — consistent with the crisis "
              "surface being\n  well-identified despite wider bid-ask spreads.")
    else:
        print("  Finding: CGMY parameters shift by > 10% for at least one "
              "component\n  under the 15%-spread filter — worth noting in the "
              "paper as evidence\n  that crisis-date quotes are sensitive to "
              "spread-filter choice.")

    # ── Self-test 3: Qualitative financial story ───────────────────────────────
    print()
    print("=" * 70)
    print("SELF-TEST 3: Qualitative financial story check")
    print("=" * 70)

    # Only primary (spread05) rows for the story check
    primary = df[df["spread_filter"] == "05"].copy()
    print()
    print(f"  {'date':>12}  {'regime':>10}  {'C':>9}  {'G':>9}  "
          f"{'M':>9}  {'Y':>9}  {'iv_rmse':>9}")
    print("  " + "-" * 72)
    for _, row in primary.iterrows():
        if row["C"] is not None:
            print(f"  {row['date']:>12}  {row['regime']:>10}  "
                  f"{row['C']:>9.4f}  {row['G']:>9.4f}  "
                  f"{row['M']:>9.4f}  {row['Y']:>9.4f}  "
                  f"{row['iv_rmse']:>9.4f}")
        else:
            print(f"  {row['date']:>12}  {row['regime']:>10}  (failed)")

    print()

    # Check Y stability across regimes
    y_vals = [row["Y"] for _, row in primary.iterrows() if row["Y"] is not None]
    if len(y_vals) >= 2:
        y_range = max(y_vals) - min(y_vals)
        y_mean  = np.mean(y_vals)
        if y_range / y_mean < 0.15:
            print(f"  Y* check (PASS): Y* ranges from {min(y_vals):.4f} to "
                  f"{max(y_vals):.4f} (spread {y_range:.4f}, {y_range/y_mean*100:.1f}%"
                  f" of mean {y_mean:.4f}) — consistent with Y being a regime-stable "
                  f"fine-structure index.")
        else:
            print(f"  Y* check (FLAG): Y* ranges widely from {min(y_vals):.4f} to "
                  f"{max(y_vals):.4f} (spread {y_range:.4f}, {y_range/y_mean*100:.1f}%"
                  f" of mean {y_mean:.4f}) — Y is not regime-stable in this sample; "
                  f"may reflect boundary effects (Y hitting 1.05 lower bound).")
    else:
        print("  Y* check: insufficient data")

    print()

    # Check C* crisis elevation
    c_by_date = {row["date"]: row["C"] for _, row in primary.iterrows() if row["C"] is not None}
    c_crisis  = c_by_date.get("2020-03-18")
    c_others  = {d: v for d, v in c_by_date.items() if d != "2020-03-18"}

    if c_crisis is not None and c_others:
        c_max_other = max(c_others.values())
        if c_crisis > c_max_other:
            print(f"  C* check (PASS): C*_crisis = {c_crisis:.4f} > "
                  f"max(C*_others) = {c_max_other:.4f} — crisis date shows "
                  f"elevated jump intensity as expected.")
        else:
            print(f"  C* check (FLAG): C*_crisis = {c_crisis:.4f} is NOT "
                  f"higher than other dates (max = {c_max_other:.4f}). "
                  f"This is counterintuitive — double-check the 2020-03-18 "
                  f"surface construction (Step 5) and whether the wide bid-ask "
                  f"spreads during the crisis led to a sparse or contaminated "
                  f"surface. Reporting the raw number without the 'elevated C*' "
                  f"claim in the paper text.")
    else:
        print("  C* check: missing crisis or non-crisis data — cannot evaluate")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
