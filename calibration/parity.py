"""
Put-call parity fit for SPX chains (Section 7, Step 3).

fit_forward_discount pairs calls and puts on identical strikes K,
fits the linear model  C_mid - P_mid = a - D*K  by OLS, and returns
the implied forward price F = a/D and discount factor D.
"""

import numpy as np
import pandas as pd
from scipy import stats


def fit_forward_discount(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    T: float,
    context: str = "",
) -> tuple[float, float, dict]:
    """Fit put-call parity to extract the forward price and discount factor.

    For each strike K present in both calls and puts, define
        y_i = C_mid(K_i) - P_mid(K_i)
    and fit the OLS line
        y_i = a - D * K_i   (slope = -D, intercept = a)

    Then:
        D  = -slope              (discount factor)
        F  =  a / D             (implied forward price)
        r  = -ln(D) / T        (implied zero rate)

    Parameters
    ----------
    calls, puts : pd.DataFrame
        Each must contain columns 'K' and 'mid'.  Rows with the same K
        from different strikes in calls/puts are paired by inner join.
    T       : float
        Time to expiry in years, used only to compute r_implied.
    context : str, optional
        Free-text label (e.g. "2020-01-15 / 93d") printed in warnings.

    Returns
    -------
    F    : float   implied forward price
    D    : float   discount factor
    diag : dict    keys: n_pairs, r_squared, slope, intercept, slope_se,
                         intercept_se, D, F, r_implied

    Raises
    ------
    ValueError  if n_pairs < 4 (fit is unreliable with too few strikes)
    """
    # ── 1. Inner join on strike ───────────────────────────────────────────────
    c_slim = calls[["K", "mid"]].rename(columns={"mid": "C_mid"})
    p_slim = puts[["K", "mid"]].rename(columns={"mid": "P_mid"})

    paired = pd.merge(c_slim, p_slim, on="K")
    n_pairs = len(paired)

    if n_pairs < 4:
        raise ValueError(
            f"put-call parity fit requires >= 4 paired strikes, "
            f"got {n_pairs}"
            + (f" [{context}]" if context else "")
        )

    # ── 2. Compute y = C_mid - P_mid and fit y = a - D*K ────────────────────
    K_arr = paired["K"].to_numpy(dtype=float)
    y_arr = (paired["C_mid"] - paired["P_mid"]).to_numpy(dtype=float)

    lr = stats.linregress(K_arr, y_arr)

    slope     = float(lr.slope)
    intercept = float(lr.intercept)
    r_sq      = float(lr.rvalue ** 2)

    D = -slope
    F = intercept / D if D != 0.0 else float("nan")
    r_implied = -np.log(D) / T if (T > 0 and D > 0) else float("nan")

    diag = {
        "n_pairs":      n_pairs,
        "r_squared":    r_sq,
        "slope":        slope,
        "intercept":    intercept,
        "slope_se":     float(lr.stderr),
        "intercept_se": float(lr.intercept_stderr),
        "D":            D,
        "F":            F,
        "r_implied":    r_implied,
    }

    # ── 3. Guard-rail warnings ────────────────────────────────────────────────
    ctx = f" [{context}]" if context else ""

    if r_sq < 0.95:
        print(
            f"  WARNING: R² = {r_sq:.4f} < 0.95 for parity fit{ctx}. "
            f"Stale or crossed quotes may contaminate the chain."
        )

    # Discount factor sanity band for T <= 1.05 years
    if T <= 1.05 and not (0.80 < D < 1.05):
        print(
            f"  WARNING: D = {D:.5f} outside (0.80, 1.05) for T = {T:.4f}y{ctx}. "
            f"Check for data or sign errors."
        )

    return F, D, diag


# ---------------------------------------------------------------------------
# Self-test: all 3 dates × 5 maturities
# ---------------------------------------------------------------------------

# Rough SPX levels for gross F sanity check (0.7x – 1.3x band)
_SPX_ROUGH = {
    "2020-01-15": 3310.0,
    "2020-03-18": 2500.0,   # mid-COVID crash, wide band intentional
    "2021-09-15": 4460.0,
}


if __name__ == "__main__":
    import sys
    import io
    import contextlib
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
    from calibration.data_prep import load_raw_chain, select_target_maturities

    DATA_DIR = pathlib.Path(__file__).parents[1] / "data"
    FILES = [
        (DATA_DIR / "2020-01-15.csv", "2020-01-15"),
        (DATA_DIR / "2020-03-18.csv", "2020-03-18"),
        (DATA_DIR / "2021-09-15.csv", "2021-09-15"),
    ]
    TARGET_DAYS = (7, 30, 93, 182, 366)

    # ── Table header ─────────────────────────────────────────────────────────
    hdr = (f"{'date':>12}  {'tgt_d':>6}  {'act_d':>6}  "
           f"{'F':>8}  {'D':>8}  {'r_impl':>7}  "
           f"{'n_pairs':>7}  {'R²':>8}")
    print(hdr)
    print("-" * len(hdr))

    n_total   = 0
    n_pass_r2 = 0
    n_fail_r2 = 0
    fail_r2_rows = []
    all_finite  = True

    for fpath, date_str in FILES:
        # Load quietly
        with contextlib.redirect_stdout(io.StringIO()):
            df = load_raw_chain(str(fpath))
            buckets = select_target_maturities(df, target_days=TARGET_DAYS)

        spx_ref = _SPX_ROUGH[date_str]
        f_lo, f_hi = 0.7 * spx_ref, 1.3 * spx_ref

        for tgt, sub in buckets.items():
            T        = float(sub["T_years"].iloc[0])
            act_days = int(round(T * 365))
            calls    = sub[sub["cp_flag"] == "C"]
            puts     = sub[sub["cp_flag"] == "P"]

            ctx = f"{date_str} / {tgt}d"
            try:
                F, D, diag = fit_forward_discount(calls, puts, T, context=ctx)
            except ValueError as exc:
                print(f"  ERROR {ctx}: {exc}")
                all_finite = False
                continue

            n_total  += 1
            r2        = diag["r_squared"]
            r_impl    = diag["r_implied"]
            n_pairs   = diag["n_pairs"]

            if r2 >= 0.95:
                n_pass_r2 += 1
            else:
                n_fail_r2 += 1
                fail_r2_rows.append(ctx)

            finite_flag = (np.isfinite(F) and np.isfinite(D))
            if not finite_flag:
                all_finite = False

            # SPX gross-level check
            f_warn = ""
            if not (f_lo <= F <= f_hi):
                f_warn = f"  <-- F outside [{f_lo:.0f},{f_hi:.0f}] for SPX ref {spx_ref:.0f}"

            print(
                f"{date_str:>12}  {tgt:>6}  {act_days:>6}  "
                f"{F:>8.2f}  {D:>8.5f}  {r_impl:>7.4f}  "
                f"{n_pairs:>7}  {r2:>8.6f}"
                f"{f_warn}"
            )

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print(f"Rows evaluated: {n_total}/15")
    print(f"R² >= 0.95:     {n_pass_r2}/{n_total}"
          + ("  PASS" if n_fail_r2 <= 3 else "  FAIL"))
    if fail_r2_rows:
        print(f"  R² < 0.95 in: {fail_r2_rows}")
    print(f"All F,D finite: {'PASS' if all_finite else 'FAIL'}")
    overall = all_finite and (n_fail_r2 <= 3) and (n_total == 15)
    print()
    print("=" * 65)
    print("OVERALL:", "PASS" if overall else "FAIL")
    print("=" * 65)
    sys.exit(0 if overall else 1)
