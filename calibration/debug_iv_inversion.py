"""
Diagnostic for the 1-month IV-inversion step (Section 7, Step 7 objective).

For the 1-month maturity slice, pulls every (K, model_price) pair exactly as
iv_rmse_and_max_err evaluates them (same pricer_fn call, same per-row F/D),
and prints -- BEFORE calling brentq -- the bracket endpoints
bs_put_price(..., vol_lo) / bs_put_price(..., vol_hi) and the no-arbitrage
put intrinsic, so bracket failures and sub-intrinsic prices can be told
apart from each other and from a successful-but-extreme inversion.

Convention note: model_price from pricer_fn is OTM-selected (put for K<F,
call for K>=F), matching the production convention in calibration/pricer.py
and calibration/objective.py. Implied vol is identical whether inverted via
the OTM-selected formula or its put-call-parity-equivalent put price, so
every row here is converted to its PUT-equivalent price
(put_equiv = model_price for K<F; put_equiv = model_price - D*(F-K) for
K>=F) and evaluated uniformly against the PUT formula and PUT intrinsic
bound, per the task's request to use bs_put_price throughout.

Does NOT modify calibration/objective.py or calibration/pricer.py.

Run:
    python calibration/debug_iv_inversion.py
"""

import math
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from calibration.calibrate import _load_surface
from calibration.pricer import make_cached_pricer
from calibration.objective import implied_vol_from_price

RES_DIR = ROOT / "cgmy_bspline" / "results"
DATA_DIR = ROOT / "data" / "processed"

# (date_str, csv_name, role)
_DATES = [
    ("2020-03-18", "surface_2020-03-18_spread05.csv", "PRIMARY (crisis)"),
    ("2020-01-15", "surface_2020-01-15_spread05.csv", "CONTROL (low-vol)"),
]

_TARGET_T_LO, _TARGET_T_HI = 0.060, 0.120  # "1 month" nominal tenor bucket (matches make_figures.py _TENORS)

# Settings matching make_cached_pricer's defaults / make_figures.py diagnostics.
_N_EVAL, _N_TAU, _BW, _HALF_W = 64, 10, 128, 2.0

# Production default bracket used live inside implied_vol_from_price.
_PROD_VOL_LO, _PROD_VOL_HI = 1e-4, 5.0

# Wider probe bracket requested for this diagnostic.
_PROBE_VOL_LO, _PROBE_VOL_HI = 0.001, 10.0


def bs_put_price(F: float, K: float, T: float, D: float, sigma: float) -> float:
    """Black-76 forward put price -- same convention as objective._black_otm's put branch."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def _load_theta(date_str: str):
    df_cal = pd.read_csv(RES_DIR / "calibration_results.csv", dtype={"spread_filter": str})
    row = df_cal[(df_cal["date"] == date_str) & (df_cal["spread_filter"] == "05")].iloc[0]
    return float(row["C"]), float(row["G"]), float(row["M"]), float(row["Y"])


def analyze_date(date_str: str, csv_name: str):
    surface, S0, rates = _load_surface(str(DATA_DIR / csv_name))
    gkw = dict(N=_N_EVAL, N_tau=_N_TAU, domain_half_width=_HALF_W, bandwidth=_BW)
    pricer = make_cached_pricer(surface, S0, rates, grid_kwargs=gkw)

    theta_cgmy = _load_theta(date_str)

    df_1m, T_sel = None, None
    for df_mat in surface.values():
        T = float(df_mat["T"].iloc[0])
        if _TARGET_T_LO <= T <= _TARGET_T_HI:
            df_1m, T_sel = df_mat, T
            break
    if df_1m is None:
        return None

    K_arr = df_1m["K"].to_numpy(dtype=float)
    F_arr = df_1m["F"].to_numpy(dtype=float)
    D_arr = df_1m["D"].to_numpy(dtype=float)
    lm_arr = df_1m["log_moneyness"].to_numpy(dtype=float)
    mkt_iv_arr = df_1m["market_iv"].to_numpy(dtype=float)

    model_prices = np.asarray(pricer(theta_cgmy, T_sel, K_arr, option_type="auto"), dtype=float)

    rows = []
    for K, F, D, lm, mp, miv in zip(K_arr, F_arr, D_arr, lm_arr, model_prices, mkt_iv_arr):
        is_put = K < F
        put_equiv = mp if is_put else mp - D * (F - K)  # parity: Put = Call - D*(F-K)

        intrinsic = D * max(K - F, 0.0)
        p_lo_prod = bs_put_price(F, K, T_sel, D, _PROD_VOL_LO)
        p_hi_prod = bs_put_price(F, K, T_sel, D, _PROD_VOL_HI)
        p_lo_probe = bs_put_price(F, K, T_sel, D, _PROBE_VOL_LO)
        p_hi_probe = bs_put_price(F, K, T_sel, D, _PROBE_VOL_HI)

        bracket_fail_prod = not (p_lo_prod < put_equiv < p_hi_prod)
        bracket_fail_probe = not (p_lo_probe < put_equiv < p_hi_probe)
        sub_intrinsic = put_equiv < intrinsic - 1e-8

        model_iv_prod = implied_vol_from_price(mp, F, K, T_sel, D)  # exactly as iv_rmse_and_max_err computes it
        model_iv_probe = implied_vol_from_price(mp, F, K, T_sel, D, lo=_PROBE_VOL_LO, hi=_PROBE_VOL_HI)

        rows.append(dict(
            K=K, F=F, D=D, log_moneyness=lm, is_put=is_put,
            model_price=mp, put_equiv=put_equiv, intrinsic=intrinsic,
            p_lo_prod=p_lo_prod, p_hi_prod=p_hi_prod,
            p_lo_probe=p_lo_probe, p_hi_probe=p_hi_probe,
            bracket_fail_prod=bracket_fail_prod, bracket_fail_probe=bracket_fail_probe,
            sub_intrinsic=sub_intrinsic,
            market_iv=miv, model_iv_prod=model_iv_prod, model_iv_probe=model_iv_probe,
        ))

    df = pd.DataFrame(rows).sort_values("log_moneyness").reset_index(drop=True)
    return T_sel, S0, df


def print_rows(date_str: str, role: str, T_sel: float, S0: float, df: pd.DataFrame):
    print(f"\n{'=' * 110}")
    print(f"{role}: {date_str}   T={T_sel:.4f}y (1-month)   S0={S0:.2f}   n={len(df)}")
    print("=" * 110)
    hdr = (f"{'K':>9} {'lm':>7} {'type':>4} {'model_px':>10} {'put_equiv':>10} {'intrinsic':>10} "
           f"{'bs(lo=1e-4)':>11} {'bs(hi=5)':>10} {'bs(lo=.001)':>11} {'bs(hi=10)':>10} "
           f"{'mkt_iv':>7} {'mdl_iv_prod':>11} {'mdl_iv_probe':>12}  flags")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        flags = []
        if r["bracket_fail_prod"]:
            flags.append("BRACKET_FAIL(prod)")
        if r["bracket_fail_probe"]:
            flags.append("BRACKET_FAIL(probe)")
        if r["sub_intrinsic"]:
            flags.append("SUB_INTRINSIC")
        flag_str = ",".join(flags) if flags else ""

        def fmt_iv(v):
            return f"{v*100:6.2f}" if np.isfinite(v) else "   nan"

        print(
            f"{r['K']:>9.2f} {r['log_moneyness']:>7.3f} {'P' if r['is_put'] else 'C':>4} "
            f"{r['model_price']:>10.4f} {r['put_equiv']:>10.4f} {r['intrinsic']:>10.4f} "
            f"{r['p_lo_prod']:>11.6f} {r['p_hi_prod']:>10.4f} {r['p_lo_probe']:>11.6f} {r['p_hi_probe']:>10.4f} "
            f"{r['market_iv']*100:>7.2f} {fmt_iv(r['model_iv_prod']):>11} {fmt_iv(r['model_iv_probe']):>12}  {flag_str}"
        )


def main():
    summary = {}
    for date_str, csv_name, role in _DATES:
        result = analyze_date(date_str, csv_name)
        if result is None:
            print(f"\n{role}: {date_str} -- no 1-month slice found (filtered out).")
            continue
        T_sel, S0, df = result
        print_rows(date_str, role, T_sel, S0, df)

        n_bracket_fail_prod = int(df["bracket_fail_prod"].sum())
        n_bracket_fail_probe = int(df["bracket_fail_probe"].sum())
        n_sub_intrinsic = int(df["sub_intrinsic"].sum())
        n = len(df)

        print(f"\n  counts: bracket_fail(prod lo=1e-4,hi=5)={n_bracket_fail_prod}/{n}  "
              f"bracket_fail(probe lo=.001,hi=10)={n_bracket_fail_probe}/{n}  "
              f"sub_intrinsic={n_sub_intrinsic}/{n}")

        summary[date_str] = dict(
            role=role, n=n,
            n_bracket_fail_prod=n_bracket_fail_prod,
            n_bracket_fail_probe=n_bracket_fail_probe,
            n_sub_intrinsic=n_sub_intrinsic,
        )

    # ── SELF-TEST ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("SELF-TEST: bracket-failure / sub-intrinsic counts, 1-month, crisis vs control")
    print("=" * 110)
    hdr = (f"{'date':>12} {'role':>20} {'n':>5} {'bracket_fail(prod)':>20} "
           f"{'bracket_fail(probe)':>21} {'sub_intrinsic':>14}")
    print(hdr)
    print("-" * len(hdr))
    for date_str, info in summary.items():
        print(
            f"{date_str:>12} {info['role']:>20} {info['n']:>5} "
            f"{info['n_bracket_fail_prod']:>20} {info['n_bracket_fail_probe']:>21} "
            f"{info['n_sub_intrinsic']:>14}"
        )

    print()
    if "2020-03-18" in summary and "2020-01-15" in summary:
        crisis, control = summary["2020-03-18"], summary["2020-01-15"]
        crisis_rate_prod = crisis["n_bracket_fail_prod"] / crisis["n"]
        control_rate_prod = control["n_bracket_fail_prod"] / control["n"]
        crisis_rate_sub = crisis["n_sub_intrinsic"] / crisis["n"]
        control_rate_sub = control["n_sub_intrinsic"] / control["n"]

        print(f"  crisis bracket_fail(prod) rate  = {crisis_rate_prod:.1%}   "
              f"control = {control_rate_prod:.1%}")
        print(f"  crisis sub_intrinsic rate       = {crisis_rate_sub:.1%}   "
              f"control = {control_rate_sub:.1%}")
        print()

        fixed_by_wider_bracket = crisis["n_bracket_fail_prod"] - crisis["n_bracket_fail_probe"]

        if crisis_rate_sub > control_rate_sub and crisis["n_sub_intrinsic"] > 0:
            print("  Crisis 1-month shows MATERIALLY MORE sub-intrinsic model prices than the")
            print("  low-vol control. A put priced below its no-arbitrage floor is a PDE-SOLVE")
            print("  bug, not an inversion-routine bug -- consistent with the spatial")
            print("  under-resolution near the payoff kink found in debug_short_maturity.py")
            print("  (D3's 'third bug') and/or the domain-truncation leak found in")
            print("  debug_domain_truncation.py (D4), both of which are worse for this date's")
            print("  heavy left tail (G*=0.10). The fix belongs in the PDE solve, not brentq.")
        elif crisis["n_bracket_fail_prod"] > control["n_bracket_fail_prod"] and fixed_by_wider_bracket > 0:
            print("  Crisis 1-month shows MORE production bracket failures than control, and")
            print(f"  widening the bracket to [{_PROBE_VOL_LO},{_PROBE_VOL_HI}] recovers "
                  f"{fixed_by_wider_bracket} of them. The underlying")
            print("  model price is fine (not sub-intrinsic) but implies a vol the production")
            print(f"  bracket [{_PROD_VOL_LO},{_PROD_VOL_HI}] cannot reach -- consistent with the >65-87%")
            print("  market IVs observed on this date/tenor. The fix belongs in the INVERSION")
            print("  ROUTINE (widen the default hi bracket), not the PDE solve.")
        else:
            print("  Crisis and control show comparable rates of both failure modes at the")
            print("  1-month tenor -- the 'especially bad in the crisis period' symptom is not")
            print("  explained by either bracket failures or sub-intrinsic prices here; look")
            print("  elsewhere (e.g. genuinely large model-vs-market IV residual with a valid,")
            print("  in-bracket, super-intrinsic price).")


if __name__ == "__main__":
    main()
