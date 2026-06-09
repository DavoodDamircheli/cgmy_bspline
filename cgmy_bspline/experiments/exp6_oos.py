"""
Step 12 — Out-of-sample test: 20-day rolling calibration with drifting parameters.

Protocol
--------
Day 0: one-shot DE + L-BFGS-B calibration using synthetic surface.
Days 1-19: L-BFGS-B warm-start from previous day's result (fast rolling update).
OOS: each day's calibrated model prices the NEXT day's surface; report RMSE.

True parameters drift linearly over 20 days:
  Y: 1.5 → 1.3, G: 2.0 → 2.5, M: 3.5 → 4.0, C: 0.10 → 0.12
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

from cgmy_bspline.src.cgmy_params import CGMYParams as SrcParams
from cgmy_bspline.src.cos_pricer import cos_put_price
from cgmy_bspline.greeks import bs_implied_vol
from cgmy_bspline.calibration import CGMYCalibrator

# ── Experiment settings ───────────────────────────────────────────────────────
N_DAYS     = 20
R          = 0.1
K_REF      = 100.0
STRIKES    = np.array([90.0, 95.0, 100.0, 105.0, 110.0], dtype=float)
MATURITIES = [0.2, 0.3, 0.5]
N_CAL      = 32
BW_CAL     = 16

# Phase-1 DE (day 0 only) — kept small
DE_KW = dict(popsize=4, maxiter=30, seed=0, tol=1e-3, polish=False)
BOUNDS = [(0.01, 5.0), (0.5, 20.0), (1.01, 20.0), (0.01, 1.99)]


def true_params(day):
    t = day / (N_DAYS - 1)
    return dict(C=0.10 + 0.02 * t, G=2.00 + 0.50 * t,
                M=3.50 + 0.50 * t, Y=1.50 - 0.20 * t)


def make_full_surface(params_dict):
    mat = np.full((len(MATURITIES), len(STRIKES)), np.nan)
    for i, T_i in enumerate(MATURITIES):
        for j, K_j in enumerate(STRIKES):
            p = SrcParams(
                C=params_dict['C'], G=params_dict['G'],
                M=params_dict['M'], Y=params_dict['Y'],
                r=R, T=T_i, K=K_REF, S0=K_j,
            )
            price = cos_put_price(p, N_cos=2048)
            iv = bs_implied_vol(price, S=K_j, K=K_REF, T=T_i, r=R)
            mat[i, j] = iv
    return mat


def _make_calibrator(iv_surface):
    return CGMYCalibrator(
        market_ivs=iv_surface,
        strikes=STRIKES,
        maturities=MATURITIES,
        r=R, S0=K_REF, K_ref=K_REF,
        N=N_CAL, bandwidth=BW_CAL,
    )


def calibrate_day0(cal):
    """Full DE + L-BFGS-B for day 0."""
    return cal.calibrate(de_kwargs=DE_KW)


def calibrate_warmstart(cal, x0):
    """Warm-start L-BFGS-B from previous day's parameters."""
    cal._n_evals = 0
    t0 = time.time()
    lb = minimize(
        cal.objective, x0,
        method='L-BFGS-B', bounds=BOUNDS,
        options=dict(ftol=1e-10, gtol=1e-7, maxiter=200),
    )
    theta_star = np.array(lb.x, dtype=float)
    C, G, M, Y = theta_star

    prices = cal._price_puts(C, G, M, Y)
    sq = []
    for i, T_i in enumerate(cal.maturities):
        for j, K_j in enumerate(cal.strikes):
            iv_m = bs_implied_vol(prices[i, j], S=K_j, K=cal.K_ref,
                                  T=T_i, r=cal.r)
            if not np.isnan(iv_m):
                sq.append((iv_m - cal.market_ivs[i, j]) ** 2)
    rmse = float(np.sqrt(np.mean(sq))) if sq else float('nan')
    return {
        'theta_star': theta_star,
        'C': float(C), 'G': float(G), 'M': float(M), 'Y': float(Y),
        'rmse_vol': rmse,
        'n_func_evals': cal._n_evals,
        'cpu_time': time.time() - t0,
    }


def oos_rmse(params_dict, surface_next):
    """Price next-day surface using today's calibrated params; return RMSE."""
    from cgmy_bspline.parameters import CGMYParams as OldParams
    from cgmy_bspline.solver_cn import solve_cn
    from cgmy_bspline.grid import make_grid
    from cgmy_bspline.projection import eval_solution

    sq = []
    for i, T_i in enumerate(MATURITIES):
        N_tau = max(10, round(200 * T_i))
        old_p = OldParams(C=params_dict['C'], G=params_dict['G'],
                          M=params_dict['M'], Y=params_dict['Y'],
                          r=R, K=K_REF, T=T_i)
        res = solve_cn(old_p, N=64, p=3, N_tau=N_tau, bandwidth=64)
        from cgmy_bspline.grid import make_grid
        g = make_grid(old_p, 64)
        from cgmy_bspline.projection import eval_solution
        x_ev = np.log(STRIKES)
        prices = eval_solution(res['c_final'], g, 3, x_ev)
        for j, (K_j, pr) in enumerate(zip(STRIKES, prices)):
            iv_m = bs_implied_vol(pr, S=K_j, K=K_REF, T=T_i, r=R)
            iv_true = surface_next[i, j]
            if not np.isnan(iv_m) and not np.isnan(iv_true):
                sq.append((iv_m - iv_true) ** 2)
    return float(np.sqrt(np.mean(sq))) if sq else float('nan')


def main():
    os.makedirs("cgmy_bspline/results", exist_ok=True)

    print(f"Generating {N_DAYS+1} synthetic IV surfaces...", flush=True)
    surfaces = [make_full_surface(true_params(d)) for d in range(N_DAYS + 1)]
    print("Done.")

    C_hist, G_hist, M_hist, Y_hist = [], [], [], []
    oos_rmse_hist, is_rmse_hist = [], []

    lines = [
        "=== Out-of-Sample Test: 20-Day Rolling Calibration ===",
        f"  N_cal={N_CAL}, bandwidth={BW_CAL}, warm-start L-BFGS-B (day 0: DE+LBFGS)",
        "",
        f"  {'Day':>4}  {'C_hat':>7}  {'G_hat':>7}  {'M_hat':>7}  {'Y_hat':>7}  "
        f"{'IS_RMSE':>9}  {'OOS_RMSE':>9}  {'CPU(s)':>7}",
        "  " + "-" * 72,
    ]
    print(lines[-2])
    print("  " + "-" * 72)

    theta_prev = None
    for day in range(N_DAYS):
        cal = _make_calibrator(surfaces[day])
        t0 = time.perf_counter()

        if day == 0:
            result = calibrate_day0(cal)
        else:
            result = calibrate_warmstart(cal, theta_prev)

        cpu = time.perf_counter() - t0
        theta_prev = result['theta_star']

        C_h = result['C']; G_h = result['G']
        M_h = result['M']; Y_h = result['Y']
        is_r = result['rmse_vol']

        C_hist.append(C_h); G_hist.append(G_h)
        M_hist.append(M_h); Y_hist.append(Y_h)
        is_rmse_hist.append(is_r)

        oos_r = oos_rmse({'C': C_h, 'G': G_h, 'M': M_h, 'Y': Y_h}, surfaces[day + 1])
        oos_rmse_hist.append(oos_r)

        row = (f"  {day+1:>4}  {C_h:>7.3f}  {G_h:>7.3f}  {M_h:>7.3f}  {Y_h:>7.3f}  "
               f"{is_r:>9.5f}  {oos_r:>9.5f}  {cpu:>7.1f}")
        lines.append(row)
        print(row, flush=True)

    mean_is  = float(np.nanmean(is_rmse_hist))
    mean_oos = float(np.nanmean(oos_rmse_hist))
    for l in [
        "  " + "-" * 72,
        f"  Mean in-sample  RMSE: {mean_is:.5f}",
        f"  Mean OOS        RMSE: {mean_oos:.5f}",
    ]:
        print(l); lines.append(l)

    path_tbl = "cgmy_bspline/results/table_oos.txt"
    with open(path_tbl, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {path_tbl}")

    # ── Parameter stability plot ──────────────────────────────────────────────
    days = np.arange(1, N_DAYS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, vals, key, lo, hi in zip(
        axes.flat,
        [C_hist, G_hist, M_hist, Y_hist],
        ['C', 'G', 'M', 'Y'],
        [0.08, 1.5, 3.0, 1.2],
        [0.15, 3.5, 5.0, 1.6],
    ):
        true_v = [true_params(d)[key] for d in range(N_DAYS)]
        ax.plot(days, vals, 'b-o', ms=4, label='calibrated')
        ax.plot(days, true_v, 'r--', lw=1.5, label='true')
        ax.set_title(f"Parameter {key}")
        ax.set_xlabel("Day")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Rolling calibration: parameter stability (20-day window)")
    plt.tight_layout()
    png_path = "cgmy_bspline/results/param_stability.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
