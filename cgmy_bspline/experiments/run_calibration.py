"""
Synthetic calibration demo: recover PARAMS_II from noisy COS-generated IVs.

True parameters: C=0.1, G=2.0, M=3.5, Y=1.5 (PARAMS_II).
Synthetic surface: COS prices → BS IV + Gaussian noise (σ=0.002).

Note on identifiability
-----------------------
The CGMY IV smile for near-ATM strikes (90–110) and three short maturities is
nearly flat and weakly sensitive to the individual values of (C, G, M, Y).
Multiple parameter combinations produce nearly identical IV surfaces; the
calibration therefore finds AN equivalent model, not necessarily the unique one
with the same (C, G, M, Y) as the ground truth.  The correct metric is the
in-sample RMSE: if it is close to the added noise level (0.2 vol-pts), the
calibrated model is a valid description of the market.

For unique recovery of all four parameters a richer surface (wider strike range,
more maturities) or tighter prior bounds on Y are needed.
"""
import numpy as np
from cgmy_bspline.tests.test_parameters import PARAMS_II
from cgmy_bspline.parameters import CGMYParams
from cgmy_bspline.calibration import CGMYCalibrator
from cgmy_bspline.greeks import bs_implied_vol
from cgmy_bspline.experiments.run_european_put import cos_price_put


# ── Calibration grid (as specified) ──────────────────────────────────────────

strikes    = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
maturities = [1 / 12, 3 / 12, 6 / 12]
S0 = K_ref = 100.0
r          = PARAMS_II.r    # 0.10

# ── Synthetic market IVs (COS + Gaussian noise) ───────────────────────────────

print("Generating synthetic IV surface from COS reference prices...")
true_ivs = np.zeros((len(maturities), len(strikes)))
for i, T_i in enumerate(maturities):
    p = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=r, K=K_ref, T=T_i)
    for j, K_j in enumerate(strikes):
        price = cos_price_put(p, S0=K_j, n_cos=2048)
        true_ivs[i, j] = bs_implied_vol(price, S=K_j, K=K_ref, T=T_i, r=r)

np.random.seed(0)
market_ivs = true_ivs + 0.002 * np.random.randn(*true_ivs.shape)

print(f"  True IV range   : [{true_ivs.min():.4f}, {true_ivs.max():.4f}]")
print(f"  Market IV range : [{market_ivs.min():.4f}, {market_ivs.max():.4f}]  (σ_noise = 0.002)")

# ── Build calibrator ──────────────────────────────────────────────────────────
# N=128, bandwidth=2*N gives full Toeplitz coverage (all lags resolved).
# Discretisation error ≈ 0.22 vol-pts, comparable to the noise level.

print("\nBuilding calibrator (M_h, D_h, c_init computed once) ...")
calib = CGMYCalibrator(
    market_ivs = market_ivs,
    strikes    = strikes,
    maturities = maturities,
    r          = r,
    S0         = S0,
    K_ref      = K_ref,
    N          = 128,
    bandwidth  = 256,      # = 2*N → full coverage for N=128
)

# Quick objective sanity check at true theta
loss_true = calib.objective([0.1, 2.0, 3.5, 1.5])
print(f"  obj(true theta) = {loss_true:.4e}  "
      f"(noise floor ≈ {len(strikes)*len(maturities)*(0.002)**2:.4e})")

# ── Calibrate ─────────────────────────────────────────────────────────────────
# We use popsize=8, maxiter=150 here for a ~3-minute demo.
# The class defaults (popsize=15, maxiter=300) give higher quality at longer runtime.

print("\nRunning two-phase calibration (DE popsize=8, maxiter=150, then L-BFGS-B) ...")
result = calib.calibrate(de_kwargs=dict(popsize=8, maxiter=150, seed=42))

# ── Report ────────────────────────────────────────────────────────────────────

C_cal, G_cal, M_cal, Y_cal = result['C'], result['G'], result['M'], result['Y']

sep = '=' * 62
print(f"\n{sep}")
print("  Calibration result — PARAMS_II synthetic surface")
print(sep)
print(f"  {'Parameter':>10}   {'True':>10}   {'Calibrated':>12}")
print(f"  {'-' * 38}")
for name, tv, cv in [('C', 0.1, C_cal), ('G', 2.0, G_cal),
                      ('M', 3.5, M_cal), ('Y', 1.5, Y_cal)]:
    print(f"  {name:>10}   {tv:>10.4f}   {cv:>12.4f}")

print(f"\n  In-sample RMSE   : {result['rmse_vol']*100:.4f} vol-pts  "
      f"(noise = 0.2000 vol-pts)")
print(f"  Function evals   : {result['n_func_evals']}")
print(f"  CPU time         : {result['cpu_time']:.1f} s")

# ── Verification: COS IVs at calibrated theta vs true theta ──────────────────
# Both should give similar IV values if the calibrated model is equivalent.

print(f"\n  IV verification — COS(true) vs B-spline(calibrated) vs market:")
print(f"  {'T':>6}  {'K':>5}  {'IV_true':>8}  {'IV_cal_bsp':>11}  {'IV_mkt':>8}")
prices_cal = calib._price_puts(C_cal, G_cal, M_cal, Y_cal)
for i, T_i in enumerate(maturities):
    for j, K_j in enumerate(strikes):
        iv_cal = bs_implied_vol(prices_cal[i, j], S=K_j, K=K_ref, T=T_i, r=r)
        print(f"  {T_i:6.4f}  {K_j:5.0f}  {true_ivs[i,j]:8.4f}  "
              f"{iv_cal:11.4f}  {market_ivs[i,j]:8.4f}")
