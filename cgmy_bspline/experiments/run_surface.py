"""
Implied-volatility surface and Greeks for PARAMS_II (Y=1.5).

Outputs
-------
  - ATM IV summary table (printed)
  - iv_surface_params_ii.csv  (maturity, strike, iv)
  - Delta / Gamma table for selected spot values (printed)
"""
import numpy as np

from cgmy_bspline.tests.test_parameters import PARAMS_II
from cgmy_bspline.greeks import compute_greeks, iv_surface
from cgmy_bspline.solver_cn import solve_cn
from cgmy_bspline.projection import eval_solution
from cgmy_bspline.grid import make_grid


# ── Parameters ────────────────────────────────────────────────────────────────

params     = PARAMS_II
strikes    = np.linspace(80, 120, 40)
maturities = [1 / 12, 3 / 12, 6 / 12, 1.0]
N, p       = 256, 3

# ── 1. IV surface ─────────────────────────────────────────────────────────────

print("Computing IV surface (4 maturities × 40 strikes)...")
iv_mat = iv_surface(params, N=N, p=p, strikes=strikes, maturities=maturities,
                    N_tau_per_unit=200, bandwidth=2 * N)

# ── 2. ATM summary table ──────────────────────────────────────────────────────

atm_idx = int(np.argmin(np.abs(strikes - params.K)))
atm_K   = strikes[atm_idx]

sep = '=' * 46
print(f"\n{sep}")
print(f"  ATM IV summary — PARAMS_II (Y=1.5), K={params.K}")
print(f"  Nearest-ATM strike in grid: {atm_K:.2f}")
print(sep)
print(f"{'Maturity':>10}  {'ATM IV':>10}  {'ATM IV (%)':>12}")
print('-' * 46)
for i, T_i in enumerate(maturities):
    iv_atm = iv_mat[i, atm_idx]
    pct    = f"{iv_atm * 100:.2f}%" if not np.isnan(iv_atm) else "  N/A"
    print(f"{T_i:>10.4f}  {iv_atm:>10.4f}  {pct:>12}")

# ── 3. Save CSV ───────────────────────────────────────────────────────────────

csv_path = "iv_surface_params_ii.csv"
with open(csv_path, "w") as fh:
    fh.write("maturity,strike,iv\n")
    for i, T_i in enumerate(maturities):
        for j, K_j in enumerate(strikes):
            fh.write(f"{T_i:.6f},{K_j:.4f},{iv_mat[i, j]:.6f}\n")
print(f"\nIV surface saved to {csv_path}")

# ── 4. Greeks ─────────────────────────────────────────────────────────────────

print("\nComputing Delta and Gamma for PARAMS_II, T=0.2, N=256...")
N_tau_greeks = 100
result = solve_cn(params, N=N, p=p, N_tau=N_tau_greeks, bandwidth=2 * N)
g      = make_grid(params, N)

S_eval  = np.linspace(80, 120, 200)
x_eval  = np.log(S_eval)
Delta, Gamma = compute_greeks(result['c_final'], g, p, x_eval)

S_print = [85, 90, 95, 100, 105, 110, 115]

sep2 = '=' * 50
print(f"\n{sep2}")
print(f"  Greeks — PARAMS_II (T={params.T}, Y={params.Y})")
print(sep2)
print(f"{'S':>8}  {'Delta':>12}  {'Gamma':>12}")
print('-' * 50)
for S_target in S_print:
    idx = int(np.argmin(np.abs(S_eval - S_target)))
    print(f"{S_eval[idx]:>8.2f}  {Delta[idx]:>12.6f}  {Gamma[idx]:>12.6f}")
