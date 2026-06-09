"""
Step 10 — IV surface and Greeks for Set (ii): Y=1.5.

Generates:
  • ATM IV table across maturities (printed + csv)
  • Greeks (Delta, Gamma) at ATM (printed)
  • iv_surface.csv  — full K × T IV matrix
  • iv_surface.png  — 3-D surface plot
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgmy_bspline.parameters import CGMYParams
from cgmy_bspline.solver_cn import solve_cn
from cgmy_bspline.grid import make_grid
from cgmy_bspline.projection import eval_solution
from cgmy_bspline.greeks import iv_surface, compute_greeks, bs_implied_vol

PARAMS = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=0.1, K=100.0, T=0.2)
K_REF = 100.0
N = 256
p = 3

STRIKES     = np.array([85, 90, 95, 100, 105, 110, 115], dtype=float)
MATURITIES  = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def main():
    os.makedirs("cgmy_bspline/results", exist_ok=True)

    # ── IV surface ────────────────────────────────────────────────────────────
    print("Computing IV surface (may take a minute)...", flush=True)
    t0 = time.perf_counter()
    iv_mat = iv_surface(
        PARAMS, N=N, p=p,
        strikes=STRIKES,
        maturities=MATURITIES,
        N_tau_per_unit=4 * N,
        bandwidth=2 * N,
        mode='banded',
    )
    cpu = time.perf_counter() - t0
    print(f"  Done in {cpu:.1f}s")

    # ── Print ATM IV table ────────────────────────────────────────────────────
    atm_idx = np.argmin(np.abs(STRIKES - K_REF))
    print("\n=== ATM Implied Volatility (K=100) vs Maturity ===")
    print(f"  {'T':>6}  {'IV (ATM)':>12}")
    print("  " + "-" * 22)
    atm_lines = ["=== ATM Implied Volatility (K=100), Set (ii) Y=1.5 ===",
                 f"  {'T':>6}  {'IV (ATM)':>12}",
                 "  " + "-" * 22]
    for i, T_i in enumerate(MATURITIES):
        iv_val = iv_mat[i, atm_idx]
        row = f"  {T_i:>6.2f}  {iv_val:>12.4f}"
        print(row)
        atm_lines.append(row)

    # ── Print IV smile at T=0.2 ───────────────────────────────────────────────
    T_idx = MATURITIES.index(0.2)
    print(f"\n=== IV Smile at T=0.2 ===")
    print(f"  {'K':>6}  {'Moneyness':>10}  {'IV':>10}")
    print("  " + "-" * 30)
    smile_lines = ["\n=== IV Smile at T=0.2, Set (ii) Y=1.5 ===",
                   f"  {'K':>6}  {'Moneyness':>10}  {'IV':>10}",
                   "  " + "-" * 30]
    for j, K_j in enumerate(STRIKES):
        iv_val = iv_mat[T_idx, j]
        mon = K_j / K_REF
        row = f"  {K_j:>6.0f}  {mon:>10.4f}  {iv_val:>10.4f}"
        print(row)
        smile_lines.append(row)

    # ── Greeks at ATM ─────────────────────────────────────────────────────────
    print("\n=== Greeks at ATM (S0=K=100) vs Maturity ===")
    print(f"  {'T':>6}  {'Delta':>10}  {'Gamma':>10}  {'Price':>10}")
    print("  " + "-" * 44)
    greeks_lines = ["\n=== Greeks at ATM, Set (ii) Y=1.5 ===",
                    f"  {'T':>6}  {'Delta':>10}  {'Gamma':>10}  {'Price':>10}",
                    "  " + "-" * 44]

    x0 = np.log(K_REF)
    for T_i in MATURITIES:
        N_tau = max(1, round(4 * N * T_i))
        params_i = CGMYParams(
            C=PARAMS.C, G=PARAMS.G, M=PARAMS.M, Y=PARAMS.Y,
            r=PARAMS.r, K=K_REF, T=T_i,
        )
        result = solve_cn(params_i, N=N, p=p, N_tau=N_tau,
                          bandwidth=2 * N, mode='banded')
        g = make_grid(params_i, N)
        price = eval_solution(result['c_final'], g, p, np.array([x0]))[0]
        delta, gamma = compute_greeks(result['c_final'], g, p, np.array([x0]))
        row = (f"  {T_i:>6.2f}  {float(delta[0]):>10.4f}  "
               f"{float(gamma[0]):>10.4f}  {float(price):>10.4f}")
        print(row)
        greeks_lines.append(row)

    # ── Save full IV surface CSV ───────────────────────────────────────────────
    csv_path = "cgmy_bspline/results/iv_surface.csv"
    header = "T," + ",".join(f"K={K_j:.0f}" for K_j in STRIKES)
    rows = []
    for i, T_i in enumerate(MATURITIES):
        row = f"{T_i:.2f}," + ",".join(f"{iv_mat[i, j]:.6f}" for j in range(len(STRIKES)))
        rows.append(row)
    with open(csv_path, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows))
    print(f"\nSaved: {csv_path}")

    # ── Save full ATM + smile + Greeks table ─────────────────────────────────
    txt_path = "cgmy_bspline/results/table_iv_surface.txt"
    all_lines = atm_lines + smile_lines + greeks_lines
    with open(txt_path, "w") as f:
        f.write("\n".join(all_lines))
    print(f"Saved: {txt_path}")

    # ── 3-D surface plot ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    K_grid, T_grid = np.meshgrid(STRIKES, MATURITIES)
    surf = ax.plot_surface(K_grid, T_grid, iv_mat * 100.0,
                           cmap='viridis', edgecolor='none', alpha=0.85)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T")
    ax.set_zlabel("Implied Vol (%)")
    ax.set_title("IV Surface — CGMY Y=1.5 (B-spline Galerkin, p=3)")
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08, label="IV (%)")
    png_path = "cgmy_bspline/results/iv_surface.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
