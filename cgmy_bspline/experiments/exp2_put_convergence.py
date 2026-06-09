"""
Step 8 — European put convergence vs COS reference (Ludvigsson parameter sets).

Set (i):  C=0.5, G=5.0, M=5.0, Y=0.5, r=0.1, T=0.2, K=100, S0=100
Set (ii): C=0.1, G=2.0, M=3.5, Y=1.5, r=0.1, T=0.2, K=100, S0=100

Reference: COS pricer with N_cos=2048.
Errors are measured on a FIXED evaluation grid (moneyness 0.80–1.20) to
avoid boundary-condition artefacts at x_min / x_max.
dt = T / (4*N).
"""
import time
import os
import numpy as np

from cgmy_bspline.parameters import CGMYParams
from cgmy_bspline.solver_cn import solve_cn
from cgmy_bspline.grid import make_grid
from cgmy_bspline.projection import eval_solution
from cgmy_bspline.src.cos_pricer import cos_put_price
from cgmy_bspline.src.cgmy_params import CGMYParams as SrcParams

p  = 3
Ns = [64, 128, 256, 512]

# Fixed evaluation grid: 80 log-moneyness points, K*[0.80, 1.20]
N_EVAL  = 80
K_EVAL  = 100.0

PARAMS_I  = CGMYParams(C=0.5, G=5.0, M=5.0, Y=0.5, r=0.1, K=K_EVAL, T=0.2)
PARAMS_II = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=0.1, K=K_EVAL, T=0.2)


def _cos_grid(params_old, S0_arr, N_cos=2048):
    """COS prices at each S0 in S0_arr."""
    out = np.empty(len(S0_arr))
    for i, S0 in enumerate(S0_arr):
        sp = SrcParams(C=params_old.C, G=params_old.G, M=params_old.M,
                       Y=params_old.Y, r=params_old.r, T=params_old.T,
                       K=params_old.K, S0=S0)
        out[i] = cos_put_price(sp, N_cos=N_cos)
    return out


def run_table(params_old, label, table_lines):
    # Fixed evaluation spots: S0 in [0.80*K, 1.20*K]
    S0_eval = params_old.K * np.linspace(0.80, 1.20, N_EVAL)
    x_eval  = np.log(S0_eval)

    print(f"  Computing COS reference ({N_EVAL} points)...", flush=True)
    V_cos = _cos_grid(params_old, S0_eval, N_cos=2048)

    header = (f"\n=== European Put Convergence, {label}, Y={params_old.Y}, p={p} ===")
    dash = "=" * (len(header) - 1)
    table_lines.append(dash)
    table_lines.append(header)
    table_lines.append(dash)
    col = (f"  {'N':>5}  {'dt':>10}  {'||e||_L2':>12}  {'Order':>8}  "
           f"{'||e||_Linf':>12}  {'CPU(s)':>8}")
    table_lines.append(col)
    table_lines.append("  " + "-" * 68)

    prev_err_L2 = None
    for N in Ns:
        N_tau = 4 * N
        t0 = time.perf_counter()
        result = solve_cn(params_old, N=N, p=p, N_tau=N_tau, bandwidth=2 * N)
        cpu = time.perf_counter() - t0
        g = make_grid(params_old, N)
        V_h = eval_solution(result['c_final'], g, p, x_eval)

        diff     = V_h - V_cos
        dx       = x_eval[1] - x_eval[0]
        err_L2   = np.sqrt(dx * np.sum(diff ** 2))
        err_Linf = np.max(np.abs(diff))

        if prev_err_L2 is not None and prev_err_L2 > 0 and err_L2 > 0:
            order = np.log2(prev_err_L2 / err_L2)
        else:
            order = float('nan')
        prev_err_L2 = err_L2

        dt = params_old.T / N_tau
        ord_str = f"{order:8.3f}" if not np.isnan(order) else "     ---"
        row = (f"  {N:>5}  {dt:>10.5f}  {err_L2:>12.3e}  {ord_str}  "
               f"{err_Linf:>12.3e}  {cpu:>8.3f}")
        table_lines.append(row)
        print(row, flush=True)

    # ATM reference
    atm_price = cos_put_price(SrcParams(C=params_old.C, G=params_old.G,
                                         M=params_old.M, Y=params_old.Y,
                                         r=params_old.r, T=params_old.T,
                                         K=params_old.K, S0=params_old.K),
                               N_cos=2048)
    table_lines.append(f"\n  COS ATM reference (S0=K=100): {atm_price:.6f}")


def main():
    os.makedirs("cgmy_bspline/results", exist_ok=True)
    lines = []

    print("=== European Put Convergence, Set (i) Y=0.5, p=3 ===")
    run_table(PARAMS_I,  "Set (i)",  lines)
    print()
    print("=== European Put Convergence, Set (ii) Y=1.5, p=3 ===")
    run_table(PARAMS_II, "Set (ii)", lines)

    out  = "\n".join(lines)
    path = "cgmy_bspline/results/table_put_convergence.txt"
    with open(path, "w") as f:
        f.write(out)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
