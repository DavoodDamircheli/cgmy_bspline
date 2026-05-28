"""
Table 6: method comparison for PARAMS_II (Y=1.5), N=512, N_tau=100.

Four variants:
  A  mode='banded',           bandwidth=64
  B  mode='banded',           bandwidth=128
  C  mode='toeplitz'          (full Toeplitz, GMRES, no preconditioner)
  D  mode='toeplitz_precond'  (full Toeplitz, GMRES, banded LU preconditioner bw=64)

Metrics: price @ S0=100, error vs COS reference, CPU time, avg GMRES iters/step.
"""
import numpy as np

from cgmy_bspline.tests.test_parameters import PARAMS_II
from cgmy_bspline.solver_cn import solve_cn
from cgmy_bspline.projection import eval_solution
from cgmy_bspline.grid import make_grid
from cgmy_bspline.experiments.run_european_put import cos_price_put


VARIANTS = [
    dict(label='A', mode='banded',           bandwidth=64),
    dict(label='B', mode='banded',           bandwidth=128),
    dict(label='C', mode='toeplitz',         bandwidth=64),   # bw unused for C
    dict(label='D', mode='toeplitz_precond', bandwidth=64),
]

N      = 512
N_tau  = 100
p      = 3
params = PARAMS_II
S0     = params.K   # at-the-money: S0 = K = 100
x0     = np.log(S0)


def run_table6():
    cos_ref = cos_price_put(params, S0, n_cos=2048)

    sep = '=' * 88
    print(f"\n{sep}")
    print("  Table 6: Method comparison — PARAMS_II (Y=1.5),  N=512,  N_tau=100")
    print(sep)
    print(f"{'Var':>4}  {'Mode':<20}  {'bw':>4}  {'price@S0':>12}  "
          f"{'err@S0':>12}  {'CPU(s)':>8}  {'GMRES/step':>10}")
    print('-' * 88)

    prices = {}
    for v in VARIANTS:
        result = solve_cn(
            params, N=N, p=p, N_tau=N_tau,
            bandwidth=v['bandwidth'],
            mode=v['mode'],
        )
        g        = make_grid(params, N)
        price    = eval_solution(result['c_final'], g, p, np.array([x0]))[0]
        err      = abs(price - cos_ref)
        cpu      = result['cpu_time']
        iters    = result['gmres_iters']
        bw_str   = str(v['bandwidth']) if v['mode'] != 'toeplitz' else 'full'
        iter_str = f"{iters:.1f}" if iters is not None else 'N/A'

        print(f"{v['label']:>4}  {v['mode']:<20}  {bw_str:>4}  {price:>12.6f}  "
              f"{err:>12.2e}  {cpu:>8.2f}  {iter_str:>10}")

        prices[v['label']] = price

    print(f"\n  COS reference price: {cos_ref:.6f}")

    # Verification
    # C and D use the full Toeplitz operator (no truncation); they must agree with
    # the COS reference and with each other.
    toeplitz_spread = abs(prices['C'] - prices['D'])
    toeplitz_err_C  = abs(prices['C'] - cos_ref)
    toeplitz_err_D  = abs(prices['D'] - cos_ref)

    print()
    if toeplitz_spread < 1e-7:
        print(f"  [OK] C and D agree within GMRES tolerance (spread = {toeplitz_spread:.2e})")
    else:
        print(f"  [WARN] C and D disagree: spread = {toeplitz_spread:.2e}")

    if toeplitz_err_C < 1e-3 and toeplitz_err_D < 1e-3:
        print(f"  [OK] Toeplitz results agree with COS within 1e-3  "
              f"(err C = {toeplitz_err_C:.2e},  err D = {toeplitz_err_D:.2e})")
    else:
        print(f"  [FAIL] Toeplitz results diverge from COS:  "
              f"err C = {toeplitz_err_C:.2e},  err D = {toeplitz_err_D:.2e}")

    # Banded variants with small bandwidth exhibit truncation error for large N.
    # For N=512 and PARAMS_II (Y=1.5) the Lévy tail is heavy; bandwidth 64/128
    # covers only ±32/±64 of the 514 lags, producing O(1e-1) truncation error.
    banded_err_A = abs(prices['A'] - cos_ref)
    banded_err_B = abs(prices['B'] - cos_ref)
    print(f"  [NOTE] Banded truncation errors:  A = {banded_err_A:.2e},  B = {banded_err_B:.2e}  "
          f"(expected for bw << N=512)")


if __name__ == '__main__':
    run_table6()
