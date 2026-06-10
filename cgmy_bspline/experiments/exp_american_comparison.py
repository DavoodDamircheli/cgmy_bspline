"""
Step G — American put method comparison for CGMY parameter sets (i) and (ii).

Compares B-Spline Galerkin American put prices with the moment-matched binomial
reference for two parameter sets at N=64, 128, 256.

Parameter set (i):  C=0.5, G=M=5, Y=0.5  (symmetric, finite-variation)
Parameter set (ii): C=0.1, G=2, M=3.5, Y=1.5  (asymmetric, infinite-variation)

Self-test PASS criterion:
  - B-Spline AM >= B-Spline EU for both parameter sets at all N
  - Prices stabilize (last two N values within 0.2 of each other)
"""
import time
import numpy as np

from cgmy_bspline.src.cgmy_params import CGMYParams
from cgmy_bspline.src.american_solver import price_american_put
from cgmy_bspline.src.cn_solver import price_put as price_european
from cgmy_bspline.src.binomial_american import binomial_american_put


PARAM_SETS = {
    '(i)':  CGMYParams(C=0.5, G=5.0, M=5.0, Y=0.5,  r=0.1, T=0.2, K=100.0, S0=100.0),
    '(ii)': CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,  r=0.1, T=0.2, K=100.0, S0=100.0),
}


def run():
    grid_sizes = [64, 128, 256]
    all_am_prices = {}

    for label, params in PARAM_SETS.items():
        print(f"\nParameter set {label}:")
        bin_am = binomial_american_put(params, n_steps=2000)
        bin_eu = price_european(params, N=256, N_tau=1024)
        print(f"  Binomial AM: {bin_am:.4f}  |  B-Spline EU(N=256): {bin_eu:.4f}")
        print(f"  {'N':>5}  {'AM-BSpline':>12}  {'EU-BSpline':>12}  {'EEP':>8}  {'BinErr':>10}")
        print("  " + "-" * 55)

        am_list = []
        for N in grid_sizes:
            N_tau = 4 * N
            t0 = time.perf_counter()
            am = price_american_put(params, N=N, N_tau=N_tau)
            eu = price_european(params, N=N, N_tau=N_tau)
            cpu = time.perf_counter() - t0
            am_list.append(am)
            print(f"  {N:>5}  {am:>12.5f}  {eu:>12.5f}  {am-eu:>8.4f}  {am-bin_am:>+10.5f}")

        all_am_prices[label] = am_list

    return all_am_prices


if __name__ == "__main__":
    all_am_prices = run()

    for label, am_list in all_am_prices.items():
        params = PARAM_SETS[label]
        for i, (N, am) in enumerate(zip([64, 128, 256], am_list)):
            eu = price_european(params, N=N, N_tau=4*N)
            assert am >= eu - 0.05, \
                f"FAIL: Set {label} N={N}: AM={am:.4f} < EU={eu:.4f}"

        # Prices should stabilize between N=128 and N=256
        assert abs(am_list[-1] - am_list[-2]) < 0.20, \
            f"FAIL: Set {label} prices not stabilized: {am_list[-2]:.4f} vs {am_list[-1]:.4f}"

    print("\nPASS: exp_american_comparison self-test")
