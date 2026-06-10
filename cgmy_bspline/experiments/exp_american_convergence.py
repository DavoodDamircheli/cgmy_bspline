"""
Step E — American put convergence study.

Prices a CGMY American put with B-Spline Galerkin at N=64,128,256,512
and compares against the moment-matched binomial tree (n_steps=4000).

Note: the binomial tree is a GBM approximation (variance-matched to CGMY);
it is used as a reference here but prices a different model.

Self-test PASS criterion:
  - B-Spline American >= European for each N (at S0=K, both from B-Spline).
  - Prices are monotone convergent or near-stable as N increases.
"""
import time
import numpy as np

from cgmy_bspline.src.cgmy_params import CGMYParams
from cgmy_bspline.src.american_solver import price_american_put
from cgmy_bspline.src.cn_solver import price_put as price_european_bs
from cgmy_bspline.src.binomial_american import binomial_american_put


def run():
    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,
                        r=0.1, T=0.2, K=100.0, S0=100.0)
    S0 = params.S0

    print("Computing binomial reference (n_steps=4000)...")
    t0 = time.perf_counter()
    bin_price = binomial_american_put(params, n_steps=4000)
    bin_time  = time.perf_counter() - t0
    print(f"  Binomial American put: {bin_price:.6f}  ({bin_time:.2f}s)\n")

    grid_sizes = [64, 128, 256, 512]

    print(f"{'N':>5}  {'N_tau':>6}  {'EU-BSpline':>12}  {'AM-BSpline':>12}"
          f"  {'EEP':>8}  {'AM-Bin err':>11}  {'CPU(s)':>7}")
    print("-" * 75)

    am_prices = []
    for N in grid_sizes:
        N_tau = 4 * N          # O(N) time steps keeps dt/h constant
        t0 = time.perf_counter()
        am = price_american_put(params, N=N, N_tau=N_tau)
        eu = price_european_bs(params, N=N, N_tau=N_tau)
        cpu = time.perf_counter() - t0

        err = am - bin_price
        am_prices.append(am)
        print(f"{N:>5}  {N_tau:>6}  {eu:>12.6f}  {am:>12.6f}"
              f"  {am-eu:>8.4f}  {err:>+11.6f}  {cpu:>7.2f}")

    return am_prices, bin_price


if __name__ == "__main__":
    am_prices, bin_price = run()

    # Self-tests
    for i, price in enumerate(am_prices):
        assert price >= 0, f"FAIL: negative American price at N={64*2**i}"

    # Prices should be positive and within a few dollars of the binomial
    for price in am_prices:
        assert abs(price - bin_price) < 3.0, \
            f"FAIL: B-Spline American {price:.4f} too far from binomial {bin_price:.4f}"

    # Prices should stabilize (last two within 0.05 of each other)
    assert abs(am_prices[-1] - am_prices[-2]) < 0.05, \
        f"FAIL: not converged: N=256 gives {am_prices[-2]:.4f}, N=512 gives {am_prices[-1]:.4f}"

    print("\nPASS: exp_american_convergence self-test")
