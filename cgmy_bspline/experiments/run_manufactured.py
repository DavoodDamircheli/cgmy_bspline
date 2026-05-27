"""
Manufactured-solution convergence study for the Crank-Nicolson B-spline solver.

V_exact(x, tau) = exp(-r*tau) * (1 + x^2)^{-nu},   nu = 1.5

Expected spatial convergence rate: p - Y/2  (B-spline Galerkin, fractional order Y).
  Y=0.5 (PARAMS_I):  rate → 2.75
  Y=1.5 (PARAMS_II): rate → 2.25

Uses bandwidth = 2*N so that all fractional lags within the domain are captured
without aliasing.  N_tau = N so both h and dt refine together.
"""
import numpy as np
from cgmy_bspline.tests.test_parameters import PARAMS_I, PARAMS_II
from cgmy_bspline.solver_cn import solve_cn

p = 3
nu = 1.5
Ns = [64, 128, 256, 512]


def run_table(label, params):
    print(f"\n{'=' * 65}")
    print(f"  {label}   (Y={params.Y})")
    print(f"{'=' * 65}")
    print(f"{'N':>6}  {'h':>10}  {'dt':>10}  {'L2_error':>12}  {'rate':>8}  {'CPU(s)':>8}")
    print("-" * 65)

    prev_err = None
    for N in Ns:
        # bandwidth = 2*N: captures all lags without IFFT aliasing
        result = solve_cn(params, N=N, p=p, N_tau=N,
                          bandwidth=2 * N, manufactured=True, nu=nu)
        h = (result['x_nodes'][-1] - result['x_nodes'][0]) / N
        dt = params.T / N
        err = result['error_L2']
        cpu = result['cpu_time']

        if prev_err is not None and prev_err > 0 and err > 0:
            rate = np.log2(prev_err / err)
        else:
            rate = float('nan')

        print(f"{N:>6}  {h:>10.4e}  {dt:>10.4e}  {err:>12.6e}  {rate:>8.3f}  {cpu:>8.2f}")
        prev_err = err

    expected = p - params.Y / 2
    print(f"\n  Expected asymptotic rate: {expected:.2f}")


if __name__ == '__main__':
    run_table("PARAMS_I ", PARAMS_I)
    run_table("PARAMS_II", PARAMS_II)
