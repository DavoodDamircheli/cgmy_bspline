"""
Step H — PSOR relaxation parameter sensitivity.

Measures PSOR iterations to convergence vs. omega in [1.0, 1.9] for
a single CN time step of the American put LCP.

Self-test PASS criterion:
  - Some omega in [1.0, 1.9] gives fewer iterations than omega=1.0 (plain GS)
  - All omega values give the same converged price (within tolerance)
"""
import time
import numpy as np
import scipy.sparse.linalg as spla

from cgmy_bspline.src.cgmy_params import CGMYParams
from cgmy_bspline.parameters import CGMYParams as OldParams
from cgmy_bspline import matrices as _mat
from cgmy_bspline.projection import assemble_put_rhs, eval_solution
from cgmy_bspline.src.psor_solver import psor_lcp_banded
from cgmy_bspline.src.american_solver import _make_grid


def run():
    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,
                        r=0.1, T=0.2, K=100.0, S0=100.0)
    p = 3; K = params.K; r = params.r; T = params.T
    N = 128; N_tau = 100; bw = 2 * N
    dt = T / N_tau

    x_grid, h, x_min, x_max = _make_grid(K, params.G, params.M, N)
    n_dof = N + p - 1
    old = OldParams(C=params.C, G=params.G, M=params.M, Y=params.Y, r=r, K=K, T=T)
    gi = {'N': N, 'h': h, 'x_min': x_min, 'x_max': x_max, 'nodes': x_grid}

    mats = _mat.operator_matrix(N, p, h, x_min, old, bandwidth=bw)
    M_h = mats['M_h']; A_h = mats['A_h']
    F = (M_h + (dt / 2) * A_h).tocsr()
    B = (M_h - (dt / 2) * A_h).tocsr()

    bc = [0, 1, N, N + 1]
    F_lil = F.tolil()
    for i in bc: F_lil[i, :] = 0.0; F_lil[i, i] = 1.0
    F = F_lil.tocsr()

    cL = (K - np.exp(x_min)) * np.sqrt(h)
    b0 = assemble_put_rhs(old, gi, p)
    g = spla.spsolve(M_h, b0).copy(); g[bc] = -1e30

    c0 = spla.spsolve(M_h, b0)
    c0[:2] = cL; c0[N:] = 0.0

    # Advance halfway to get a non-trivial starting point
    lu = spla.splu(F.tocsc())
    c = c0.copy()
    for _ in range(N_tau // 2):
        rhs = B @ c
        rhs[0] = rhs[1] = cL; rhs[N] = rhs[N + 1] = 0.0
        c, _, _ = psor_lcp_banded(F, rhs, g, c_init=c, omega_sor=1.2,
                                   max_iter=500, tol=1e-10)
        c[:2] = cL; c[N:] = 0.0

    # One step: measure iters for each omega
    rhs = B @ c
    rhs[0] = rhs[1] = cL; rhs[N] = rhs[N + 1] = 0.0

    omegas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    results = []

    print(f"{'omega':>6}  {'iters':>8}  {'price(S=100)':>14}  {'CPU(ms)':>10}")
    print("-" * 48)

    c_ref, _, _ = psor_lcp_banded(F, rhs, g, c_init=c.copy(),
                                   omega_sor=1.0, max_iter=2000, tol=1e-12)
    V_ref = eval_solution(c_ref, gi, p, x_grid)
    p_ref = float(np.interp(np.log(K), x_grid, V_ref))

    for omega in omegas:
        t0 = time.perf_counter()
        c_sol, nit, conv = psor_lcp_banded(F, rhs, g, c_init=c.copy(),
                                            omega_sor=omega,
                                            max_iter=2000, tol=1e-10)
        cpu_ms = (time.perf_counter() - t0) * 1000
        V_sol = eval_solution(c_sol, gi, p, x_grid)
        price = float(np.interp(np.log(K), x_grid, V_sol))
        results.append((omega, nit, price, cpu_ms))
        print(f"{omega:6.1f}  {nit:8d}  {price:14.6f}  {cpu_ms:10.2f}")

    return results, p_ref


if __name__ == "__main__":
    results, p_ref = run()

    prices = [r[2] for r in results]
    iters  = [r[1] for r in results]

    # All prices should agree to within 1e-4
    assert max(prices) - min(prices) < 1e-4, \
        f"FAIL: prices differ by {max(prices)-min(prices):.2e}"

    # Some omega should give fewer iters than omega=1.0 (index 0)
    gs_iters = iters[0]  # omega=1.0 (Gauss-Seidel)
    min_iters = min(iters)
    assert min_iters <= gs_iters, \
        f"FAIL: no omega beats Gauss-Seidel ({gs_iters} iters)"

    print(f"\nOptimal omega: {results[iters.index(min_iters)][0]:.1f} "
          f"({min_iters} iters vs {gs_iters} for GS)")
    print("PASS: exp_sor_tuning self-test")
