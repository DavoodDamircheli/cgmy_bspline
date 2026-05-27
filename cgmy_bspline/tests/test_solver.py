import numpy as np
import pytest
from cgmy_bspline.tests.test_parameters import PARAMS_I, PARAMS_II
from cgmy_bspline.solver_cn import solve_cn

p = 3


# ── test 1 ──────────────────────────────────────────────────────────────────

def test_manufactured_n64_error_small():
    """Manufactured solution, PARAMS_II, N=64: error_L2 < 0.1."""
    result = solve_cn(PARAMS_II, N=64, p=p, N_tau=64, manufactured=True)
    err = result['error_L2']
    print(f"\nN=64  error_L2={err:.4e}  cpu={result['cpu_time']:.2f}s")
    assert err < 0.1, f"error_L2 = {err:.4e} >= 0.1"


# ── test 2 ──────────────────────────────────────────────────────────────────

def test_manufactured_n128_error_smaller():
    """Manufactured solution, PARAMS_II, N=128: error_L2 < error at N=64."""
    r64  = solve_cn(PARAMS_II, N=64,  p=p, N_tau=64,  manufactured=True)
    r128 = solve_cn(PARAMS_II, N=128, p=p, N_tau=128, manufactured=True)
    e64, e128 = r64['error_L2'], r128['error_L2']
    print(f"\nN= 64: error_L2 = {e64:.4e}")
    print(f"N=128: error_L2 = {e128:.4e}")
    assert e128 < e64, f"Error did not decrease: {e128:.4e} >= {e64:.4e}"


# ── test 3 ──────────────────────────────────────────────────────────────────

def test_european_put_nonnegative():
    """European put, PARAMS_I, N=64: V_final >= -O(h) everywhere."""
    result = solve_cn(PARAMS_I, N=64, p=p, N_tau=64)
    V = result['V_final']
    tol = PARAMS_I.K / 64  # O(h) undershoot allowed near the payoff kink
    assert V.min() >= -tol, f"V_final min = {V.min():.3e}, tol = {-tol:.3e}"


# ── test 4 ──────────────────────────────────────────────────────────────────

def test_european_put_value_at_strike():
    """European put, PARAMS_I, N=64: V_final at log(K) is in (0, K)."""
    result = solve_cn(PARAMS_I, N=64, p=p, N_tau=64)
    x0 = np.log(PARAMS_I.K)
    idx = np.argmin(np.abs(result['x_nodes'] - x0))
    V_at_K = result['V_final'][idx]
    print(f"\nV_final(log(K)) = {V_at_K:.4e}")
    assert 0.0 <= V_at_K <= PARAMS_I.K, (
        f"V_final(log(K)) = {V_at_K:.4e} outside [0, K={PARAMS_I.K}]"
    )
