"""
Crank-Nicolson B-spline Galerkin solver for European put under CGMY.

Delegates to cgmy_bspline.solver_cn.solve_cn with appropriate parameter
conversion (src.CGMYParams includes S0 which the old interface lacks).

solve_european_put  returns (V, x_grid)
price_put           returns scalar price at log(S0) via interpolation
"""
import numpy as np
import scipy.sparse.linalg as spla

from cgmy_bspline.src.cgmy_params import CGMYParams
from cgmy_bspline.parameters import CGMYParams as _OldParams
from cgmy_bspline.solver_cn import solve_cn
from cgmy_bspline.grid import make_grid
from cgmy_bspline.projection import eval_solution


def _to_old(params: CGMYParams) -> _OldParams:
    return _OldParams(
        C=params.C, G=params.G, M=params.M, Y=params.Y,
        r=params.r, K=params.K, T=params.T,
    )


def solve_european_put(params: CGMYParams, N: int, N_tau: int,
                       sigma_shift=None, bandwidth=None):
    """
    Crank-Nicolson B-spline Galerkin solve for a European put.

    Parameters
    ----------
    params    : CGMYParams (src)
    N         : spatial intervals
    N_tau     : time steps
    sigma_shift : (unused, reserved for stability shift)
    bandwidth : int or None; banded truncation bandwidth. Default min(2*N, 256).

    Returns
    -------
    V      : ndarray length N+1, put price at each x_grid node
    x_grid : ndarray length N+1
    """
    old = _to_old(params)
    bw = (2 * N) if bandwidth is None else int(bandwidth)
    result = solve_cn(old, N=N, p=3, N_tau=N_tau, bandwidth=bw, mode='banded')
    V = result['V_final']
    x_grid = result['x_nodes']
    return V, x_grid


def price_put(params: CGMYParams, N: int, N_tau: int, x_eval=None):
    """
    Return put price at x_eval = log(S0) (or specified log-price).

    Parameters
    ----------
    x_eval : float or None; if None, uses log(params.S0)

    Returns
    -------
    price : float
    """
    if x_eval is None:
        x_eval = np.log(params.S0)
    V, x_grid = solve_european_put(params, N, N_tau)
    return float(np.interp(x_eval, x_grid, V))


if __name__ == "__main__":
    import numpy as np
    from cgmy_bspline.src.cgmy_params import CGMYParams

    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=0.1, T=0.2, K=100.0, S0=100.0)
    V, x_grid = solve_european_put(params, N=256, N_tau=50)
    x0 = np.log(params.S0)
    put_price = np.interp(x0, x_grid, V)
    print(f"European put price (Y=1.5): {put_price:.6f}")
    assert 0.5 < put_price < 20.0, f"Implausible put price: {put_price}"
    print("PASS: cn_solver basic sanity check")
