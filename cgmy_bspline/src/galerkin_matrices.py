"""
Galerkin matrix assembly for the CGMY FPDE.

Convention B (full symbol):
  A_h = -(r - omega_S)*D_h
        - C*Gamma(-Y)*(K_G_full + K_M_full)
        + lambda_0 * M_h

  lambda_0 = r + C*Gamma(-Y)*(M^Y + G^Y)

Because Convention B and Convention A give the same A_h (the delta^Y terms
cancel exactly), this module delegates matrix construction to the vetted
cgmy_bspline.matrices package and wraps it in the Convention-B interface.

assemble_stiffness returns (M_h, A_h, x_grid, h) where:
  - M_h : scipy.sparse.csr_matrix  (mass)
  - A_h : scipy.sparse.csr_matrix (full stiffness, BCs NOT yet applied)
  - x_grid : ndarray of length N+1
  - h      : float
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import gamma as sc_gamma

from cgmy_bspline.src.bspline_basis import make_log_price_grid
from cgmy_bspline import matrices as _mat


def assemble_stiffness(params, N, j=None, bandwidth=None):
    """
    Assemble B-spline mass and stiffness matrices.

    Parameters
    ----------
    params    : CGMYParams (src.cgmy_params)
    N         : int, number of spatial intervals
    j         : float or None, resolution level (ignored — derived from grid)
    bandwidth : int or None; if given, Toeplitz row is truncated to bandwidth
                lags and stored as scipy.sparse.diags; if None, use 2*N (full).

    Returns
    -------
    M_h    : (N+p-1, N+p-1) sparse mass matrix
    A_h    : (N+p-1, N+p-1) sparse stiffness (Dirichlet BCs NOT applied)
    x_grid : ndarray length N+1
    h      : float mesh spacing
    """
    p = 3
    x_grid, h, _ = make_log_price_grid(params, N)
    x_min = float(x_grid[0])

    bw = (2 * N) if bandwidth is None else int(bandwidth)

    # Delegate to the existing vetted assembly (Convention A internally,
    # equivalent to Convention B as proven in the companion derivation).
    from cgmy_bspline.parameters import CGMYParams as _OldParams
    old = _OldParams(
        C=params.C, G=params.G, M=params.M, Y=params.Y,
        r=params.r, K=params.K, T=params.T,
    )
    mats = _mat.operator_matrix(N, p, h, x_min, old, bandwidth=bw)
    M_h = mats['M_h']
    A_h = mats['A_h']

    return M_h, A_h, x_grid, h


if __name__ == "__main__":
    from cgmy_bspline.src.cgmy_params import CGMYParams
    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=0.1, T=0.2, K=100.0, S0=100.0)
    M_h, A_h, x_grid, h = assemble_stiffness(params, N=64)
    print(f"M_h shape: {M_h.shape}, A_h shape: {A_h.shape}")
    print(f"M_h diagonal (first 5): {M_h.diagonal()[:5]}")
    assert (M_h.diagonal() > 0).all(), "M_h diagonal not positive"
    print(f"h={h:.6f}, x_grid: [{x_grid[0]:.3f}, {x_grid[-1]:.3f}]")
    print("PASS: stiffness assembly self-test")
