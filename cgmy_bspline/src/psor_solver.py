"""
Step B — Projected SOR (PSOR) solver for the Linear Complementarity Problem.

Solves:  c >= g,  F*c >= b,  (c-g)^T (F*c - b) = 0

The banded variant uses an incremental column-update strategy to maintain
F*c across the Gauss-Seidel sweep in O(bandwidth) work per node.
"""
import numpy as np
import scipy.sparse as sp


def psor_lcp(F_dense, b, g, c_init, omega_sor=1.2, max_iter=2000, tol=1e-10):
    """
    PSOR for LCP with dense F matrix (Toeplitz / small problems).

    Parameters
    ----------
    F_dense  : (n, n) ndarray
    b        : (n,) RHS vector
    g        : (n,) constraint lower bound  (c >= g)
    c_init   : (n,) initial guess
    omega_sor: SOR relaxation parameter in (0, 2)
    max_iter : maximum SOR sweeps
    tol      : convergence tolerance on max|c_new - c_old|

    Returns
    -------
    c         : solution
    n_iter    : iterations performed
    converged : bool
    """
    c = c_init.copy()
    n = len(c)
    F_diag = np.diag(F_dense)
    Fc = F_dense @ c

    for iteration in range(max_iter):
        c_old = c.copy()
        for k in range(n):
            c_gs  = c[k] + (b[k] - Fc[k]) / F_diag[k]
            c_sor = c[k] + omega_sor * (c_gs - c[k])
            c_new_k = max(c_sor, g[k])
            delta = c_new_k - c[k]
            if abs(delta) > 1e-15:
                Fc += F_dense[:, k] * delta
            c[k] = c_new_k

        if np.max(np.abs(c - c_old)) < tol:
            return c, iteration + 1, True

    return c, max_iter, False


def psor_lcp_banded(F_sparse, b, g, c_init,
                    omega_sor=1.2, max_iter=2000, tol=1e-10):
    """
    PSOR for LCP with sparse (banded) F matrix.

    Uses incremental column-update of the running F*c vector for O(bandwidth)
    work per node update.

    Parameters
    ----------
    F_sparse : scipy sparse matrix (will be converted to CSC + CSR internally)
    b        : (n,) RHS vector
    g        : (n,) constraint lower bound  (c >= g)
    c_init   : (n,) initial guess
    omega_sor: SOR relaxation parameter in (0, 2)
    max_iter : maximum SOR sweeps
    tol      : convergence tolerance on max|c_new - c_old|

    Returns
    -------
    c         : solution
    n_iter    : iterations performed
    converged : bool
    """
    c = c_init.copy()
    n = len(c)

    F_csc = F_sparse.tocsc()
    F_diag = np.asarray(F_sparse.diagonal(), dtype=float)

    # Pre-extract column structure for fast incremental updates
    col_ptrs = F_csc.indptr
    col_inds = F_csc.indices
    col_data = F_csc.data

    # Initial F*c
    Fc = F_sparse.tocsr().dot(c)

    for iteration in range(max_iter):
        c_old = c.copy()

        for k in range(n):
            c_gs  = c[k] + (b[k] - Fc[k]) / F_diag[k]
            c_sor = c[k] + omega_sor * (c_gs - c[k])
            c_new_k = c_sor if c_sor >= g[k] else g[k]
            delta = c_new_k - c[k]
            if abs(delta) > 1e-15:
                s = col_ptrs[k]; e = col_ptrs[k + 1]
                Fc[col_inds[s:e]] += col_data[s:e] * delta
            c[k] = c_new_k

        if np.max(np.abs(c - c_old)) < tol:
            return c, iteration + 1, True

    return c, max_iter, False


if __name__ == "__main__":
    import numpy as np
    from scipy.sparse import diags

    # Unconstrained case: tridiagonal Poisson
    N = 50
    F = diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format='csr')
    b = np.ones(N)
    g = np.zeros(N)
    c, n_iter, converged = psor_lcp_banded(F, b, g, np.zeros(N),
                                           omega_sor=1.5, tol=1e-12,
                                           max_iter=5000)
    residual = F @ c - b
    assert np.all(residual >= -1e-8), f"LCP infeasible: min residual = {residual.min():.3e}"
    assert converged, f"SOR did not converge: n_iter={n_iter}"
    print(f"SOR converged in {n_iter} iterations")

    # Constrained case: g = 0.5, forces nodes to clamp
    g2 = 0.5 * np.ones(N)
    c2, n_iter2, conv2 = psor_lcp_banded(F, b, g2, g2.copy(),
                                          omega_sor=1.5, tol=1e-12,
                                          max_iter=5000)
    assert np.all(c2 >= g2 - 1e-10), "Constraint c >= g violated"
    print(f"Constrained SOR converged in {n_iter2} iterations")
    print("PASS: psor_solver self-test")
