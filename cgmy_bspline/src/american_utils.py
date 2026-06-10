"""
Step A — Payoff projection and early exercise boundary for American put.
"""
import numpy as np
import scipy.sparse.linalg as spla


def put_payoff_projection(x_grid, K, M_h):
    """
    L2 projection of the put payoff onto the B-spline space.

    Solves M_h * g = M_h * psi, returning g = psi (identity for SPD M_h).

    Parameters
    ----------
    x_grid : array of log-price nodes
    K      : strike
    M_h    : sparse mass matrix (N+p-1 square, SPD)

    Returns
    -------
    g : ndarray — payoff at each grid node (constraint vector)
    """
    psi = np.maximum(K - np.exp(x_grid), 0.0)
    rhs = M_h @ psi
    g = spla.spsolve(M_h, rhs)
    return g


def early_exercise_boundary(c, x_grid, K, tol=1e-6):
    """
    Locate the early exercise boundary in log-price space.

    Returns x_f = the largest x_grid[k] where the coefficient c[k] is at
    or below the intrinsic value + tol (option held rather than exercised).

    Parameters
    ----------
    c      : coefficient vector
    x_grid : log-price grid nodes
    K      : strike
    tol    : tolerance for detecting exercise

    Returns
    -------
    x_f : float  (log-price at the boundary)
    """
    intrinsic = np.maximum(K - np.exp(x_grid), 0.0)
    # Exercise region: deep ITM where option = intrinsic
    in_exercise = (intrinsic > 0) & (c <= intrinsic + tol)
    if not np.any(in_exercise):
        return x_grid[0]
    last_idx = int(np.where(in_exercise)[0][-1])
    return float(x_grid[last_idx])


if __name__ == "__main__":
    import numpy as np
    from scipy.sparse import eye as speye

    x_grid = np.linspace(3.0, 5.5, 200)
    K = 100.0
    M_h = speye(200, format='csr')
    g = put_payoff_projection(x_grid, K, M_h)
    psi = np.maximum(K - np.exp(x_grid), 0)
    assert np.max(np.abs(g - psi)) < 1e-10, "payoff projection failed with identity mass"

    x_f = early_exercise_boundary(psi, x_grid, K, tol=1e-6)
    print(f"Early exercise boundary x_f = {x_f:.4f}, S_f = {np.exp(x_f):.4f}")
    assert np.exp(x_f) < K + 1.0, "exercise boundary should be at or below strike"
    print("PASS: american_utils self-test")
