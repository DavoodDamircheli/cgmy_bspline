"""
Cardinal B-spline basis functions and log-price grid construction.

cardinal_bspline : Cox–de Boor recursion, N_p(x), support [0, p].
make_log_price_grid : uniform log-price grid centred at log(S0).
"""
import numpy as np


def cardinal_bspline(x, p):
    """
    Evaluate the cardinal B-spline N_p at scalar or array x.
    Cox–de Boor recursion; support [0, p].
    N_1 uses half-open convention: 1 if x in [0,1), else 0.
    Returns zeros for p <= 0.
    """
    x = np.asarray(x, dtype=float)
    if p <= 0:
        return np.zeros_like(x)
    if p == 1:
        return np.where((x >= 0) & (x < 1), 1.0, 0.0)
    return (x / (p - 1)) * cardinal_bspline(x, p - 1) + \
           ((p - x) / (p - 1)) * cardinal_bspline(x - 1, p - 1)


def make_log_price_grid(params, N, buffer_left=5.0, buffer_right=5.0):
    """
    Build a uniform log-price grid on [x_min, x_max].

    x_min = log(S0) - buffer_left
    x_max = log(S0) + buffer_right
    h     = (x_max - x_min) / N
    j     = -log2(h)  (resolution level, float)

    Returns
    -------
    x_grid : ndarray of length N+1, uniformly spaced
    h      : float, mesh spacing
    j      : float, resolution level
    """
    x0 = np.log(params.S0)
    x_min = x0 - buffer_left
    x_max = x0 + buffer_right
    h = (x_max - x_min) / N
    j = -np.log2(h)
    x_grid = x_min + np.arange(N + 1, dtype=float) * h
    return x_grid, h, j


if __name__ == "__main__":
    import numpy as np
    from cgmy_bspline.src.bspline_basis import cardinal_bspline, make_log_price_grid

    # Test N_p integrates to 1 for p=1,2,3
    for p in [1, 2, 3]:
        xs = np.linspace(0, p, 10000)
        integral = np.trapz(cardinal_bspline(xs, p), xs)
        assert abs(integral - 1.0) < 1e-4, f"N_{p} integral={integral}"
        print(f"N_{p} integrates to {integral:.8f}  ✓")

    # Test symmetry: N_p(x) = N_p(p-x) for p >= 2
    for p in [2, 3]:
        xs = np.linspace(0, p, 100)
        diff = np.max(np.abs(cardinal_bspline(xs, p) - cardinal_bspline(p - xs, p)))
        assert diff < 1e-12, f"N_{p} not symmetric: max diff={diff}"
        print(f"N_{p} symmetry error = {diff:.2e}  ✓")

    # Test grid
    from cgmy_bspline.src.cgmy_params import CGMYParams
    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=0.1, T=0.2, K=100.0, S0=100.0)
    x_grid, h, j = make_log_price_grid(params, N=256)
    assert len(x_grid) == 257
    assert abs(x_grid[1] - x_grid[0] - h) < 1e-14
    print(f"Grid: N=256, h={h:.6f}, j={j:.4f}, x in [{x_grid[0]:.3f}, {x_grid[-1]:.3f}]")
    print("PASS: bspline_basis self-test")
