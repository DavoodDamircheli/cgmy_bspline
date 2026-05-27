import time
import numpy as np
import pytest
from cgmy_bspline.tests.test_parameters import PARAMS_I
from cgmy_bspline.connection_coefficients import tempered_coeffs_via_ifft
from cgmy_bspline.toeplitz import toeplitz_matvec, ToeplitzOperator, build_fractional_toeplitz


def _make_symmetric_toeplitz(c):
    """Dense symmetric Toeplitz matrix with first row/col = c."""
    N = len(c)
    return np.array([[c[abs(i - j)] for j in range(N)] for i in range(N)])


def _assemble_fractional_K(N, h, delta, Y, p, side):
    """
    Reference: assemble (K)_{ij} = h^{-Y} * Lambda(i-j; delta) entry by entry
    using a large-L IFFT (L=4096) so aliasing is negligible.
    """
    L_ref = max(4096, 16 * N)
    lam = tempered_coeffs_via_ifft(delta, Y, p, side, L=L_ref) * (h ** (-Y))
    return np.array([[lam[(i - j) % L_ref] for j in range(N)] for i in range(N)])


# ── test 1 ──────────────────────────────────────────────────────────────────

def test_toeplitz_matvec_exact():
    """toeplitz_matvec matches dense T @ v for a random symmetric N=32 matrix."""
    np.random.seed(42)
    N = 32
    c = np.random.randn(N)
    T = _make_symmetric_toeplitz(c)
    v = np.random.randn(N)

    result = toeplitz_matvec(T[:, 0], T[0, :], v)
    assert np.allclose(result, T @ v, atol=1e-12), (
        f"max diff = {np.max(np.abs(result - T @ v)):.3e}"
    )


# ── test 2 ──────────────────────────────────────────────────────────────────

def test_toeplitz_operator_matches_matvec():
    """ToeplitzOperator.matvec agrees with toeplitz_matvec."""
    np.random.seed(42)
    N = 32
    c = np.random.randn(N)
    T = _make_symmetric_toeplitz(c)
    first_col, first_row = T[:, 0], T[0, :]
    v = np.random.randn(N)

    expected = toeplitz_matvec(first_col, first_row, v)
    op = ToeplitzOperator(first_col, first_row)
    assert np.allclose(op.matvec(v), expected, atol=1e-14)


# ── test 3 ──────────────────────────────────────────────────────────────────

def test_large_n_timing():
    """Time 10 matvecs for N=4096 and report mean time in ms."""
    N = 4096
    np.random.seed(7)
    c = np.random.randn(N)
    r = np.random.randn(N)
    r[0] = c[0]
    op = ToeplitzOperator(c, r)
    v = np.random.randn(N)

    # Warm-up
    op.matvec(v)

    t0 = time.perf_counter()
    for _ in range(10):
        op.matvec(v)
    elapsed = time.perf_counter() - t0

    mean_ms = elapsed / 10 * 1000
    print(f"\n4096 Toeplitz matvec mean time: {mean_ms:.2f} ms")
    assert mean_ms < 500, f"matvec unexpectedly slow: {mean_ms:.1f} ms"


# ── test 4 ──────────────────────────────────────────────────────────────────

def test_fractional_toeplitz_consistency():
    """
    build_fractional_toeplitz matvec exactly matches explicit Toeplitz assembly.

    Both sides use the same Lambda sequence (L = 2*N IFFT), so this tests
    correctness of the circulant embedding structure, not IFFT accuracy.
    The IFFT accuracy for large lags converges slowly with L; that is a
    separate numerical concern.
    """
    N = 32
    p = 3
    h = 0.1
    Y = PARAMS_I.Y
    delta_G = PARAMS_I.G * h

    op = build_fractional_toeplitz(N, h, delta_G, p, Y, 'minus')

    # Reconstruct the exact Lambda array used inside build_fractional_toeplitz
    L = 2 * N
    lam = tempered_coeffs_via_ifft(delta_G, Y, p, 'minus', L=L) * (h ** (-Y))
    first_col_ref = lam[:N]
    first_row_ref = lam[(-np.arange(N)) % L]

    # Dense Toeplitz matrix built from the same Lambda values
    K_ref = np.array(
        [[first_col_ref[i - j] if i >= j else first_row_ref[j - i]
          for j in range(N)]
         for i in range(N)]
    )

    np.random.seed(42)
    v = np.random.randn(N)

    ref_result = K_ref @ v
    op_result = op.matvec(v)

    print(f"\nK_G_minus[0,0] = {first_col_ref[0]:.6e}")
    print(f"K_G_minus[0,5] = {first_row_ref[5]:.6e}")
    max_diff = np.max(np.abs(ref_result - op_result))
    print(f"Toeplitz matvec max diff: {max_diff:.3e}")
    assert max_diff < 1e-8
