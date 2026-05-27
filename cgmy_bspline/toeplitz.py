import numpy as np
import scipy.sparse.linalg as spla
from cgmy_bspline.connection_coefficients import tempered_coeffs_via_ifft


def toeplitz_matvec(first_col, first_row, v):
    """
    Compute T @ v where T is an N x N Toeplitz matrix via circulant embedding.

    Parameters
    ----------
    first_col : array, length N  -- first column of T
    first_row : array, length N  -- first row of T  (first_row[0] == first_col[0])
    v         : array, length N

    Returns
    -------
    array, length N, real
    """
    N = len(first_col)
    # Circulant embedding: build z of length 2N
    z = np.empty(2 * N, dtype=complex)
    z[0] = first_col[0]
    z[1:N] = first_col[1:]
    z[N] = 0.0
    z[N + 1 : 2 * N] = first_row[1:][::-1]

    Z = np.fft.fft(z)
    V = np.fft.fft(np.concatenate([v, np.zeros(N)]))
    return np.real(np.fft.ifft(Z * V)[:N])


class ToeplitzOperator:
    """
    Toeplitz matrix-vector product with one-time circulant FFT precomputation.

    Precomputes FFT(z) in __init__; each subsequent matvec costs only two FFTs.
    """

    def __init__(self, first_col, first_row):
        N = len(first_col)
        self.N = N
        z = np.empty(2 * N, dtype=complex)
        z[0] = first_col[0]
        z[1:N] = first_col[1:]
        z[N] = 0.0
        z[N + 1 : 2 * N] = first_row[1:][::-1]
        self.circulant_fft = np.fft.fft(z)

    def matvec(self, v):
        N = self.N
        V = np.fft.fft(np.concatenate([np.asarray(v, dtype=float), np.zeros(N)]))
        return np.real(np.fft.ifft(self.circulant_fft * V)[:N])

    def as_linear_operator(self):
        return spla.LinearOperator(
            shape=(self.N, self.N),
            matvec=self.matvec,
            dtype=float,
        )


def build_fractional_toeplitz(N, h, delta, p, Y, side):
    """
    Build a ToeplitzOperator for the fractional stiffness matrix.

    (T)_{ij} = h^{-Y} * Lambda(i - j; delta)

    Parameters
    ----------
    N     : matrix size (number of basis functions)
    h     : mesh spacing
    delta : scaled tempering parameter (G*h for 'minus', M*h for 'plus')
    p     : B-spline order
    Y     : CGMY fine-structure parameter
    side  : 'minus' for K_G_minus, 'plus' for K_M_plus

    Uses L = 2*N IFFT points, which resolves lags 0,...,N-1 (positive) and
    -(N-1),...,-1 (negative) without aliasing.
    """
    L = 2 * N
    lam = tempered_coeffs_via_ifft(delta, Y, p, side, L=L) * (h ** (-Y))

    # first_col[k] = T[k, 0] = Lambda(k - 0) = Lambda(k)
    # IFFT indices 0,...,N-1 correspond to positive lags 0,...,N-1 (since N < L//2)
    first_col = lam[:N].copy()

    # first_row[k] = T[0, k] = Lambda(0 - k) = Lambda(-k)
    # IFFT index for Lambda(-k) is (-k) % L
    row_idx = (-np.arange(N)) % L
    first_row = lam[row_idx].copy()

    return ToeplitzOperator(first_col, first_row)
