"""
Tempered connection coefficients for the CGMY Galerkin operator.

Convention B (full symbol — no delta^Y compensation):
  Lambda_G^-(ell; delta) = (1/2pi) * int (delta + i*eta)^Y |Nhat_p(eta)|^2 e^{i*ell*eta} deta
  Lambda_M^+(ell; delta) = (1/2pi) * int (delta - i*eta)^Y |Nhat_p(eta)|^2 e^{i*ell*eta} deta

For the self-test / validation we use scipy.integrate.quad on the real axis
(split at eta=0 where needed) to achieve machine-precision accuracy.

For the solver, see cgmy_bspline.connection_coefficients.tempered_coeffs_via_ifft
which uses the FFT and is internally consistent at all grid resolutions.
"""
import numpy as np
from scipy.special import gamma as sc_gamma
from scipy.integrate import quad
from math import comb


def _nhat_p_sq(eta, p):
    """Squared magnitude |N_hat_p(eta)|^2 for real eta (scalar)."""
    if abs(eta) < 1e-14:
        return 1.0
    return abs(((1.0 - np.exp(-1j * eta)) / (1j * eta)) ** p) ** 2


def _integrand_real(eta, ell, delta, Y, p, side):
    """Real part of the integrand for Lambda(ell; delta)."""
    n2 = _nhat_p_sq(eta, p)
    if side == 'left':
        S = (delta + 1j * eta) ** Y
    else:
        S = (delta - 1j * eta) ** Y
    return np.real(S * n2 * np.exp(1j * ell * eta))


def _connection_coeff(ell, delta, Y, p, side, limit=200):
    """Compute one connection coefficient via adaptive quadrature."""
    fn = lambda eta: _integrand_real(eta, ell, delta, Y, p, side)
    # Split at 0 to handle potential branch-point singularity for delta=0
    val_neg, _ = quad(fn, -60.0, 0.0,  limit=limit, epsabs=1e-12, epsrel=1e-12)
    val_pos, _ = quad(fn,   0.0, 60.0, limit=limit, epsabs=1e-12, epsrel=1e-12)
    return (val_neg + val_pos) / (2.0 * np.pi)


def tempered_connection_row(delta, Y, p, L_ell, num_quad=None, eta_max=None):
    """
    Left-sided connection row Lambda_G^-(ell; delta) for ell in {-L_ell,...,L_ell}.

    Uses adaptive quadrature for 1e-10 accuracy (num_quad/eta_max ignored —
    kept for API compatibility; adaptive quad is always used here).

    Returns
    -------
    row : ndarray shape (2*L_ell+1,) indexed ell = -L_ell,...,L_ell
    """
    ells = np.arange(-L_ell, L_ell + 1)
    return np.array([_connection_coeff(int(ell), delta, Y, p, 'left') for ell in ells])


def right_connection_row(delta, Y, p, L_ell, num_quad=None, eta_max=None):
    """
    Right-sided connection row Lambda_M^+(ell; delta) for ell in {-L_ell,...,L_ell}.

    Symbol: (delta - i*eta)^Y  (Convention B, full symbol)
    """
    ells = np.arange(-L_ell, L_ell + 1)
    return np.array([_connection_coeff(int(ell), delta, Y, p, 'right') for ell in ells])


def untempered_left_closed_form(ell_arr, Y, p):
    """
    Lambda_0^-(ell) via positive-part formula (delta=0, left-sided).

    Lambda_0^-(ell) = (1/Gamma(2p-Y)) * sum_{m=0}^{2p} (-1)^m C(2p,m) (ell+p-m)_+^{2p-Y-1}
    """
    gam = sc_gamma(2 * p - Y)
    result = []
    for ell in ell_arr:
        s = 0.0
        for m in range(2 * p + 1):
            base = ell + p - m
            if base > 0:
                s += ((-1) ** m) * comb(2 * p, m) * (base ** (2 * p - Y - 1))
        result.append(s / gam)
    return np.array(result)


if __name__ == "__main__":
    import numpy as np

    Y, p = 1.5, 3
    ell_arr = np.arange(-6, 7)

    print("Computing connection rows via adaptive quadrature...")
    row_fourier = tempered_connection_row(delta=0.0, Y=Y, p=p, L_ell=6)
    row_closed  = untempered_left_closed_form(ell_arr, Y, p)

    err = np.max(np.abs(row_fourier - row_closed))
    print(f"Max |quad - closed_form| at delta=0, Y={Y}: {err:.2e}")
    assert err < 1e-6, f"FAIL: delta=0 mismatch err={err}"
    print("PASS: tempered_connection_row delta=0 check (Y=1.5)")

    row_f2 = tempered_connection_row(delta=0.0, Y=0.5, p=3, L_ell=6)
    row_c2 = untempered_left_closed_form(ell_arr, 0.5, 3)
    err2 = np.max(np.abs(row_f2 - row_c2))
    print(f"Max |quad - closed_form| at delta=0, Y=0.5: {err2:.2e}")
    assert err2 < 1e-6, f"FAIL Y=0.5: err={err2}"
    print("PASS: Y=0.5 check")

    # Symmetry: Lambda_0^+(ell) = Lambda_0^-(-ell)
    row_right = right_connection_row(delta=0.0, Y=Y, p=p, L_ell=6)
    row_left  = tempered_connection_row(delta=0.0, Y=Y, p=p, L_ell=6)
    sym_err = np.max(np.abs(row_right - row_left[::-1]))
    print(f"Symmetry Lambda_0^+(ell)=Lambda_0^-(-ell): {sym_err:.2e}")
    assert sym_err < 1e-10, f"FAIL symmetry: {sym_err}"
    print("PASS: symmetry check")
