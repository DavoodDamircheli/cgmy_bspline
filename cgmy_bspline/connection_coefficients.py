import numpy as np


def tempered_coeffs_via_ifft(delta, Y, p, sign, L):
    """
    Compute L connection coefficients Lambda(m; delta) via IFFT.

    Uses the Fourier-space representation
      Lambda(m) = (1/2pi) * integral |hat_N_p(tau)|^2 * S(tau; delta) * e^{i*tau*m} dtau
    discretised on the standard FFT grid so that IFFT gives the coefficients directly.

    Parameters
    ----------
    delta : float  -- scaled decay: G*h ('minus') or M*h ('plus')
    Y     : float  -- CGMY fine-structure, Y in (0,2), Y != 1
    p     : int    -- B-spline order
    sign  : str    -- 'plus' (right-jump part) or 'minus' (left-jump part)
    L     : int    -- number of frequency/output points

    Returns
    -------
    Lambda : ndarray, shape (L,), real
      Lambda[m] = Lambda(m) for m=0,...,L//2-1 (positive lags);
      Lambda[L-k] = Lambda(-k) for k=1,...,L//2 (negative lags, IFFT wrapping).
    """
    tau = np.fft.fftfreq(L) * (2.0 * np.pi)

    # hat_N_p(tau) = ((1 - e^{-i*tau}) / (i*tau))^p, hat_N_p(0) = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        hat_Np = np.where(
            np.abs(tau) < 1e-14,
            complex(1.0),
            ((1.0 - np.exp(-1j * tau)) / (1j * tau)) ** p,
        )

    if sign == 'plus':
        # Right-jump symbol (positive jumps, decay M):
        # S_+(tau) = (delta - i*tau)^Y - delta^Y
        # Matches C*Gamma(-Y)*[(M-iu)^Y - M^Y] in the CGMY characteristic function
        # for all Y in (0,2); no additional compensation needed because the drift
        # (r - omega_S) already handles the centering for Y in (1,2).
        S = (delta - 1j * tau) ** Y - delta ** Y
    else:
        # Left-jump symbol (negative jumps, decay G):
        # S_-(tau) = (delta + i*tau)^Y - delta^Y
        S = (delta + 1j * tau) ** Y - delta ** Y

    G = np.abs(hat_Np) ** 2 * S

    # G is Hermitian on the FFT grid => IFFT is real
    return np.real(np.fft.ifft(G))
