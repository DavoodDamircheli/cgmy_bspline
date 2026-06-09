"""
COS method pricer for European put under CGMY (Fang & Oosterlee 2008).

The characteristic function of log(S_T/S_0) under risk-neutral Q is:
  phi(u) = exp(T * [i*u*(r - omega_S) + Psi(u)])
where Psi(u) = C*Gamma(-Y)*[(M-iu)^Y - M^Y + (G+iu)^Y - G^Y].

Truncation interval [a, b] from cumulants:
  c1 = (r - omega_S)*T
  c2 = C*|Gamma(-Y)|*Y*(Y-1)*(M^{Y-2} + G^{Y-2})*T
  a = c1 - 12*sqrt(|c2|),  b = c1 + 12*sqrt(|c2|)

Put price (K - S_T)_+:
  P = exp(-r*T) * sum_{k=0}^{N-1}' V_k * Re[phi(k*pi/(b-a)) * exp(-ik*pi*a/(b-a))]
  V_k = (2/(b-a)) * [K * Chi_k - S0 * Psi_k]
  (integration of the payoff cosine coefficients over the ITM region)
"""
import numpy as np
from scipy.special import gamma as sc_gamma


def _char_exp_rn(u, params, T):
    """
    Full risk-neutral characteristic function of log(S_T/S_0):
      phi(u) = exp(T * [i*u*(r - omega_S) + Psi_cgmy(u)])
    """
    C, G, M, Y = params.C, params.G, params.M, params.Y
    gY = sc_gamma(-Y)
    omega_S = C * gY * ((M - 1) ** Y - M ** Y + (G + 1) ** Y - G ** Y)
    u = np.asarray(u, dtype=complex)
    psi_cgmy = C * gY * (
        (M - 1j * u) ** Y - M ** Y
        + (G + 1j * u) ** Y - G ** Y
    )
    return np.exp(T * (1j * u * (params.r - omega_S) + psi_cgmy))


def cos_put_price(params, N_cos=1024):
    """
    COS method for European put price.

    Parameters
    ----------
    params : CGMYParams (src)
    N_cos  : number of COS terms (default 1024)

    Returns
    -------
    price : float
    """
    C, G, M, Y = params.C, params.G, params.M, params.Y
    K, T, r, S0 = params.K, params.T, params.r, params.S0

    gY = sc_gamma(-Y)
    omega_S = C * gY * ((M - 1) ** Y - M ** Y + (G + 1) ** Y - G ** Y)

    # Cumulants of log(S_T/S_0)
    c1 = (r - omega_S) * T
    c2 = C * abs(gY) * Y * (Y - 1) * (M ** (Y - 2) + G ** (Y - 2)) * T
    a = c1 - 12.0 * np.sqrt(abs(c2))
    b = c1 + 12.0 * np.sqrt(abs(c2))
    ba = b - a

    # Integration limit for put payoff: non-zero for log(S_T/S_0) < log(K/S0)
    x_cut = min(np.log(K / S0), b)
    d = x_cut - a  # integration width from a

    k = np.arange(N_cos, dtype=float)
    omega_k = k[1:] * np.pi / ba

    # Cosine coefficients of 1 on [a, x_cut]: Chi_k = int_a^{x_cut} cos(omega_k*(x-a)) dx
    Chi = np.empty(N_cos)
    Chi[0] = d
    Chi[1:] = np.sin(omega_k * d) / omega_k

    # Cosine coefficients of exp(x) on [a, x_cut]
    ea, ec = np.exp(a), np.exp(x_cut)
    Psi_arr = np.empty(N_cos)
    Psi_arr[0] = ec - ea
    od = omega_k * d
    Psi_arr[1:] = ((ec * np.cos(od) - ea) + omega_k * ec * np.sin(od)) / (1.0 + omega_k ** 2)

    V_k = (2.0 / ba) * (K * Chi - S0 * Psi_arr)

    # COS sum
    phi_vals = _char_exp_rn(k * np.pi / ba, params, T)
    coeff = np.real(phi_vals * np.exp(-1j * k * np.pi * a / ba))
    coeff[0] *= 0.5

    return float(np.exp(-r * T) * np.dot(coeff, V_k))


def cos_put_surface(params, x_eval_arr, N_cos=1024):
    """
    European put prices at a range of log-strike values.

    Parameters
    ----------
    params      : CGMYParams — params.K is used as the fixed strike K
    x_eval_arr  : array of log(S0) values (i.e., log-moneyness sweep)
    N_cos       : COS terms

    Returns
    -------
    prices : ndarray same length as x_eval_arr
    """
    from cgmy_bspline.src.cgmy_params import CGMYParams
    prices = np.empty(len(x_eval_arr))
    for i, lx in enumerate(x_eval_arr):
        p_i = CGMYParams(
            C=params.C, G=params.G, M=params.M, Y=params.Y,
            r=params.r, T=params.T, K=params.K, S0=np.exp(lx),
        )
        prices[i] = cos_put_price(p_i, N_cos=N_cos)
    return prices


if __name__ == "__main__":
    import numpy as np
    from cgmy_bspline.src.cgmy_params import CGMYParams

    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=0.1, T=0.2, K=100.0, S0=100.0)
    price = cos_put_price(params, N_cos=1024)
    print(f"COS put price (Y=1.5, N=1024): {price:.6f}")

    params2 = CGMYParams(C=0.5, G=5.0, M=5.0, Y=0.5, r=0.1, T=0.2, K=100.0, S0=100.0)
    price2 = cos_put_price(params2, N_cos=1024)
    print(f"COS put price (Y=0.5, N=1024): {price2:.6f}")

    # Stability check: 512 vs 1024 terms differ by < 1e-5
    p512  = cos_put_price(params, N_cos=512)
    p1024 = cos_put_price(params, N_cos=1024)
    diff = abs(p512 - p1024)
    print(f"COS convergence |P_512 - P_1024| = {diff:.2e}  (should be < 1e-5)")

    assert 0.5 < price  < 20.0, f"COS price out of range: {price}"
    assert 0.5 < price2 < 20.0, f"COS price2 out of range: {price2}"
    assert diff < 1e-5,         f"COS not converged: {diff}"
    print("PASS: cos_pricer self-test")
