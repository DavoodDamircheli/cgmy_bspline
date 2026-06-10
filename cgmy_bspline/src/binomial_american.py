"""
Step D — Binomial tree (CRR) reference pricer for American put under CGMY.

Moment-matches the CGMY variance:
  kappa_2 = C * Gamma(2-Y) * (M^{Y-2} + G^{Y-2})
           = C * Y*(Y-1) * Gamma(-Y) * (M^{Y-2} + G^{Y-2})

CRR parameters:
  sigma = sqrt(kappa_2)
  u     = exp(sigma * sqrt(dt))
  d     = 1/u
  p_up  = (exp(r*dt) - d) / (u - d)

This is a rough approximation; the binomial tree prices a GBM put with
the same instantaneous variance as CGMY, not the exact Lévy process.
"""
import numpy as np
from scipy.special import gamma


def cgmy_variance(C, G, M, Y):
    """Second cumulant (variance per unit time) of CGMY(C, G, M, Y).

    Uses Gamma(2-Y) = Y*(Y-1)*Gamma(-Y), valid for Y in (0, 2), Y != 1.
    """
    return C * Y * (Y - 1) * gamma(-Y) * (M ** (Y - 2) + G ** (Y - 2))


def binomial_american_put(params, n_steps=2000):
    """
    Price an American put via CRR binomial tree, moment-matched to CGMY variance.

    Parameters
    ----------
    params  : object with .C .G .M .Y .r .T .K .S0
    n_steps : number of time steps

    Returns
    -------
    price : float — American put price at S0
    """
    C, G, M, Y = params.C, params.G, params.M, params.Y
    r, T, K, S0 = params.r, params.T, params.K, float(params.S0)

    dt = T / n_steps
    kappa_2 = cgmy_variance(C, G, M, Y)
    sigma = np.sqrt(abs(kappa_2))

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p_up = (np.exp(r * dt) - d) / (u - d)
    p_dn = 1.0 - p_up

    if not (0.0 < p_up < 1.0):
        raise ValueError(f"Risk-neutral probability out of range: p_up={p_up:.4f}. "
                         "Increase n_steps or check params.")

    # Terminal stock prices: S0 * u^(n-j) * d^j, j=0..n
    j = np.arange(n_steps + 1, dtype=float)
    ST = S0 * (u ** (n_steps - j)) * (d ** j)
    V = np.maximum(K - ST, 0.0)

    # Backward induction with early-exercise projection
    for step in range(n_steps - 1, -1, -1):
        V = disc * (p_up * V[:step + 1] + p_dn * V[1:step + 2])
        # Stock prices at this node (j=0..step)
        j_step = np.arange(step + 1, dtype=float)
        ST_step = S0 * (u ** (step - j_step)) * (d ** j_step)
        V = np.maximum(V, np.maximum(K - ST_step, 0.0))

    return float(V[0])


def binomial_european_put(params, n_steps=2000):
    """European put via binomial tree, for comparison."""
    C, G, M, Y = params.C, params.G, params.M, params.Y
    r, T, K, S0 = params.r, params.T, params.K, float(params.S0)

    dt = T / n_steps
    kappa_2 = cgmy_variance(C, G, M, Y)
    sigma = np.sqrt(abs(kappa_2))

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p_up = (np.exp(r * dt) - d) / (u - d)
    p_dn = 1.0 - p_up

    j = np.arange(n_steps + 1, dtype=float)
    ST = S0 * (u ** (n_steps - j)) * (d ** j)
    V = np.maximum(K - ST, 0.0)

    for step in range(n_steps - 1, -1, -1):
        V = disc * (p_up * V[:step + 1] + p_dn * V[1:step + 2])

    return float(V[0])


if __name__ == "__main__":
    from cgmy_bspline.src.cgmy_params import CGMYParams

    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,
                        r=0.1, T=0.2, K=100.0, S0=100.0)

    print("Computing binomial American put (n_steps=2000)...", flush=True)
    am_bin = binomial_american_put(params, n_steps=2000)
    eu_bin = binomial_european_put(params, n_steps=2000)
    kappa2 = cgmy_variance(params.C, params.G, params.M, params.Y)

    print(f"CGMY variance (kappa_2):  {kappa2:.6f}")
    print(f"Implied sigma:            {kappa2**0.5:.6f}")
    print(f"Binomial European put:    {eu_bin:.6f}")
    print(f"Binomial American put:    {am_bin:.6f}")
    print(f"Early exercise premium:   {am_bin - eu_bin:.6f}")

    assert am_bin >= eu_bin - 1e-8, "FAIL: American < European"
    assert 0.0 < am_bin <= params.K, f"FAIL: price {am_bin:.4f} out of range [0, K]"
    assert am_bin > eu_bin + 1e-4, "WARN: EEP essentially zero"
    print("PASS: binomial_american self-test")
