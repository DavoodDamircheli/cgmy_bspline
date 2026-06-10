"""
Longstaff-Schwartz Monte Carlo American put pricer.

Reference: Longstaff & Schwartz (2001), "Valuing American Options by
Simulation: A Simple Least-Squares Approach", RFS 14(1):113-147.

Algorithm
---------
1. Simulate n_paths stock-price paths S[i, t] (t=0..n_steps).
2. Initialise cash flow at maturity: cf[i] = (K - S[i,T])+
3. Backward loop t = n_steps-1 .. 1:
   a. Identify in-the-money (ITM) paths at time t.
   b. Discount ITM paths' future cash flows to time t:
        y[i] = cf[i] * exp(-r*(stop_step[i]-t)*dt)
   c. Regress y on polynomial basis functions of S[i,t] (OLS).
   d. Where intrinsic value > fitted continuation: exercise now.
4. Price = E[ cf[i] * exp(-r * stop_step[i] * dt) ]
"""
import numpy as np


def _basis(S_itm, K, degree=3):
    """Monomial basis [1, s, s^2, ..., s^degree] where s = S/K."""
    x = S_itm / K
    return np.column_stack([x ** d for d in range(degree + 1)])


def lsm_american_put(S, K, r, T, degree=3):
    """
    Price an American put via Longstaff-Schwartz regression.

    Parameters
    ----------
    S      : ndarray (n_paths, n_steps+1) — simulated stock prices
    K      : float  — strike
    r      : float  — risk-free rate
    T      : float  — maturity
    degree : int    — polynomial degree for regression basis

    Returns
    -------
    price  : float — LSM American put price
    stderr : float — Monte Carlo standard error
    """
    n_paths, n_steps_p1 = S.shape
    n_steps = n_steps_p1 - 1
    dt = T / n_steps
    disc_step = np.exp(-r * dt)

    # Intrinsic payoff at every time step
    payoff = np.maximum(K - S, 0.0)          # (n_paths, n_steps+1)

    # --- Initial state: all paths stop at maturity ---
    stop_cf   = payoff[:, -1].copy()         # cash flow when exercised
    stop_step = np.full(n_paths, n_steps, dtype=np.int32)

    # --- Backward recursion ---
    for t in range(n_steps - 1, 0, -1):
        itm = payoff[:, t] > 0.0
        n_itm = int(itm.sum())
        if n_itm < 2 * (degree + 1):        # too few paths to fit
            continue

        # Continuation value in present value at time t
        dt_to_stop = (stop_step[itm] - t).astype(float)
        y = stop_cf[itm] * (disc_step ** dt_to_stop)

        # OLS regression
        X = _basis(S[itm, t], K, degree)
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        cont = X @ coef                       # fitted continuation

        # Exercise where intrinsic > estimated continuation
        exercise = payoff[itm, t] > cont
        ex_idx   = np.where(itm)[0][exercise]
        stop_cf[ex_idx]   = payoff[ex_idx, t]
        stop_step[ex_idx] = t

    # --- Discount all cash flows to t=0 ---
    pv = stop_cf * (disc_step ** stop_step)

    price  = float(np.mean(pv))
    stderr = float(np.std(pv) / np.sqrt(n_paths))
    return price, stderr


def lsm_european_put(S, K, r, T):
    """
    European put price from the same MC paths (payoff at maturity only).
    Useful for validating the path simulator.
    """
    disc = np.exp(-r * T)
    pv = disc * np.maximum(K - S[:, -1], 0.0)
    price  = float(np.mean(pv))
    stderr = float(np.std(pv) / np.sqrt(len(S)))
    return price, stderr


# ---------------------------------------------------------------------------
# Self-test (GBM paths for known Black-Scholes price)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from scipy.stats import norm

    def bs_put(S0, K, r, T, sigma):
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

    rng = np.random.default_rng(0)
    S0, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
    n_paths, n_steps = 100_000, 50
    dt = T / n_steps

    # GBM paths
    W = rng.standard_normal((n_paths, n_steps))
    log_ret = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*W
    log_S = np.hstack([np.full((n_paths,1), np.log(S0)),
                       np.cumsum(log_ret, axis=1)])
    S = np.exp(log_S + np.log(S0) - log_S[:, :1])   # correct initial value
    # Actually simpler:
    log_S2 = np.zeros((n_paths, n_steps+1))
    log_S2[:, 0] = np.log(S0)
    for t in range(n_steps):
        log_S2[:, t+1] = log_S2[:, t] + (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*W[:,t]
    S = np.exp(log_S2)

    eu_mc, eu_se = lsm_european_put(S, K, r, T)
    eu_bs = bs_put(S0, K, r, T, sigma)
    am_mc, am_se = lsm_american_put(S, K, r, T)

    print(f"EU-MC:  {eu_mc:.4f} ± {eu_se:.4f}   BS: {eu_bs:.4f}")
    print(f"AM-LSM: {am_mc:.4f} ± {am_se:.4f}   (should be >= EU)")
    assert abs(eu_mc - eu_bs) < 5 * eu_se, "EU-MC deviates >5σ from BS"
    assert am_mc >= eu_mc - 3 * eu_se,     "AM-LSM < EU-MC (impossible)"
    print("PASS: lsm_pricer self-test")
