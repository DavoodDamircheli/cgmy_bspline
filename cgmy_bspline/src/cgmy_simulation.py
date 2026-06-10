"""
CGMY stock-price path simulation for Monte Carlo pricing.

Increments are sampled from the exact CGMY characteristic function via
FFT-based PDF inversion (no approximation beyond the FFT discretisation).

Risk-neutral measure: log(S_{t+dt}/S_t) = (r - omega_S)*dt + X_{dt}
where omega_S = params.omega_S ensures E[S_T] = S_0 * exp(r*T).
"""
import numpy as np
from scipy.special import gamma as sc_gamma


# ---------------------------------------------------------------------------
# Internal: CDF of one CGMY(C,G,M,Y,dt) increment
# ---------------------------------------------------------------------------

def _build_cdf(C, G, M, Y, dt, N_fft=None, n_std=22):
    # Y < 1: CF decays as exp(-const*xi^Y) — very slowly. Need N~2^22.
    # Y ≥ 1: CF decays as exp(-const*xi^Y) much faster; 2^15 is sufficient
    #   with L = max(n_std*std, 5/min(G,M)), giving ≥200 pts/σ resolution.
    if N_fft is None:
        N_fft = 2**22 if Y < 1 else 2**15
    """
    Compute x grid and CDF of X_dt ~ CGMY(C,G,M,Y) over one step dt.

    Method
    ------
    Uses numpy's FFT frequency convention (fftfreq), which includes both
    positive and negative frequencies — essential for asymmetric processes.

    Derivation:
      f(x) = (1/2pi) int phi(xi) exp(-i xi x) dxi
      Discretise with xi_k = fftfreq(N,dx)*2pi (standard FFT convention):
        f(k*dx) ≈ (dxi/2pi) * FFT(phi)[k]  =  FFT(phi)[k] / (N*dx)
      fftshift recentres the x grid from [0, 2L) to (-L, L].

    Normalisation check:
      sum_k FFT(phi)[k] = N * phi[0] = N * 1 = N
      sum_k f(k*dx) * dx = N/(N*dx) * dx = 1  ✓
    """
    gam_Y = sc_gamma(-Y)

    # Domain half-width strategy:
    #
    # Y < 1  (finite variation, slow CF decay): the CF decays only as
    #   exp(-const * xi^Y), so N_fft~2^22 is needed AND the domain must
    #   reach the exponential tail: L = max(n_std*std, 10/min(G,M)).
    #   n_std*std ≈ 0.28 for Set (i) dt=0.002, but 10/M = 2.0 captures tails.
    #
    # Y ≥ 1  (infinite variation, faster CF decay): N_fft=2^15 with
    #   L = max(n_std*std, 5/min(G,M)) gives ≥200 pts/σ AND covers the
    #   Lévy tail deep enough that P(clip) < 1e-7 per step — otherwise
    #   large negative jumps are suppressed and the put is underpriced.
    kappa2 = abs(C * Y * (Y - 1) * gam_Y * (M ** (Y - 2) + G ** (Y - 2)))
    std = np.sqrt(kappa2 * dt)
    if Y < 1.0:
        L = float(max(n_std * std, 10.0 / min(G, M)))
    else:
        L = float(max(n_std * std, 5.0 / min(G, M)))

    dx = 2.0 * L / N_fft

    # CORRECT: use numpy FFT frequency convention — includes negative frequencies
    # xi[k] = 2pi * k / (N*dx)  for k = 0..N/2-1  (positive)
    # xi[k] = 2pi * (k-N) / (N*dx)  for k = N/2..N-1  (negative)
    xi = np.fft.fftfreq(N_fft, d=dx) * 2.0 * np.pi

    # phi(xi_k) = exp(dt * psi(xi_k)), psi = CGMY characteristic exponent
    psi_xi = C * gam_Y * (
        (M - 1j * xi) ** Y - M ** Y
        + (G + 1j * xi) ** Y - G ** Y
    )
    phi = np.exp(dt * psi_xi)

    # PDF: f(k*dx) ≈ FFT(phi)[k] / (N*dx), then fftshift to centre on 0
    pdf_raw = np.real(np.fft.fft(phi)) / (N_fft * dx)
    pdf = np.fft.fftshift(np.maximum(pdf_raw, 0.0))

    x_vals = (np.arange(N_fft) - N_fft // 2) * dx   # from -L to L-dx

    cdf = np.cumsum(pdf) * dx
    cdf /= cdf[-1]                            # normalise to [0, 1]
    return x_vals, cdf


def _inv_cdf_sample(x_vals, cdf, u):
    """Inverse-CDF sampling: given uniform u, return samples via linear interp.

    The CDF is built with a left-endpoint cumulative sum, so
      cdf[k] ≈ F(x_vals[k] + dx/2)  (midpoint-rule convention).
    To remove the resulting -dx/2 bias each sample is shifted by +dx/2.
    """
    dx = x_vals[1] - x_vals[0]
    idx = np.searchsorted(cdf, u, side='right')
    idx = np.clip(idx, 1, len(cdf) - 1)
    lo = cdf[idx - 1]
    hi = cdf[idx]
    frac = (u - lo) / np.maximum(hi - lo, 1e-14)
    return x_vals[idx - 1] + 0.5 * dx + frac * (x_vals[idx] - x_vals[idx - 1])


# ---------------------------------------------------------------------------
# Public: path simulator
# ---------------------------------------------------------------------------

def cgmy_paths(params, n_paths, n_steps, seed=42, N_fft=None):
    """
    Simulate risk-neutral CGMY stock price paths.

    Parameters
    ----------
    params   : CGMYParams
    n_paths  : int   — number of Monte Carlo paths
    n_steps  : int   — number of time steps (uniform spacing dt = T/n_steps)
    seed     : int   — RNG seed for reproducibility
    N_fft    : int   — FFT size for density inversion (power of 2)

    Returns
    -------
    S : ndarray, shape (n_paths, n_steps + 1)
        Stock price at each time point.
    """
    rng = np.random.default_rng(seed)
    C, G, M, Y = params.C, params.G, params.M, params.Y
    r, T, S0 = params.r, params.T, params.S0
    dt = T / n_steps

    omega = params.omega_S                   # martingale correction (exact)
    drift = (r - omega) * dt                # log-price drift per step

    x_vals, cdf = _build_cdf(C, G, M, Y, dt, N_fft)  # N_fft=None → auto

    # Generate all increments at once (vectorised)
    u_all = rng.uniform(size=(n_steps, n_paths))           # (n_steps, n_paths)
    inc_all = _inv_cdf_sample(x_vals, cdf, u_all)          # (n_steps, n_paths)

    # Cumulative sum → log-price paths
    log_S = np.empty((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum(drift + inc_all, axis=0).T

    return np.exp(log_S)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from cgmy_bspline.src.cgmy_params import CGMYParams
    from cgmy_bspline.src.cos_pricer import cos_put_price

    for label, p in [
        ("Set (i)  Y=0.5", CGMYParams(C=0.5, G=5.0, M=5.0, Y=0.5,
                                       r=0.1, T=0.2, K=100., S0=100.)),
        ("Set (ii) Y=1.5", CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,
                                       r=0.1, T=0.2, K=100., S0=100.)),
    ]:
        n_paths, n_steps = 200_000, 100
        S = cgmy_paths(p, n_paths, n_steps, seed=7)

        # Martingale check: E[S_T] = S0 * exp(r*T)
        mean_ST = np.mean(S[:, -1])
        expected = p.S0 * np.exp(p.r * p.T)
        mge_err = abs(mean_ST - expected) / expected
        print(f"{label}: E[S_T]={mean_ST:.4f}  expected={expected:.4f}  "
              f"rel-err={mge_err:.2%}")
        assert mge_err < 0.01, f"Martingale check failed: {mge_err:.2%}"

        # European put price: payoff at maturity
        disc = np.exp(-p.r * p.T)
        eu_mc = disc * np.mean(np.maximum(p.K - S[:, -1], 0.0))
        eu_se = disc * np.std(np.maximum(p.K - S[:, -1], 0.0)) / np.sqrt(n_paths)
        eu_cos = cos_put_price(p, N_cos=1024)
        diff = abs(eu_mc - eu_cos)
        print(f"  EU-MC={eu_mc:.4f} ± {eu_se:.4f}   COS={eu_cos:.4f}   "
              f"|diff|={diff:.4f}  ({diff/eu_se:.1f}σ)")
        assert diff < 5 * eu_se, f"EU-MC vs COS outside 5σ: {diff/eu_se:.1f}σ"

    print("\nPASS: cgmy_simulation self-test")
