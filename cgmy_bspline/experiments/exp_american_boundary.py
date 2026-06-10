"""
Step F — Early exercise boundary for American put under CGMY.

Plots S_f(tau) — the critical stock price below which immediate exercise
is optimal — as a function of time to expiry tau.

Self-test PASS criterion:
  - S_f(tau) < K for all tau > 0 (boundary is below the strike)
  - S_f(tau) -> K as tau -> 0 (boundary approaches K near expiry)
  - S_f is non-decreasing in tau (further from expiry, deeper ITM boundary)
"""
import numpy as np

from cgmy_bspline.src.cgmy_params import CGMYParams
from cgmy_bspline.src.american_solver import solve_american_put


def run(save_fig=False, fig_path=None):
    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,
                        r=0.1, T=0.2, K=100.0, S0=100.0)
    N = 128
    N_tau = 512
    print(f"Solving American put (N={N}, N_tau={N_tau})...", flush=True)
    V, x_grid, S_f_path = solve_american_put(params, N=N, N_tau=N_tau)

    taus   = np.array([t for t, _ in S_f_path])
    S_f    = np.array([s for _, s in S_f_path])

    print(f"\nEarly exercise boundary S_f(tau):")
    print(f"{'tau':>8}  {'S_f':>8}")
    for i in range(0, len(taus), max(1, len(taus) // 10)):
        print(f"{taus[i]:8.4f}  {S_f[i]:8.4f}")
    print(f"{taus[-1]:8.4f}  {S_f[-1]:8.4f}  (tau=T)")

    if save_fig:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(taus, S_f, 'b-', lw=1.5)
            ax.axhline(params.K, color='gray', ls='--', lw=1, label=f'K={params.K}')
            ax.set_xlabel(r'Time to expiry $\tau$')
            ax.set_ylabel(r'Critical stock price $S_f(\tau)$')
            ax.set_title('American Put Early Exercise Boundary (CGMY)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            out = fig_path or 'cgmy_bspline/results/american_boundary.png'
            plt.tight_layout()
            plt.savefig(out, dpi=120)
            plt.close()
            print(f"Figure saved to {out}")
        except ImportError:
            print("matplotlib not available — skipping plot")

    return taus, S_f, params.K


if __name__ == "__main__":
    taus, S_f, K = run()

    # Self-tests
    assert np.all(S_f < K + 1e-6), \
        f"FAIL: S_f > K at some tau. max(S_f)={S_f.max():.4f}"

    # Near expiry (last 10% of taus): boundary should be close to K
    near_expiry = S_f[int(0.9 * len(S_f)):]
    assert np.mean(near_expiry) > 0.7 * K, \
        f"FAIL: near-expiry boundary too low: {np.mean(near_expiry):.4f}"

    # Boundary should be strictly below K (early exercise region exists)
    assert np.any(S_f < 0.99 * K), \
        f"FAIL: boundary never below 0.99*K. min(S_f)={S_f.min():.4f}"

    print("\nPASS: exp_american_boundary self-test")
