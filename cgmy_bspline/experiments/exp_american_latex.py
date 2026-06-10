"""
Step I — LaTeX table snippet for American put section.

Generates a LaTeX tabular with:
  - Convergence table (N, EU-BSpline, AM-BSpline, EEP, Binomial) for Set (ii)
  - Exercise boundary S_f at selected tau values

Self-test PASS criterion:
  - LaTeX output is non-empty
  - Contains expected column headers
"""
import numpy as np

from cgmy_bspline.src.cgmy_params import CGMYParams
from cgmy_bspline.src.american_solver import price_american_put, solve_american_put
from cgmy_bspline.src.cn_solver import price_put as price_european
from cgmy_bspline.src.binomial_american import binomial_american_put


def run():
    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,
                        r=0.1, T=0.2, K=100.0, S0=100.0)

    # Convergence table
    N_values  = [64, 128, 256]
    N_tau_fn  = lambda N: 4 * N
    bin_am    = binomial_american_put(params, n_steps=2000)

    rows = []
    for N in N_values:
        am = price_american_put(params, N=N, N_tau=N_tau_fn(N))
        eu = price_european(params, N=N, N_tau=N_tau_fn(N))
        rows.append((N, eu, am, am - eu, am - bin_am))

    # Early exercise boundary (N=128, N_tau=512)
    _, _, S_f_path = solve_american_put(params, N=128, N_tau=512)
    taus = np.array([t for t, _ in S_f_path])
    S_f  = np.array([s for _, s in S_f_path])

    # Select ~6 representative tau values
    idx = np.linspace(0, len(taus) - 1, 8, dtype=int)
    bdy_rows = [(taus[i], S_f[i]) for i in idx]

    latex = r"""
% ── American Put Convergence Table ──────────────────────────────────────────
% CGMY(0.1, 2, 3.5, 1.5), r=0.1, T=0.2, K=100, S_0=100.
% Binomial (GBM moment-matched) reference: """ + f"{bin_am:.4f}" + r"""
\begin{table}[ht]
\centering
\caption{American put prices under CGMY parameter set (ii),
  $C=0.1$, $G=2$, $M=3.5$, $Y=1.5$, $r=0.1$, $T=0.2$, $K=S_0=100$.}
\begin{tabular}{rcccc}
\toprule
$N$ & EU B-Spline & AM B-Spline & EEP & AM $-$ Binomial \\
\midrule
"""
    for (N, eu, am, eep, berr) in rows:
        latex += (f"  {N} & {eu:.4f} & {am:.4f} & {eep:.4f}"
                  f" & {berr:+.4f} \\\\\n")

    latex += r"""\bottomrule
\end{tabular}
\end{table}

% ── Early Exercise Boundary ──────────────────────────────────────────────────
\begin{table}[ht]
\centering
\caption{Early exercise boundary $S_f(\tau)$ for the American put
  under CGMY parameter set (ii). $N=128$, $N_\tau=512$.}
\begin{tabular}{cc}
\toprule
$\tau$ & $S_f(\tau)$ \\
\midrule
"""
    for (tau, sf) in bdy_rows:
        latex += f"  {tau:.4f} & {sf:.4f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    print(latex)
    return latex


if __name__ == "__main__":
    latex = run()
    assert len(latex) > 100, "FAIL: LaTeX output is empty"
    assert "AM B-Spline" in latex, "FAIL: missing column header"
    assert "tabular" in latex, "FAIL: missing tabular environment"
    print("PASS: exp_american_latex self-test")
