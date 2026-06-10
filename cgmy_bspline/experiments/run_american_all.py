"""
Step J — Master script: run all American option experiments (Steps A–I).

Each step is imported and run in sequence. Any failure raises an AssertionError
and halts the script with a FAIL message. All steps passing prints PASS at end.
"""
import sys
import traceback


def run_step(name, fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        fn()
        print(f"  >>> {name}: PASS")
    except AssertionError as e:
        print(f"  >>> {name}: FAIL — {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  >>> {name}: ERROR — {e}")
        traceback.print_exc()
        sys.exit(1)


def step_A():
    """american_utils: payoff projection and exercise boundary."""
    from cgmy_bspline.src.american_utils import put_payoff_projection, early_exercise_boundary
    import numpy as np
    from scipy.sparse import eye as speye
    x_grid = np.linspace(3.0, 5.5, 200)
    K = 100.0
    M_h = speye(200, format='csr')
    g = put_payoff_projection(x_grid, K, M_h)
    psi = np.maximum(K - np.exp(x_grid), 0)
    assert np.max(np.abs(g - psi)) < 1e-10, "payoff projection failed"
    x_f = early_exercise_boundary(psi, x_grid, K)
    assert np.exp(x_f) < K + 1.0, "exercise boundary should be at or below strike"


def step_B():
    """psor_solver: PSOR LCP solver for banded systems."""
    import numpy as np
    from scipy.sparse import diags
    from cgmy_bspline.src.psor_solver import psor_lcp_banded
    N = 50
    F = diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format='csr')
    b = np.ones(N)
    g = np.zeros(N)
    c, n_iter, converged = psor_lcp_banded(F, b, g, np.zeros(N),
                                            omega_sor=1.5, tol=1e-12,
                                            max_iter=5000)
    residual = F @ c - b
    assert np.all(residual >= -1e-8), f"LCP infeasible: {residual.min():.3e}"
    assert converged, f"SOR did not converge: n_iter={n_iter}"
    g2 = 0.5 * np.ones(N)
    c2, _, conv2 = psor_lcp_banded(F, b, g2, g2.copy(),
                                    omega_sor=1.5, tol=1e-12, max_iter=5000)
    assert np.all(c2 >= g2 - 1e-10), "Constraint c >= g violated"


def step_C():
    """american_solver: CN + PSOR LCP American put solver."""
    from cgmy_bspline.src.cgmy_params import CGMYParams
    from cgmy_bspline.src.american_solver import price_american_put
    from cgmy_bspline.src.cn_solver import price_put as price_european
    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,
                        r=0.1, T=0.2, K=100.0, S0=100.0)
    am = price_american_put(params, N=128, N_tau=40)
    eu = price_european(params, N=128, N_tau=40)
    assert am >= eu - 1e-8, f"AM ({am:.4f}) < EU ({eu:.4f})"
    assert am > eu + 1e-4, "EEP essentially zero"
    assert am <= params.K, f"AM ({am:.4f}) > K={params.K}"


def step_D():
    """binomial_american: binomial tree reference pricer."""
    from cgmy_bspline.src.cgmy_params import CGMYParams
    from cgmy_bspline.src.binomial_american import binomial_american_put, binomial_european_put
    params = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,
                        r=0.1, T=0.2, K=100.0, S0=100.0)
    am_bin = binomial_american_put(params, n_steps=500)
    eu_bin = binomial_european_put(params, n_steps=500)
    assert am_bin >= eu_bin - 1e-8, "AM bin < EU bin"
    assert 0.0 < am_bin <= params.K


def step_E():
    """exp_american_convergence: convergence study N=64,128,256,512."""
    from cgmy_bspline.experiments.exp_american_convergence import run
    am_prices, bin_price = run()
    for price in am_prices:
        assert price >= 0
    assert abs(am_prices[-1] - am_prices[-2]) < 0.05, \
        f"not converged: {am_prices[-2]:.4f} vs {am_prices[-1]:.4f}"


def step_F():
    """exp_american_boundary: early exercise boundary."""
    from cgmy_bspline.experiments.exp_american_boundary import run
    import numpy as np
    taus, S_f, K = run()
    assert np.all(S_f < K + 1e-6), f"S_f > K at some tau"
    assert np.any(S_f < 0.99 * K), f"boundary never below 0.99*K"


def step_G():
    """exp_american_comparison: method comparison table."""
    from cgmy_bspline.experiments.exp_american_comparison import run, PARAM_SETS
    from cgmy_bspline.src.cn_solver import price_put as price_european
    all_am_prices = run()
    for label, am_list in all_am_prices.items():
        params = PARAM_SETS[label]
        for i, (N, am) in enumerate(zip([64, 128, 256], am_list)):
            eu = price_european(params, N=N, N_tau=4*N)
            assert am >= eu - 0.05, \
                f"Set {label} N={N}: AM={am:.4f} < EU={eu:.4f}"
        assert abs(am_list[-1] - am_list[-2]) < 0.20


def step_H():
    """exp_sor_tuning: PSOR relaxation parameter sensitivity."""
    from cgmy_bspline.experiments.exp_sor_tuning import run
    results, p_ref = run()
    prices = [r[2] for r in results]
    iters  = [r[1] for r in results]
    assert max(prices) - min(prices) < 1e-4
    assert min(iters) <= iters[0]


def step_I():
    """exp_american_latex: LaTeX table generation."""
    from cgmy_bspline.experiments.exp_american_latex import run
    latex = run()
    assert len(latex) > 100
    assert "tabular" in latex


if __name__ == "__main__":
    run_step("Step A: american_utils", step_A)
    run_step("Step B: psor_solver", step_B)
    run_step("Step C: american_solver", step_C)
    run_step("Step D: binomial_american", step_D)
    run_step("Step E: exp_american_convergence", step_E)
    run_step("Step F: exp_american_boundary", step_F)
    run_step("Step G: exp_american_comparison", step_G)
    run_step("Step H: exp_sor_tuning", step_H)
    run_step("Step I: exp_american_latex", step_I)

    print(f"\n{'='*60}")
    print("  ALL STEPS PASS")
    print(f"{'='*60}")
