"""
LSM Monte Carlo benchmark for American put under CGMY.

Compares B-Spline Galerkin prices against the LSM-CGMY reference
(exact CGMY paths, no GBM approximation) for both parameter sets.

Outputs
-------
- Martingale check and European MC vs COS validation
- LSM American price with 95% confidence interval
- B-Spline prices at N=64, 128, 256 vs LSM reference
- Summary table indicating which B-Spline prices fall inside the MC CI
"""
import time
import numpy as np

from cgmy_bspline.src.cgmy_params import CGMYParams
from cgmy_bspline.src.cgmy_simulation import cgmy_paths
from cgmy_bspline.src.lsm_pricer import lsm_american_put, lsm_european_put
from cgmy_bspline.src.american_solver import price_american_put
from cgmy_bspline.src.cn_solver import price_put as price_european
from cgmy_bspline.src.cos_pricer import cos_put_price

PARAM_SETS = {
    "(i)":  CGMYParams(C=0.5, G=5.0, M=5.0, Y=0.5, r=0.1, T=0.2, K=100., S0=100.),
    "(ii)": CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=0.1, T=0.2, K=100., S0=100.),
}

N_PATHS = 200_000   # MC paths
N_STEPS = 100       # time steps for MC paths
BSPLINE_N = [64, 128, 256]
SEED = 42


def run():
    print(f"\n{'='*70}")
    print(f"  LSM-CGMY vs B-Spline Galerkin: American Put Benchmark")
    print(f"  n_paths={N_PATHS:,}  n_steps_mc={N_STEPS}  seed={SEED}")
    print(f"{'='*70}")

    all_results = {}

    for label, params in PARAM_SETS.items():
        K, r, T = params.K, params.r, params.T
        print(f"\nParameter set {label}: "
              f"C={params.C}, G={params.G}, M={params.M}, Y={params.Y}")
        print("-" * 60)

        # ── Step 1: simulate CGMY paths ──────────────────────────────
        t0 = time.perf_counter()
        S = cgmy_paths(params, N_PATHS, N_STEPS, seed=SEED)
        sim_cpu = time.perf_counter() - t0
        print(f"  Path simulation: {sim_cpu:.1f}s  "
              f"shape={S.shape}")

        # ── Step 2: martingale check ──────────────────────────────────
        mean_ST = float(np.mean(S[:, -1]))
        expected_ST = params.S0 * np.exp(r * T)
        mg_err = (mean_ST - expected_ST) / expected_ST
        print(f"  Martingale check: E[S_T]={mean_ST:.4f}  "
              f"expected={expected_ST:.4f}  rel-err={mg_err:+.4%}")

        # ── Step 3: European MC vs COS ────────────────────────────────
        eu_mc,  eu_se  = lsm_european_put(S, K, r, T)
        eu_cos = cos_put_price(params, N_cos=1024)
        eu_nsd = (eu_mc - eu_cos) / eu_se
        print(f"  EU-MC:  {eu_mc:.5f} ± {eu_se:.5f}   "
              f"COS: {eu_cos:.5f}   diff={eu_mc-eu_cos:+.5f} ({eu_nsd:+.1f}σ)")

        # ── Step 4: LSM American price ────────────────────────────────
        t0 = time.perf_counter()
        am_lsm, am_se = lsm_american_put(S, K, r, T)
        lsm_cpu = time.perf_counter() - t0
        ci95 = 1.96 * am_se
        print(f"  AM-LSM: {am_lsm:.5f} ± {am_se:.5f}   "
              f"95%CI=[{am_lsm-ci95:.4f}, {am_lsm+ci95:.4f}]   ({lsm_cpu:.1f}s)")
        print(f"  EEP-LSM: {am_lsm - eu_mc:.5f}")

        # ── Step 5: B-Spline prices ───────────────────────────────────
        print(f"\n  {'N':>5}  {'AM-BSpl':>10}  {'EU-BSpl':>10}  "
              f"{'EEP':>8}  {'vs LSM':>10}  {'σ-dist':>8}  {'in CI?':>7}")
        print("  " + "-" * 62)

        bs_am, bs_eu = [], []
        for N in BSPLINE_N:
            N_tau = 4 * N
            am_bs = price_american_put(params, N=N, N_tau=N_tau)
            eu_bs = price_european(params, N=N, N_tau=N_tau)
            diff  = am_bs - am_lsm
            n_sig = diff / am_se
            in_ci = "YES" if abs(diff) < ci95 else "NO "
            bs_am.append(am_bs)
            bs_eu.append(eu_bs)
            print(f"  {N:>5}  {am_bs:>10.5f}  {eu_bs:>10.5f}  "
                  f"{am_bs-eu_bs:>8.5f}  {diff:>+10.5f}  {n_sig:>+8.1f}  {in_ci:>7}")

        all_results[label] = {
            "lsm": am_lsm, "lsm_se": am_se,
            "eu_mc": eu_mc, "eu_cos": eu_cos,
            "bs_am": bs_am, "bs_eu": bs_eu,
        }

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY: B-Spline (N=256) vs LSM-CGMY American Put")
    print(f"{'='*70}")
    print(f"  {'Set':>6}  {'BS AM(256)':>12}  {'LSM AM':>10}  {'95% CI':>20}  "
          f"{'Match?':>8}")
    for label, res in all_results.items():
        bs256 = res["bs_am"][-1]
        lsm   = res["lsm"]
        se    = res["lsm_se"]
        lo    = lsm - 1.96 * se
        hi    = lsm + 1.96 * se
        match = "YES" if lo <= bs256 <= hi else "NO "
        print(f"  {label:>6}  {bs256:>12.5f}  {lsm:>10.5f}  "
              f"[{lo:.4f}, {hi:.4f}]  {match:>8}")

    return all_results


if __name__ == "__main__":
    results = run()
