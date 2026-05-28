"""
Master script: run every experiment needed for the paper and save all outputs.

Experiments
-----------
A  Tables 2 & 3 — manufactured-solution convergence
B  Tables 4 & 5 — European-put convergence vs COS
C  Table 6      — method comparison (banded / Toeplitz / Toeplitz+precond)
D              — Greeks (Delta, Gamma) CSV + figure
E              — Implied-volatility surface CSV + figure

All outputs go to  <package_root>/results/
"""
import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')               # no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Output directory (robust regardless of cwd) ──────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_PKG        = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_PKG, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _rpath(filename):
    return os.path.join(RESULTS_DIR, filename)


# ── Library imports (no side-effects) ────────────────────────────────────────
from cgmy_bspline.tests.test_parameters import PARAMS_I, PARAMS_II
from cgmy_bspline.parameters            import CGMYParams
from cgmy_bspline.solver_cn             import solve_cn
from cgmy_bspline.projection            import eval_solution
from cgmy_bspline.grid                  import make_grid
from cgmy_bspline.greeks                import compute_greeks, iv_surface
from cgmy_bspline.experiments.run_european_put import cos_price_put

# ── Shared constants ──────────────────────────────────────────────────────────
P   = 3
NS  = [64, 128, 256, 512]
NU  = 1.5      # manufactured-solution exponent


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _write_csv(path, header, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  → {path}")


def _latex_sep(cols):
    return '\\hline'


def _nan_str(x, fmt):
    return '--' if (x is None or (isinstance(x, float) and np.isnan(x))) else format(x, fmt)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT A — Manufactured convergence (Tables 2 & 3)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_A():
    print('\n' + '═' * 70)
    print('  EXPERIMENT A: Manufactured-solution convergence (Tables 2 & 3)')
    print('═' * 70)

    table_defs = [
        ('PARAMS_I  (Y=0.5)',  PARAMS_I,  'manufactured_convergence_Y05.csv', 2),
        ('PARAMS_II (Y=1.5)', PARAMS_II, 'manufactured_convergence_Y15.csv', 3),
    ]

    for label, params, csv_name, table_num in table_defs:
        print(f'\n  Table {table_num}: {label}')
        print(f'  Expected asymptotic rate: {P - params.Y / 2:.2f}')
        print(f'  {"N":>5}  {"h":>10}  {"dt":>10}  {"L2_err":>12}  {"rate":>6}  {"CPU":>6}')
        print(f'  {"-" * 55}')

        rows      = []
        prev_err  = None
        latex_rows = []

        for N in NS:
            result = solve_cn(params, N=N, p=P, N_tau=N,
                              bandwidth=128, manufactured=True, nu=NU)
            g   = make_grid(params, N)
            h   = g['h']
            dt  = params.T / N
            err = result['error_L2']
            cpu = result['cpu_time']

            rate = (np.log2(prev_err / err)
                    if (prev_err is not None and prev_err > 0 and err > 0)
                    else float('nan'))

            print(f'  {N:>5}  {h:>10.4e}  {dt:>10.4e}  {err:>12.6e}'
                  f'  {_nan_str(rate, ".2f"):>6}  {cpu:>6.1f}')

            rows.append([N, h, dt, err, rate, cpu])
            latex_rows.append(
                f'  {N} & {h:.4e} & {dt:.4e} & {err:.2e} & '
                f'{_nan_str(rate, ".2f")} & {cpu:.1f} \\\\'
            )
            prev_err = err

        _write_csv(_rpath(csv_name),
                   ['N', 'h', 'dt', 'L2_error', 'rate', 'CPU_time'],
                   rows)
        print(f'\n  LaTeX rows for Table {table_num}:')
        for row in latex_rows:
            print(row)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT B — European-put convergence (Tables 4 & 5)
# ══════════════════════════════════════════════════════════════════════════════

def _l2_put_error(result, params, g, n_pts=50):
    """Approximate L2 error of the put solution vs COS using n_pts interior sample."""
    x_lo = g['x_min'] + g['h']
    x_hi = g['x_max'] - g['h']
    x_s  = np.linspace(x_lo, x_hi, n_pts)
    S_s  = np.exp(x_s)
    V_b  = eval_solution(result['c_final'], g, P, x_s)
    V_c  = np.array([cos_price_put(params, float(S), n_cos=512) for S in S_s])
    h_s  = (x_hi - x_lo) / (n_pts - 1)
    return float(np.sqrt(h_s * np.sum((V_b - V_c) ** 2)))


def experiment_B():
    print('\n' + '═' * 70)
    print('  EXPERIMENT B: European-put convergence (Tables 4 & 5)')
    print('═' * 70)

    table_defs = [
        ('PARAMS_I  (Y=0.5)',  PARAMS_I,  'put_convergence_set_i.csv',  4),
        ('PARAMS_II (Y=1.5)', PARAMS_II, 'put_convergence_set_ii.csv', 5),
    ]

    for label, params, csv_name, table_num in table_defs:
        S0  = params.K
        x0  = np.log(S0)
        ref = cos_price_put(params, S0, n_cos=2048)

        print(f'\n  Table {table_num}: {label}   COS ref = {ref:.6f}')
        print(f'  {"N":>5}  {"h":>10}  {"dt":>10}  {"price":>10}  '
              f'{"err@S0":>10}  {"L2_err":>10}  {"rate":>6}  {"CPU":>6}')
        print(f'  {"-" * 72}')

        rows       = []
        prev_err   = None
        latex_rows = []

        for N in NS:
            t0     = time.time()
            result = solve_cn(params, N=N, p=P, N_tau=N, bandwidth=2 * N)
            g      = make_grid(params, N)
            price  = float(eval_solution(result['c_final'], g, P,
                                         np.array([x0]))[0])
            err_s0 = abs(price - ref)
            L2_err = _l2_put_error(result, params, g, n_pts=50)
            cpu    = time.time() - t0

            rate = (np.log2(prev_err / err_s0)
                    if (prev_err is not None and prev_err > 0 and err_s0 > 0)
                    else float('nan'))

            print(f'  {N:>5}  {g["h"]:>10.4e}  {params.T/N:>10.4e}  {price:>10.6f}  '
                  f'{err_s0:>10.2e}  {L2_err:>10.2e}  {_nan_str(rate, ".2f"):>6}  {cpu:>6.1f}')

            rows.append([N, g['h'], params.T / N, price, err_s0, L2_err,
                         rate if not np.isnan(rate) else '', cpu])
            latex_rows.append(
                f'  {N} & {g["h"]:.4e} & {params.T/N:.4e} & {price:.6f} & '
                f'{err_s0:.2e} & {L2_err:.2e} & {_nan_str(rate, ".2f")} & {cpu:.1f} \\\\'
            )
            prev_err = err_s0

        _write_csv(_rpath(csv_name),
                   ['N', 'h', 'dt', 'price_S0', 'err_S0', 'L2_err', 'rate', 'CPU_time'],
                   rows)
        print(f'\n  LaTeX rows for Table {table_num}:')
        for row in latex_rows:
            print(row)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT C — Method comparison (Table 6)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_C():
    print('\n' + '═' * 70)
    print('  EXPERIMENT C: Method comparison (Table 6)')
    print('═' * 70)

    variants = [
        dict(label='A', mode='banded',           bandwidth=64),
        dict(label='B', mode='banded',           bandwidth=128),
        dict(label='C', mode='toeplitz',         bandwidth=64),
        dict(label='D', mode='toeplitz_precond', bandwidth=64),
    ]
    params = PARAMS_II
    N      = 512
    N_tau  = 100
    S0     = params.K
    x0     = np.log(S0)
    ref    = cos_price_put(params, S0, n_cos=2048)

    print(f'  COS reference: {ref:.6f}')
    print(f'  {"Var":>3}  {"Mode":<20}  {"bw":>4}  {"price":>10}  '
          f'{"err":>10}  {"CPU":>6}  {"GMRES/step":>10}')
    print(f'  {"-" * 68}')

    rows       = []
    latex_rows = []
    for v in variants:
        result   = solve_cn(params, N=N, p=P, N_tau=N_tau,
                            bandwidth=v['bandwidth'], mode=v['mode'])
        g        = make_grid(params, N)
        price    = float(eval_solution(result['c_final'], g, P, np.array([x0]))[0])
        err      = abs(price - ref)
        cpu      = result['cpu_time']
        iters    = result['gmres_iters']
        bw_str   = 'full' if v['mode'] == 'toeplitz' else str(v['bandwidth'])
        iter_str = f"{iters:.1f}" if iters is not None else 'N/A'

        print(f'  {v["label"]:>3}  {v["mode"]:<20}  {bw_str:>4}  {price:>10.6f}  '
              f'{err:>10.2e}  {cpu:>6.1f}  {iter_str:>10}')

        rows.append([v['label'], v['mode'], bw_str, price, err, cpu,
                     iters if iters is not None else ''])
        latex_rows.append(
            f'  {v["label"]} & {v["mode"]} & {bw_str} & {price:.6f} & '
            f'{err:.2e} & {cpu:.1f} & {iter_str} \\\\'
        )

    _write_csv(_rpath('method_comparison.csv'),
               ['variant', 'mode', 'bandwidth', 'price_S0', 'err_S0',
                'CPU_time', 'GMRES_iters'],
               rows)
    print(f'\n  LaTeX rows for Table 6:')
    for row in latex_rows:
        print(row)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT D — Greeks
# ══════════════════════════════════════════════════════════════════════════════

def experiment_D():
    print('\n' + '═' * 70)
    print('  EXPERIMENT D: Greeks  (PARAMS_II, N=256, T=0.2)')
    print('═' * 70)

    params  = PARAMS_II
    N       = 256
    N_tau   = 100

    result  = solve_cn(params, N=N, p=P, N_tau=N_tau, bandwidth=2 * N)
    g       = make_grid(params, N)

    S_eval  = np.linspace(70, 130, 300)
    x_eval  = np.log(S_eval)
    Delta, Gamma = compute_greeks(result['c_final'], g, P, x_eval)

    # ── CSV ──────────────────────────────────────────────────────────────────
    rows = [[f'{S:.6f}', f'{D:.8f}', f'{G:.8f}']
            for S, D, G in zip(S_eval, Delta, Gamma)]
    _write_csv(_rpath('greeks_params_ii.csv'), ['S', 'Delta', 'Gamma'], rows)

    # ── Print selected rows ───────────────────────────────────────────────────
    print(f'\n  {"S":>8}  {"Delta":>12}  {"Gamma":>12}')
    print(f'  {"-" * 36}')
    for S_tgt in [75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125]:
        idx = int(np.argmin(np.abs(S_eval - S_tgt)))
        print(f'  {S_eval[idx]:>8.2f}  {Delta[idx]:>12.6f}  {Gamma[idx]:>12.6f}')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('CGMY Greeks — PARAMS_II (Y=1.5, T=0.2)', fontsize=13)

    ax1.plot(S_eval, Delta, color='steelblue', linewidth=1.8)
    ax1.axvline(params.K, color='grey', linestyle='--', linewidth=0.8, label='ATM')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('Spot  S', fontsize=11)
    ax1.set_ylabel('Delta', fontsize=11)
    ax1.set_title('Delta (European Put)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(S_eval, Gamma, color='darkorange', linewidth=1.8)
    ax2.axvline(params.K, color='grey', linestyle='--', linewidth=0.8, label='ATM')
    ax2.set_xlabel('Spot  S', fontsize=11)
    ax2.set_ylabel('Gamma', fontsize=11)
    ax2.set_title('Gamma (European Put)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = _rpath('greeks_params_ii.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {fig_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT E — Implied-volatility surface
# ══════════════════════════════════════════════════════════════════════════════

def experiment_E():
    print('\n' + '═' * 70)
    print('  EXPERIMENT E: Implied-volatility surface  (PARAMS_II, N=256)')
    print('═' * 70)

    params     = PARAMS_II
    N          = 256
    strikes    = np.linspace(80, 120, 40)
    maturities = [1 / 12, 3 / 12, 6 / 12, 1.0]

    print('  Computing IV surface (4 maturities × 40 strikes)...')
    iv_mat = iv_surface(params, N=N, p=P, strikes=strikes,
                        maturities=maturities, N_tau_per_unit=200,
                        bandwidth=2 * N)

    # ── ATM summary ───────────────────────────────────────────────────────────
    atm_idx = int(np.argmin(np.abs(strikes - params.K)))
    print(f'\n  ATM IV (nearest strike {strikes[atm_idx]:.2f}):')
    print(f'  {"Maturity":>10}  {"ATM IV (%)":>12}')
    for i, T_i in enumerate(maturities):
        iv_atm = iv_mat[i, atm_idx]
        pct    = f'{iv_atm * 100:.2f}%' if not np.isnan(iv_atm) else 'N/A'
        print(f'  {T_i:>10.4f}  {pct:>12}')

    # ── CSV ──────────────────────────────────────────────────────────────────
    rows = []
    for i, T_i in enumerate(maturities):
        for j, K_j in enumerate(strikes):
            rows.append([f'{T_i:.6f}', f'{K_j:.4f}', f'{iv_mat[i, j]:.6f}'])
    _write_csv(_rpath('iv_surface_params_ii.csv'),
               ['maturity', 'strike', 'iv'], rows)

    # ── Figure: heatmap ───────────────────────────────────────────────────────
    T_arr = np.array(maturities)
    K_arr = strikes

    fig, ax = plt.subplots(figsize=(9, 5))
    iv_pct  = iv_mat * 100    # convert to percentage

    cf = ax.contourf(K_arr, T_arr, iv_pct, levels=20, cmap='RdYlGn_r')
    cs = ax.contour(K_arr, T_arr, iv_pct, levels=10, colors='k',
                    linewidths=0.5, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f%%')
    cbar = fig.colorbar(cf, ax=ax, label='Implied Vol (%)')
    ax.set_xlabel('Strike  K', fontsize=11)
    ax.set_ylabel('Maturity  T (years)', fontsize=11)
    ax.set_title('CGMY Implied-Volatility Surface — PARAMS_II (Y=1.5)', fontsize=12)
    ax.axvline(params.K, color='white', linestyle='--', linewidth=1.2, label='ATM')
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    fig_path = _rpath('iv_surface_params_ii.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {fig_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_total = time.time()

    experiment_A()
    experiment_B()
    experiment_C()
    experiment_D()
    experiment_E()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print('\n' + '═' * 70)
    print(f'  All experiments complete in {elapsed:.0f} s.')
    print(f'  Files saved to  {RESULTS_DIR}/')
    print('═' * 70)

    files = sorted(os.listdir(RESULTS_DIR))
    for fname in files:
        fpath = os.path.join(RESULTS_DIR, fname)
        size  = os.path.getsize(fpath)
        unit  = 'KB' if size >= 1024 else 'B'
        sz    = size // 1024 if size >= 1024 else size
        print(f'  {fname:<45}  {sz:>5} {unit}')
