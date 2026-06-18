"""
Generate paper figures from calibration and benchmark results.

Loads calibrated parameters from calibration_results.csv, re-evaluates the
B-spline pricer at those parameters (N=64, no re-optimisation), and produces:

    fig_iv_smiles.png       3×5 grid: market vs model IV smiles per date/tenor
    fig_residuals.png       residual (model−market) IV vs log-moneyness, per date
    fig_speed_comparison.png bar chart of calibration wall-time (B-Spline vs GL)

All PNGs are written to cgmy_bspline/results/figures/.
"""

import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
RES_DIR  = ROOT / "cgmy_bspline" / "results"
FIG_DIR  = RES_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from calibration.calibrate import _load_surface
from calibration.pricer    import make_cached_pricer
from calibration.params    import CGMYParams
from calibration.objective import iv_rmse_and_max_err

# ── global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── dates to process (primary 5%-spread surfaces) ────────────────────────────
_DATES = [
    # (date_str, row_label, csv_name)
    ("2020-01-15", "Low-vol  (Jan 2020)", "surface_2020-01-15_spread05.csv"),
    ("2020-03-18", "Crisis   (Mar 2020)", "surface_2020-03-18_spread05.csv"),
    ("2021-09-15", "Recovery (Sep 2021)", "surface_2021-09-15_spread05.csv"),
]

# Nominal tenor column definitions: (label, T_lo, T_hi)
_TENORS = [
    ("1w",  0.010, 0.040),
    ("1m",  0.060, 0.120),
    ("3m",  0.200, 0.300),
    ("6m",  0.400, 0.600),
    ("1y",  0.800, 1.100),
]

# Maturity-colour mapping (up to 5 maturities)
_MAT_COLORS = plt.get_cmap("tab10").colors[:5]

# Pricer settings for diagnostic evaluation (no re-optimisation)
_N_EVAL  = 64
_N_TAU   = 10
_BW      = 2 * _N_EVAL


# ── Step 1: load calibrated parameters ───────────────────────────────────────

def _load_calibrated_params() -> dict:
    """Return {date_str: (C, G, M, Y)} for primary 5%-spread runs."""
    df = pd.read_csv(RES_DIR / "calibration_results.csv", dtype={"spread_filter": str})
    df = df[df["spread_filter"] == "05"].set_index("date")
    out = {}
    for d, row in df.iterrows():
        out[d] = (float(row["C"]), float(row["G"]),
                  float(row["M"]), float(row["Y"]))
    return out


# ── Step 2: compute per-date diagnostic DataFrames ───────────────────────────

def _build_diagnostics() -> dict:
    """
    For each date load the surface CSV, build an N=64 cached pricer, evaluate
    the calibrated theta, and return a dict {date_str: df_diag}.

    df_diag columns: date, T, K, log_moneyness, market_iv, model_iv, abs_err
    (IVs in decimal, matching the surface CSV convention).
    """
    params = _load_calibrated_params()
    diags  = {}

    for date_str, row_label, csv_name in _DATES:
        print(f"  [{date_str}] loading surface and building pricer...")
        csv_path = DATA_DIR / csv_name
        surface, S0, rates = _load_surface(str(csv_path))

        gkw    = dict(N=_N_EVAL, N_tau=_N_TAU,
                      domain_half_width=2.0, bandwidth=_BW)
        pricer = make_cached_pricer(surface, S0, rates, grid_kwargs=gkw)

        C, G, M, Y = params[date_str]
        # r, q, S0 in CGMYParams are not used by iv_rmse_and_max_err
        # (it extracts only C,G,M,Y); use mean rates as placeholders.
        r_avg = float(np.mean([v[0] for v in rates.values()]))
        q_avg = float(np.mean([v[1] for v in rates.values()]))
        theta = CGMYParams(C=C, G=G, M=M, Y=Y, r=r_avg, q=q_avg, S0=S0)

        _, _, df_diag = iv_rmse_and_max_err(theta, surface, pricer)
        df_diag["T_round"] = df_diag["T"].round(4)
        diags[date_str] = (row_label, df_diag)
        print(f"    {len(df_diag)} quotes, "
              f"{df_diag['T'].nunique()} maturities, "
              f"model IV NaN: {df_diag['model_iv'].isna().sum()}")

    return diags


def _tenor_col(T: float) -> int | None:
    """Map a maturity T (years) to a column index 0..4, or None."""
    for col_idx, (_, lo, hi) in enumerate(_TENORS):
        if lo <= T <= hi:
            return col_idx
    return None


# ── Figure 1: IV smile grid ───────────────────────────────────────────────────

def fig_iv_smiles(diags: dict):
    """3×5 grid of market (scatter) vs model (line) IV smiles."""
    n_rows, n_cols = len(_DATES), len(_TENORS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(15, 8),
        sharey=False, sharex=False,
        constrained_layout=True,
    )

    # Assign one colour per tenor column (shared across rows)
    col_colors = [_MAT_COLORS[c] for c in range(n_cols)]

    for row_idx, (date_str, row_label, _) in enumerate(_DATES):
        row_label_txt, df = diags[date_str]

        # Collect data organised by tenor
        tenor_data = {c: None for c in range(n_cols)}
        for T_val, grp in df.groupby("T_round", sort=True):
            col_idx = _tenor_col(float(T_val))
            if col_idx is not None:
                grp_sorted = grp.sort_values("log_moneyness")
                tenor_data[col_idx] = grp_sorted

        for col_idx, (tenor_label, _, _) in enumerate(_TENORS):
            ax = axes[row_idx, col_idx]
            grp = tenor_data[col_idx]

            if grp is None:
                ax.set_visible(False)
                continue

            lm = grp["log_moneyness"].to_numpy()
            mkt  = grp["market_iv"].to_numpy() * 100.0
            mod  = grp["model_iv"].to_numpy() * 100.0
            color = col_colors[col_idx]

            # Market: scatter
            ax.scatter(lm, mkt,
                       s=12, color="0.5", zorder=3,
                       label="Market" if (row_idx == 0 and col_idx == 0) else None)
            # Model: line (over finite values only)
            finite = np.isfinite(mod)
            if finite.sum() > 1:
                ax.plot(lm[finite], mod[finite],
                        color=color, lw=1.5, zorder=4,
                        label="Model" if (row_idx == 0 and col_idx == 0) else None)

            ax.axvline(0, color="0.8", lw=0.6, ls="--", zorder=1)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

            # Top-row: tenor label
            if row_idx == 0:
                ax.set_title(tenor_label, fontweight="bold")

            # Left-column: row label (rotated)
            if col_idx == 0:
                ax.set_ylabel(f"IV (%)\n{row_label_txt}", fontsize=7)
            else:
                ax.set_ylabel("")

            # Bottom-row: x-axis label
            if row_idx == n_rows - 1:
                ax.set_xlabel("log-moneyness")
            else:
                ax.set_xlabel("")

    # Shared legend (top-left occupied cell)
    # Build proxy handles
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        plt.scatter([], [], s=12, color="0.5", label="Market"),
        Line2D([0], [0], color=col_colors[2], lw=1.5, label="Model"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), framealpha=0.9)

    fig.suptitle("Market vs Model Implied Volatility Smiles (B-Spline Galerkin + CN)",
                 fontsize=10, fontweight="bold")

    out = FIG_DIR / "fig_iv_smiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}  ({out.stat().st_size/1024:.0f} KB)")
    return out


# ── Figure 2: residuals ───────────────────────────────────────────────────────

def fig_residuals(diags: dict):
    """One subplot per date: (model_iv - market_iv) in vol-pts vs log-moneyness."""
    n_dates = len(_DATES)
    fig, axes = plt.subplots(1, n_dates, figsize=(13, 4),
                             constrained_layout=True, sharey=False)

    for ax_idx, (date_str, row_label, _) in enumerate(_DATES):
        ax = axes[ax_idx]
        row_label_txt, df = diags[date_str]

        resid_vp = (df["model_iv"] - df["market_iv"]) * 100.0
        T_vals   = sorted(df["T_round"].unique())

        for t_idx, T_val in enumerate(T_vals):
            col_idx = _tenor_col(float(T_val))
            color = _MAT_COLORS[col_idx] if col_idx is not None else f"C{t_idx}"
            mask  = df["T_round"] == T_val
            lm    = df.loc[mask, "log_moneyness"].to_numpy()
            res   = resid_vp[mask].to_numpy()
            finite = np.isfinite(res)

            # Assign tenor label for legend
            if col_idx is not None:
                tenor_label = _TENORS[col_idx][0]
            else:
                tenor_label = f"T={T_val:.2f}"

            ax.scatter(lm[finite], res[finite],
                       s=14, color=color, alpha=0.75,
                       label=tenor_label, zorder=3)

        ax.axhline(0, color="0.3", lw=0.8, ls="--", zorder=2)
        ax.set_xlabel("log-moneyness")
        ax.set_ylabel("Model − Market IV (vol pts)" if ax_idx == 0 else "")
        ax.set_title(row_label_txt, fontweight="bold")
        ax.legend(loc="upper left", title="Tenor", framealpha=0.85)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f"))

    fig.suptitle("IV Residuals: Model − Market (B-Spline Galerkin + CN)",
                 fontsize=10, fontweight="bold")

    out = FIG_DIR / "fig_residuals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}  ({out.stat().st_size/1024:.0f} KB)")
    return out


# ── Figure 3: speed comparison ────────────────────────────────────────────────

def fig_speed_comparison():
    """Bar chart: B-Spline vs GL calibration wall-time with alpha annotated."""
    bench_csv = RES_DIR / "benchmark_results.csv"
    df = pd.read_csv(bench_csv)

    methods  = df["method"].tolist()
    times    = df["total_time_sec"].tolist()
    n_de     = df["N_DE"].tolist()
    n_lb     = df["N_lbfgs"].tolist()
    iv_rmses = (df["iv_rmse"] * 100).tolist()

    alpha = times[1] / times[0]  # GL / B-Spline

    # Bar colours
    bar_colors = ["#2166AC", "#D6604D"]   # blue for B-Spline, red-orange for GL

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    x_pos = [0, 1]
    bars  = ax.bar(x_pos, times, width=0.5,
                   color=bar_colors, edgecolor="0.3", linewidth=0.7)

    # Value labels on bars
    for bar, t, rmse in zip(bars, times, iv_rmses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                t + 1.5,
                f"{t:.0f} s\n(IV RMSE {rmse:.2f} vp)",
                ha="center", va="bottom", fontsize=8)

    # Alpha annotation (arrow between bars)
    y_mid = (times[0] + times[1]) / 2.0
    ax.annotate(
        "",
        xy=(x_pos[1], times[1] + 4),
        xytext=(x_pos[0], times[1] + 4),
        arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.2),
    )
    ax.text(0.5, times[1] + 8,
            r"$\alpha = T_{\mathrm{GL}}/T_{\mathrm{BS}} = $" + f"{alpha:.2f}",
            ha="center", va="bottom", fontsize=9, color="0.2")

    # x-axis labels with N info
    x_labels = [
        f"B-Spline Galerkin\n(N={n_de[0]}/{n_lb[0]})",
        f"GL Finite-Diff.\n(N={n_de[1]}/{n_lb[1]})",
    ]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Calibration wall-time (s)")
    ax.set_ylim(0, max(times) * 1.35)
    ax.set_title("Calibration Speed: B-Spline vs GL (accuracy-matched)\n"
                 "2020-03-18 crisis surface, identical optimizer settings",
                 fontsize=9, fontweight="bold")

    out = FIG_DIR / "fig_speed_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}  ({out.stat().st_size/1024:.0f} KB)")
    return out


# ── Self-test ─────────────────────────────────────────────────────────────────

def _selftest(paths: list):
    """Confirm files exist, are non-trivially sized, and smile figure is ≥1200×800."""
    print()
    print("=" * 60)
    print("SELF-TEST")
    print("=" * 60)
    all_pass = True

    for p in paths:
        p = pathlib.Path(p)
        exists = p.exists()
        size   = p.stat().st_size if exists else 0
        ok_exists = exists and size > 10_000   # >10 KB minimum
        tag = "OK" if ok_exists else "FAIL"
        if not ok_exists:
            all_pass = False
        print(f"  {p.name:<30}  exists={exists}  size={size/1024:.1f} KB  [{tag}]")

    # Pixel dimension check on smile figure
    smile_path = FIG_DIR / "fig_iv_smiles.png"
    try:
        from PIL import Image
        with Image.open(smile_path) as im:
            w, h = im.size
        ok_dims = w >= 1200 and h >= 800
        tag = "OK" if ok_dims else "FAIL"
        if not ok_dims:
            all_pass = False
        print(f"  fig_iv_smiles.png  dimensions: {w}×{h} px  (≥1200×800 required)  [{tag}]")
    except ImportError:
        print("  PIL not available — skipping pixel dimension check")
    except Exception as e:
        print(f"  PIL check failed: {e}")
        all_pass = False

    print()
    print("SELF-TEST:", "PASS" if all_pass else "FAIL")
    print("=" * 60)
    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("make_figures.py  —  generating calibration figures")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 60)

    print("\nStep 1: Building diagnostic DataFrames (N=64 pricer)...")
    diags = _build_diagnostics()

    print("\nStep 2: fig_iv_smiles.png (3×5 IV smile grid)...")
    p1 = fig_iv_smiles(diags)

    print("\nStep 3: fig_residuals.png (IV residuals per date)...")
    p2 = fig_residuals(diags)

    print("\nStep 4: fig_speed_comparison.png (B-Spline vs GL wall-time)...")
    p3 = fig_speed_comparison()

    _selftest([p1, p2, p3])


if __name__ == "__main__":
    main()
