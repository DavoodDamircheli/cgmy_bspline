"""
Regenerate cgmy_bspline/results/table_calibration_insample.txt for the
Section 7 SPX calibration pipeline (3 real dates), reflecting the D2 fix in
calibration/make_figures.py (crisis 1w/6m/1y tenors filled from the 15%-
spread variant instead of left blank).

NOTE on the file this overwrites: the table_calibration_insample.txt
previously on disk was produced by cgmy_bspline/experiments/exp5_calibration.py
-- an unrelated, older synthetic-recovery demo (Step 11 of an earlier
roadmap; true/recovered C,G,M,Y on a synthetic surface, nothing to do with
the 2020-01-15 / 2020-03-18 / 2021-09-15 SPX calibration this session has
been investigating). It is not \input/\verbatiminput anywhere in
calibration_report.tex or summary.tex (those tables are hand-typed), so
overwriting it does not change any compiled report. This script replaces
its content with the actually-relevant Section 7 per-maturity in-sample
table.

Run:
    python calibration/make_insample_table.py
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from calibration.make_figures import _DATES, _TENORS, _tenor_col, _build_diagnostics

RES_DIR = ROOT / "cgmy_bspline" / "results"
OUT_PATH = RES_DIR / "table_calibration_insample.txt"


def _rmse_vp(abs_err_series) -> float:
    errs = abs_err_series.dropna().to_numpy(dtype=float)
    if len(errs) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errs ** 2))) * 100.0


def build_table() -> str:
    print("Building diagnostics (re-evaluates calibrated theta, no re-optimisation)...")
    diags = _build_diagnostics()

    lines = []
    sep = "=" * 78
    lines.append(sep)
    lines.append("=== Section 7 Calibration: In-Sample Report (per-maturity IV RMSE) ===")
    lines.append(sep)
    lines.append("")
    lines.append("Primary (5%-spread) surfaces; crisis date (2020-03-18) tenors marked *")
    lines.append("are filled from the 15%-spread variant because the primary filter left")
    lines.append("zero rows there (see debug_data_survival.py, D2). The calibrated theta")
    lines.append("was fit WITHOUT those starred quotes -- they are an out-of-sample check")
    lines.append("against the 5%-fit theta, not part of that fit's training data.")
    lines.append("")

    date_labels = [d for d, _, _ in _DATES]
    header = f"{'Tenor':>6} {'T (y)':>8}  " + "  ".join(f"{d:>22}" for d in date_labels)
    lines.append(header)
    lines.append(f"{'':>6} {'':>8}  " + "  ".join(f"{'RMSE(vp) / n quotes':>22}" for _ in date_labels))
    lines.append("-" * len(header))

    per_date_tenor_rmse = {d: {} for d in date_labels}
    per_date_tenor_n     = {d: {} for d in date_labels}
    per_date_fallback    = {}

    for date_str, _, _ in _DATES:
        row_label, df_diag, fallback_labels = diags[date_str]
        per_date_fallback[date_str] = set(fallback_labels)
        for T_val, grp in df_diag.groupby("T_round"):
            col = _tenor_col(float(T_val))
            if col is None:
                continue
            tenor_label = _TENORS[col][0]
            per_date_tenor_rmse[date_str][tenor_label] = _rmse_vp(grp["abs_err"])
            per_date_tenor_n[date_str][tenor_label] = int(grp["abs_err"].notna().sum())

    for tenor_label, T_lo, T_hi in _TENORS:
        T_mid = (T_lo + T_hi) / 2.0
        cells = []
        for date_str, _, _ in _DATES:
            rmse = per_date_tenor_rmse[date_str].get(tenor_label)
            n    = per_date_tenor_n[date_str].get(tenor_label)
            if rmse is None:
                cells.append(f"{'filtered':>22}")
            else:
                star = "*" if tenor_label in per_date_fallback[date_str] else ""
                cell = f"{rmse:.2f}{star} / {n}"
                cells.append(f"{cell:>22}")
        lines.append(f"{tenor_label:>6} {T_mid:>8.3f}  " + "  ".join(cells))

    lines.append("-" * len(header))

    # ── Aggregate RMSE: primary-only (5% filter, excludes fallback rows) vs
    #    including the fallback-filled crisis tenors ──────────────────────────
    for date_str, _, _ in _DATES:
        _, df_diag, fallback_labels = diags[date_str]
        fb_set = set(fallback_labels)

        is_fallback_row = df_diag["T_round"].apply(
            lambda t: (_TENORS[_tenor_col(float(t))][0] in fb_set) if _tenor_col(float(t)) is not None else False
        )
        primary_only = df_diag.loc[~is_fallback_row, "abs_err"]
        all_rows     = df_diag["abs_err"]

        rmse_primary = _rmse_vp(primary_only)
        rmse_all     = _rmse_vp(all_rows)
        n_primary    = int(primary_only.notna().sum())
        n_all        = int(all_rows.notna().sum())

        lines.append("")
        lines.append(f"  {date_str}:")
        lines.append(f"    All, primary 5% filter only:      "
                      f"RMSE={rmse_primary:6.2f} vp   n={n_primary}")
        if fb_set:
            lines.append(f"    All, incl. 15%-fallback tenors {sorted(fb_set)}: "
                          f"RMSE={rmse_all:6.2f} vp   n={n_all}")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def main():
    table_text = build_table()
    print()
    print(table_text)
    OUT_PATH.write_text(table_text + "\n")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
