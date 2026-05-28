# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Implements a B-Spline Galerkin method for pricing options under the CGMY (Carr–Géman–Madan–Yor) tempered stable process, following the paper *"A B-Spline Galerkin Method for Option Pricing under the CGMY Tempered Stable Process"*. The PDE is a fractional backward FPDE solved with Crank–Nicolson time-stepping.

## Environment

All work happens inside the project-local venv — never the system Python.

```bash
source .venv/bin/activate
```

The package is installed in editable mode (`pip install -e .`), so source changes are reflected immediately.

## Commands

```bash
# run all tests
.venv/bin/python -m pytest cgmy_bspline/tests/ -v

# run a single test file
.venv/bin/python -m pytest cgmy_bspline/tests/test_bspline.py -v

# run a single test by name
.venv/bin/python -m pytest cgmy_bspline/tests/test_bspline.py::test_partition_of_unity -v

# run individual experiment scripts
.venv/bin/python cgmy_bspline/experiments/run_european_put.py
.venv/bin/python cgmy_bspline/experiments/run_convergence.py
.venv/bin/python cgmy_bspline/experiments/run_manufactured.py
.venv/bin/python cgmy_bspline/experiments/run_method_comparison.py
.venv/bin/python cgmy_bspline/experiments/run_surface.py
.venv/bin/python cgmy_bspline/experiments/run_calibration.py

# run all paper experiments at once (saves CSVs + PNGs to cgmy_bspline/results/)
.venv/bin/python cgmy_bspline/experiments/run_all_paper_experiments.py
```

## Architecture

The solver pipeline flows in this order:

```
parameters.py → grid.py → bspline_basis.py → connection_coefficients.py
                                                      ↓
                                 toeplitz.py → matrices.py → boundary.py
                                                      ↓
                              projection.py → solver_cn.py → payoff.py / greeks.py
                                                      ↓
                                           calibration.py
```

| Module | Role |
|--------|------|
| `parameters.py` | CGMY model parameters (C, G, M, Y), risk-free rate, maturity, strike |
| `grid.py` | Uniform/non-uniform spatial grid on the log-price domain |
| `bspline_basis.py` | B-spline basis construction (Cox–de Boor), evaluation, knot vectors |
| `connection_coefficients.py` | Fractional connection coefficients for the CGMY integral operator |
| `toeplitz.py` | Toeplitz/circulant structure exploitation for the integral operator matrix |
| `matrices.py` | Assembly of stiffness, mass, and jump matrices from basis integrals |
| `boundary.py` | Boundary condition enforcement (Dirichlet at domain edges) |
| `projection.py` | L² projection of the payoff onto the B-spline basis at t = T |
| `solver_cn.py` | Crank–Nicolson time-stepping loop; three solver modes (see below) |
| `payoff.py` | Payoff functions (European put/call, etc.) |
| `greeks.py` | Delta, Gamma, Black-Scholes implied vol, and IV surface generation |
| `calibration.py` | `CGMYCalibrator`: calibration of CGMY parameters to an IV surface |
| `experiments/` | Standalone driver scripts; each produces printed tables, CSVs, or figures |

The test suite in `cgmy_bspline/tests/` mirrors the module layout one-to-one.

## Solver modes (`solver_cn.py`)

`solve_cn(params, N, p, N_tau, bandwidth, mode)` supports three modes:

| Mode | Description |
|------|-------------|
| `'banded'` (default) | Sparse banded matrices; direct LU at each time step. Fast for moderate N. |
| `'toeplitz'` | Full Toeplitz operators; GMRES with no preconditioner. Best accuracy. |
| `'toeplitz_precond'` | Full Toeplitz + banded LU preconditioner. Best convergence per iteration. |

The `bandwidth` parameter controls how many off-diagonal lags are included in the sparse banded approximation (and the preconditioner in `toeplitz_precond` mode). Use `bandwidth=2*N` for full coverage.

## Experiment scripts

| Script | Paper section | Output |
|--------|--------------|--------|
| `run_manufactured.py` | §6.1 manufactured convergence | printed table |
| `run_convergence.py` | §6.2 EU put convergence | printed table |
| `run_method_comparison.py` | §6.3 Table 6: solver mode comparison | printed table |
| `run_surface.py` | §6.4 IV surface + Greeks | ATM IV table, Greeks table, `iv_surface_params_ii.csv` |
| `run_calibration.py` | §6.5 calibration demo | printed calibration report |
| `run_european_put.py` | reference pricer | single put price via COS method |
| `run_all_paper_experiments.py` | all of the above | CSVs + PNGs in `cgmy_bspline/results/` |

Generated outputs (CSVs, PNGs) are written to `cgmy_bspline/results/` by `run_all_paper_experiments.py` and are excluded from version control.

## Key numerical details

- The CGMY operator is non-local; its Galerkin matrix is dense but has Toeplitz structure that `toeplitz.py` exploits for O(N log N) matrix–vector products.
- The fractional order `Y ∈ (0, 2)` controls the singularity of the Lévy measure; branching on `Y < 1` vs `Y ≥ 1` appears in `connection_coefficients.py`.
- Crank–Nicolson is unconditionally stable for this problem; the linear system at each time step is solved with a direct solver (scipy sparse LU) unless overridden.
- `lambda0` in `parameters.py` returns just `r` (the risk-free rate), not the full kill rate.
- `fractional_stiffness_matrices` truncates at ±`half_bw` (= `bandwidth // 2`), not ±`bandwidth`.
- `CGMYCalibrator` caches M_h, D_h, payoff projection, and Toeplitz sparsity across all optimizer calls; only IFFT arrays are recomputed when (G, M, Y) change.
- Near-ATM IV surfaces (strikes 90–110, short maturities) are nearly flat under CGMY; multiple parameter sets can produce equivalent smiles — in-sample RMSE is the correct calibration quality metric.
