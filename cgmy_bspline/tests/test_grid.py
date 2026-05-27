import math
import numpy as np
import pytest
from cgmy_bspline.grid import make_grid, make_grid_manual
from cgmy_bspline.tests.test_parameters import PARAMS_I, PARAMS_II


def test_params_i_grid():
    g = make_grid(PARAMS_I, N=64)
    log_K = math.log(100.0)
    assert g["x_min"] == log_K - 10 / 5.0
    assert g["x_max"] == log_K + 10 / 5.0
    assert g["x_min"] < log_K < g["x_max"]
    assert g["h"] == (g["x_max"] - g["x_min"]) / 64
    assert len(g["nodes"]) == 65


def test_params_ii_grid():
    g = make_grid(PARAMS_II, N=64)
    log_K = math.log(100.0)
    h = g["h"]
    # log_K must sit exactly on a grid node after the nested snap
    frac = (log_K - g["x_min"]) / h
    assert abs(frac - round(frac)) < 1e-10, f"log_K not on grid node: frac={frac}"
    # x_min is fixed so that grids with N=64,128,256,512 are nested;
    # domain endpoints stay within one h_base cell of the target values
    h_base = (10 / 2.0 + 10 / 3.5) / 64          # cell size at coarsest N
    assert abs(g["x_min"] - (log_K - 10 / 2.0)) <= h_base
    assert abs(g["x_max"] - (log_K + 10 / 3.5)) <= h_base


def test_make_grid_manual():
    g = make_grid_manual(-5.0, 5.0, 128)
    assert g["h"] == 10.0 / 128
    assert g["nodes"][0] == -5.0
    assert g["nodes"][-1] == 5.0
