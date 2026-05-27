import numpy as np
import pytest
from cgmy_bspline.tests.test_parameters import PARAMS_I, PARAMS_II
from cgmy_bspline.matrices import (
    mass_matrix,
    derivative_matrix,
    fractional_stiffness_matrices,
    operator_matrix,
)

N, p = 16, 3
h, x_min = 0.1, 0.0


def test_mass_symmetry():
    M = mass_matrix(N, p, h, x_min)
    M_arr = M.toarray()
    assert np.allclose(M_arr, M_arr.T, atol=1e-12), "M_h is not symmetric"


def test_mass_positive_definite():
    M = mass_matrix(N, p, h, x_min)
    eigs = np.linalg.eigvalsh(M.toarray())
    assert np.all(eigs > 0), f"M_h has non-positive eigenvalue: {eigs.min():.3e}"


def test_mass_bandwidth():
    M = mass_matrix(N, p, h, x_min)
    M_arr = M.toarray()
    n = N + p - 1
    for i in range(n):
        for j in range(n):
            if abs(i - j) >= p:
                assert M_arr[i, j] == pytest.approx(0.0, abs=1e-14), (
                    f"M_h[{i},{j}] = {M_arr[i,j]:.3e} but |i-j|={abs(i-j)} >= p={p}"
                )


def test_derivative_row_sum():
    D = derivative_matrix(N, p, h, x_min)
    D_arr = D.toarray()
    for row in range(5, N + p - 5):
        row_sum = D_arr[row, :].sum()
        assert abs(row_sum) < 1e-10, (
            f"D_h row {row} sum = {row_sum:.3e}, expected ~0"
        )


def test_fractional_stiffness_assembles():
    for label, params in [("PARAMS_I", PARAMS_I), ("PARAMS_II", PARAMS_II)]:
        K_G, K_M = fractional_stiffness_matrices(N, p, h, params, bandwidth=64)
        arr_G = K_G.toarray()
        print(f"\n{label} K_G_minus[0,0] = {arr_G[0,0]:.6e}")
        print(f"{label} K_G_minus[0,5] = {arr_G[0,5]:.6e}")
        assert K_G.shape == (N + p - 1, N + p - 1)
        assert K_M.shape == (N + p - 1, N + p - 1)


def test_operator_matrix_assembles():
    result = operator_matrix(N, p, h, x_min, PARAMS_I, bandwidth=64)
    A = result['A_h']
    print(f"\nA_h shape: {A.shape}, nnz: {A.nnz}")
    assert A.shape == (N + p - 1, N + p - 1)
    assert A.nnz > 0
    assert 'M_h' in result and 'D_h' in result
    assert 'K_minus' in result and 'K_plus' in result
