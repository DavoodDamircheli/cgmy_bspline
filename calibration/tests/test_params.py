"""
Self-tests for calibration.params.

Run:  .venv/bin/python -m pytest calibration/tests/test_params.py -v

Each pytest test also prints an explicit PASS / FAIL banner so results are
visible even when running as a plain script.
"""

import math
import numpy as np
import pytest
from scipy.special import gamma as sc_gamma

from calibration.params import (
    CGMYParams,
    inverse_transform,
    lambda0,
    omega_S,
    transform_to_unconstrained,
)


# ---------------------------------------------------------------------------
# Independent helper used ONLY in this test file — derived from the
# Lévy-Khintchine characteristic exponent, NOT from omega_S().
# ---------------------------------------------------------------------------


def _psi_char_exp(C: float, G: float, M: float, Y: float, xi: complex) -> complex:
    """CGMY characteristic exponent at complex xi:

        Psi(xi) = C * Gamma(-Y) * [(M - i*xi)^Y - M^Y + (G + i*xi)^Y - G^Y]

    This is an INDEPENDENT implementation of the characteristic exponent.
    It is used in part (a) only; the omega_S function in params.py must NOT
    be called from here (that would make the test circular).

    Theoretical identity: omega_S = Psi(-i)
    Proof: the risk-neutral martingale condition E_Q[S_T/S_0] = e^{(r-q)T}
    requires that the log-price drift alpha satisfies
        alpha + Psi(-i) = r - q.
    Choosing alpha = r - q - omega_S and comparing gives omega_S = Psi(-i).
    (Some texts write omega_S = -Psi(-i) with the opposite sign convention
    for the drift; the convention here matches cgmy_bspline.parameters and
    the cos_pricer, where the characteristic function uses (r - omega_S) as
    the drift coefficient.)
    """
    gY = sc_gamma(-Y)
    xi = complex(xi)
    return C * gY * ((M - 1j * xi) ** Y - M ** Y + (G + 1j * xi) ** Y - G ** Y)


# ---------------------------------------------------------------------------
# Part (a) — omega_S consistency with the characteristic exponent
# ---------------------------------------------------------------------------

_TEST_PAIRS_A = [
    (0.5,  5.0,  8.0,  1.5),   # moderate activity, Y near middle of (1,2)
    (1.2,  3.0, 10.0,  1.8),   # higher activity, Y near upper end
]


@pytest.mark.parametrize("C,G,M,Y", _TEST_PAIRS_A)
def test_omega_S_matches_char_exp(C, G, M, Y):
    """(a) omega_S(C,G,M,Y) must equal Re[Psi(-i)] to relative tol 1e-10."""
    direct = omega_S(C, G, M, Y)

    # Independent path: evaluate characteristic exponent at xi = -i
    psi_at_minus_i = _psi_char_exp(C, G, M, Y, xi=-1j)

    # The result must be real (imaginary part should be machine-zero)
    assert abs(psi_at_minus_i.imag) < 1e-12, (
        f"Psi(-i) has non-negligible imaginary part {psi_at_minus_i.imag!r} "
        f"for (C,G,M,Y)=({C},{G},{M},{Y}); check complex power branches."
    )
    indirect = psi_at_minus_i.real

    try:
        np.testing.assert_allclose(
            direct, indirect, rtol=1e-10,
            err_msg=f"(C,G,M,Y)=({C},{G},{M},{Y}): omega_S={direct} vs Psi(-i)={indirect}",
        )
        print(f"PASS (a): (C,G,M,Y)=({C},{G},{M},{Y})  "
              f"omega_S={direct:.8g}  Psi(-i)={indirect:.8g}")
    except AssertionError as exc:
        print(f"FAIL (a): {exc}")
        raise


# ---------------------------------------------------------------------------
# Part (b) — ValueError on invalid parameters
# ---------------------------------------------------------------------------

_INVALID_PARAMS = [
    # (C, G, M, Y, description)
    (0.0,  5.0, 5.0, 1.5, "C=0 (not > 0)"),
    (-0.1, 5.0, 5.0, 1.5, "C<0"),
    (0.5,  0.0, 5.0, 1.5, "G=0 (not > 0)"),
    (0.5, -1.0, 5.0, 1.5, "G<0"),
    (0.5,  5.0, 1.0, 1.5, "M=1 (not > 1)"),
    (0.5,  5.0, 0.8, 1.5, "M<1"),
    (0.5,  5.0, 5.0, 1.0, "Y=1 (boundary, not in (1,2))"),
    (0.5,  5.0, 5.0, 0.8, "Y<1 (not in infinite-variation regime)"),
    (0.5,  5.0, 5.0, 2.0, "Y=2 (boundary, not in (1,2))"),
    (0.5,  5.0, 5.0, 2.1, "Y>2"),
]


@pytest.mark.parametrize("C,G,M,Y,desc", _INVALID_PARAMS)
def test_invalid_params_raise(C, G, M, Y, desc):
    """(b) CGMYParams must raise ValueError for every invalid combination."""
    try:
        with pytest.raises(ValueError):
            CGMYParams(C=C, G=G, M=M, Y=Y, r=0.05, q=0.0, S0=100.0)
        print(f"PASS (b): ValueError raised for {desc}")
    except Exception as exc:
        print(f"FAIL (b): {desc} — {exc}")
        raise


# ---------------------------------------------------------------------------
# Part (c) — transform / inverse-transform round-trip
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N_ROUND_TRIP = 5

# Draw 5 valid (C, G, M, Y) tuples with fixed seed
_ROUND_TRIP_PARAMS = []
for _ in range(_N_ROUND_TRIP):
    _C = float(_RNG.uniform(0.01, 5.0))
    _G = float(_RNG.uniform(0.5, 20.0))
    _M = 1.0 + float(_RNG.uniform(0.01, 15.0))
    _Y = 1.0 + float(_RNG.uniform(0.01, 0.98))
    _ROUND_TRIP_PARAMS.append((_C, _G, _M, _Y))


@pytest.mark.parametrize("C,G,M,Y", _ROUND_TRIP_PARAMS)
def test_transform_round_trip(C, G, M, Y):
    """(c) inverse_transform(transform_to_unconstrained(theta)) == theta to 1e-10."""
    eta1, eta2, eta3, eta4 = transform_to_unconstrained(C, G, M, Y)
    C2, G2, M2, Y2 = inverse_transform(eta1, eta2, eta3, eta4)

    try:
        np.testing.assert_allclose(
            [C2, G2, M2, Y2], [C, G, M, Y], rtol=1e-10,
            err_msg=(
                f"Round-trip failed for (C,G,M,Y)=({C:.6g},{G:.6g},{M:.6g},{Y:.6g})\n"
                f"  recovered: ({C2:.6g},{G2:.6g},{M2:.6g},{Y2:.6g})"
            ),
        )
        print(
            f"PASS (c): (C,G,M,Y)=({C:.6g},{G:.6g},{M:.6g},{Y:.6g})  "
            f"recovered=({C2:.6g},{G2:.6g},{M2:.6g},{Y2:.6g})"
        )
    except AssertionError as exc:
        print(f"FAIL (c): {exc}")
        raise


# ---------------------------------------------------------------------------
# Standalone runner (also works as: python test_params.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    all_passed = True

    print("=" * 60)
    print("Part (a): omega_S vs characteristic exponent Psi(-i)")
    print("=" * 60)
    for C, G, M, Y in _TEST_PAIRS_A:
        direct   = omega_S(C, G, M, Y)
        indirect = _psi_char_exp(C, G, M, Y, -1j).real
        rel_err  = abs(direct - indirect) / max(abs(direct), 1e-300)
        ok = rel_err < 1e-10
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  {tag} (C,G,M,Y)=({C},{G},{M},{Y})  "
              f"omega_S={direct:.8g}  Psi(-i)={indirect:.8g}  rel_err={rel_err:.2e}")

    print()
    print("=" * 60)
    print("Part (b): ValueError on invalid parameters")
    print("=" * 60)
    for C, G, M, Y, desc in _INVALID_PARAMS:
        try:
            CGMYParams(C=C, G=G, M=M, Y=Y, r=0.05, q=0.0, S0=100.0)
            print(f"  FAIL — no ValueError raised for {desc}")
            all_passed = False
        except ValueError:
            print(f"  PASS — ValueError raised for {desc}")

    print()
    print("=" * 60)
    print("Part (c): transform / inverse-transform round-trip")
    print("=" * 60)
    for C, G, M, Y in _ROUND_TRIP_PARAMS:
        eta = transform_to_unconstrained(C, G, M, Y)
        C2, G2, M2, Y2 = inverse_transform(*eta)
        errs = [abs(a - b) / max(abs(a), 1e-300)
                for a, b in zip([C, G, M, Y], [C2, G2, M2, Y2])]
        ok  = all(e < 1e-10 for e in errs)
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  {tag} (C,G,M,Y)=({C:.5g},{G:.5g},{M:.5g},{Y:.5g})  "
              f"max_rel_err={max(errs):.2e}")

    print()
    print("=" * 60)
    print("OVERALL:", "PASS" if all_passed else "FAIL")
    print("=" * 60)
    sys.exit(0 if all_passed else 1)
