import math
import pytest
from cgmy_bspline.parameters import CGMYParams, omega_stock, lambda0

PARAMS_I  = CGMYParams(C=0.5, G=5.0, M=5.0, Y=0.5,  r=0.1, K=100.0, T=0.2)
PARAMS_II = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5,  r=0.1, K=100.0, T=0.2)


def test_omega_stock_and_lambda0_finite():
    for label, p in [("PARAMS_I", PARAMS_I), ("PARAMS_II", PARAMS_II)]:
        os_val = omega_stock(p)
        l0_val = lambda0(p)
        print(f"\n{label}: omega_stock={os_val:.8f}  lambda0={l0_val:.8f}")
        assert math.isfinite(os_val), f"{label}: omega_stock is not finite"
        assert math.isfinite(l0_val), f"{label}: lambda0 is not finite"
        assert isinstance(os_val, float)
        assert isinstance(l0_val, float)


def test_convenience_methods_match_functions():
    for p in [PARAMS_I, PARAMS_II]:
        assert p.omega_stock() == omega_stock(p)
        assert p.lambda0() == lambda0(p)


def test_invalid_Y_equal_one():
    with pytest.raises(ValueError, match="Y must be"):
        CGMYParams(C=1, G=1, M=2.0, Y=1.0, r=0.1, K=100, T=1.0)


def test_invalid_M_not_greater_than_one():
    with pytest.raises(ValueError, match="M must be"):
        CGMYParams(C=1, G=1, M=0.9, Y=0.5, r=0.1, K=100, T=1.0)
