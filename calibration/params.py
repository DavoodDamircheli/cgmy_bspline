"""
CGMY parameter vector for market calibration (Section 7).

Restricted to Y in (1, 2) — infinite-variation regime required by the
paper's Section 7 calibration.  This is a distinct dataclass from
cgmy_bspline.parameters.CGMYParams (which covers Y in (0,2) and is used
by the B-spline / CN solver).  Adding q (dividend yield) and S0 here
means the two cannot be trivially merged without changing the solver
interface, so we keep them separate and bridge via explicit conversion
in the pricer wrappers.
"""

from dataclasses import dataclass
import math

from scipy.special import gamma as sc_gamma


@dataclass
class CGMYParams:
    """CGMY parameter vector for the calibration pipeline.

    Model parameters
    ----------------
    C : float  activity,          C > 0
    G : float  left-tail decay,   G > 0
    M : float  right-tail decay,  M > 1  (exponential moment condition)
    Y : float  fine-structure,    Y in (1, 2)  — infinite-variation regime

    Pricing parameters
    ------------------
    r  : float  continuously compounded risk-free rate
    q  : float  continuously compounded dividend / repo yield
    S0 : float  current stock price (spot)
    """

    C:  float
    G:  float
    M:  float
    Y:  float
    r:  float
    q:  float
    S0: float

    def __post_init__(self):
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C}")
        if self.G <= 0:
            raise ValueError(f"G must be > 0, got {self.G}")
        if self.M <= 1.0:
            raise ValueError(
                f"M must be > 1 (exponential moment / martingale correction "
                f"requires E[e^X_1] < inf), got {self.M}"
            )
        if not (1.0 < self.Y < 2.0):
            raise ValueError(
                f"Y must be in (1, 2) for Section 7 calibration "
                f"(infinite-variation regime), got {self.Y}"
            )


# ---------------------------------------------------------------------------
# Scalar formula functions (operate on raw floats, not CGMYParams, so that
# the optimiser can call them directly without constructing a dataclass).
# ---------------------------------------------------------------------------


def omega_S(C: float, G: float, M: float, Y: float) -> float:
    """Martingale correction (eq. omega):

        omega_S = C * Gamma(-Y) * ((M-1)^Y - M^Y + (G+1)^Y - G^Y)

    Under risk-neutral Q the log-price drift is r - q - omega_S, chosen so
    that E_Q[S_T/S_0] = exp((r-q)*T).  Equivalently, omega_S equals the
    CGMY characteristic exponent Psi evaluated at the purely imaginary point
    xi = -i:  omega_S = Psi(-i)  (see test_params.py part (a) for proof).

    Note: Gamma(-Y) > 0 for Y in (1, 2) because -Y lies in (-2, -1) where
    the Gamma function is positive.  We compute it without assuming the sign.
    """
    gY = sc_gamma(-Y)  # Gamma(-Y); positive for Y in (1,2), but computed generally
    return C * gY * ((M - 1.0) ** Y - M ** Y + (G + 1.0) ** Y - G ** Y)


def lambda0(C: float, G: float, M: float, Y: float, r: float) -> float:
    """Reaction constant for the Galerkin semidiscrete system:

        lambda_0 = r + C * Gamma(-Y) * (M^Y + G^Y)

    This is the full kill rate (r plus the compensation terms from the
    Lévy-Khintchine representation); it appears as the coefficient of M_h
    in the operator A_h.
    """
    gY = sc_gamma(-Y)
    return r + C * gY * (M ** Y + G ** Y)


# ---------------------------------------------------------------------------
# Unconstrained transform for gradient-based optimisers.
# ---------------------------------------------------------------------------


def transform_to_unconstrained(
    C: float, G: float, M: float, Y: float
) -> tuple[float, float, float, float]:
    """Map constrained (C, G, M, Y) to unconstrained (eta1, eta2, eta3, eta4).

    eta1 = log(C)          C > 0  <->  eta1 in R
    eta2 = log(G)          G > 0  <->  eta2 in R
    eta3 = log(M - 1)      M > 1  <->  eta3 in R
    eta4 = log((Y-1)/(2-Y))  Y in (1,2) <-> eta4 in R  (log-odds of (Y-1)/(2-Y))
    """
    eta1 = math.log(C)
    eta2 = math.log(G)
    eta3 = math.log(M - 1.0)
    eta4 = math.log((Y - 1.0) / (2.0 - Y))
    return eta1, eta2, eta3, eta4


def inverse_transform(
    eta1: float, eta2: float, eta3: float, eta4: float
) -> tuple[float, float, float, float]:
    """Inverse of transform_to_unconstrained.

    C = exp(eta1)
    G = exp(eta2)
    M = 1 + exp(eta3)
    Y = (2*exp(eta4) + 1) / (1 + exp(eta4))   -- logistic back-transform to (1,2)
    """
    C = math.exp(eta1)
    G = math.exp(eta2)
    M = 1.0 + math.exp(eta3)
    e4 = math.exp(eta4)
    Y = (2.0 * e4 + 1.0) / (1.0 + e4)
    return C, G, M, Y
