from dataclasses import dataclass
from scipy.special import gamma as sc_gamma


@dataclass
class CGMYParams:
    C: float    # activity,          C > 0
    G: float    # left-tail decay,   G > 0
    M: float    # right-tail decay,  M > 1  (risk-neutral moment condition)
    Y: float    # fine-structure,    Y in (0,1) union (1,2), Y != 1
    r: float    # risk-free rate
    K: float    # strike
    T: float    # maturity

    def __post_init__(self):
        if self.C <= 0:
            raise ValueError("C must be > 0")
        if self.G <= 0:
            raise ValueError("G must be > 0")
        if self.M <= 1:
            raise ValueError("M must be > 1 for the risk-neutral moment condition")
        if not (0 < self.Y < 2) or abs(self.Y - 1.0) < 1e-10:
            raise ValueError("Y must be in (0,1) union (1,2)")

    def omega_stock(self):
        return omega_stock(self)

    def lambda0(self):
        return lambda0(self)


def omega_stock(p: CGMYParams) -> float:
    """
    Martingale correction (eq. 4):
      omega_S = C * Gamma(-Y) * [(M-1)^Y - M^Y + (G+1)^Y - G^Y]
    """
    gY = sc_gamma(-p.Y)
    return p.C * gY * (
        (p.M - 1) ** p.Y - p.M ** p.Y + (p.G + 1) ** p.Y - p.G ** p.Y
    )


def lambda0(p: CGMYParams) -> float:
    """
    Reaction constant for the semidiscrete system: lambda_0 = r.

    The connection coefficient symbol S = (delta +/- i*tau)^Y - delta^Y already
    includes the -delta^Y compensation, contributing C*Gamma(-Y)*(G^Y+M^Y)*M_h
    as a reaction term inside K_G and K_M. Adding that term again here would
    double-count it, giving negative eigenvalues (unstable CN) for Y in (0,1).
    """
    return p.r
