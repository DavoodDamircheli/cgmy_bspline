"""
CGMYParams dataclass — Convention B (full symbol, lambda_0 includes reaction terms).

State variable   : x = log(S)
Zeroth-order:    lambda_0 = r + C*Gamma(-Y)*(M^Y + G^Y)
omega_S:         C*Gamma(-Y)*[(M-1)^Y - M^Y + (G+1)^Y - G^Y]
char_exp(xi):    C*Gamma(-Y)*[(M-i*xi)^Y - M^Y + (G+i*xi)^Y - G^Y]
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import gamma as sc_gamma


@dataclass
class CGMYParams:
    C:  float   # activity > 0
    G:  float   # left-tail decay > 0
    M:  float   # right-tail decay > 1 (risk-neutral moment condition)
    Y:  float   # fine-structure in (0,1)∪(1,2)
    r:  float   # risk-free rate
    T:  float   # maturity
    K:  float   # strike
    S0: float   # current stock price

    def __post_init__(self):
        if self.C <= 0:
            raise ValueError("C must be > 0")
        if self.G <= 0:
            raise ValueError("G must be > 0")
        if self.M <= 1:
            raise ValueError("M must be > 1 for the risk-neutral moment condition")
        if not (0 < self.Y < 2) or abs(self.Y - 1.0) < 1e-10:
            raise ValueError("Y must be in (0,1) union (1,2)")

    @property
    def omega_S(self) -> float:
        """Martingale correction omega_S = C*Gamma(-Y)*[(M-1)^Y - M^Y + (G+1)^Y - G^Y]."""
        gY = sc_gamma(-self.Y)
        return self.C * gY * (
            (self.M - 1.0) ** self.Y - self.M ** self.Y
            + (self.G + 1.0) ** self.Y - self.G ** self.Y
        )

    @property
    def lambda_0(self) -> float:
        """Reaction coefficient (Convention B): lambda_0 = r + C*Gamma(-Y)*(M^Y + G^Y)."""
        gY = sc_gamma(-self.Y)
        return self.r + self.C * gY * (self.M ** self.Y + self.G ** self.Y)

    def char_exp(self, xi) -> complex:
        """
        CGMY characteristic exponent (pure jump, no drift):
          Psi(xi) = C*Gamma(-Y)*[(M - i*xi)^Y - M^Y + (G + i*xi)^Y - G^Y]
        """
        xi = complex(xi)
        gY = sc_gamma(-self.Y)
        return self.C * gY * (
            (self.M - 1j * xi) ** self.Y - self.M ** self.Y
            + (self.G + 1j * xi) ** self.Y - self.G ** self.Y
        )


if __name__ == "__main__":
    # Ludvigsson set (i): finite variation
    p1 = CGMYParams(C=0.5, G=5.0, M=5.0, Y=0.5, r=0.1, T=0.2, K=100.0, S0=100.0)
    # Ludvigsson set (ii): infinite variation
    p2 = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=0.1, T=0.2, K=100.0, S0=100.0)

    for name, p in [("Set (i) Y=0.5", p1), ("Set (ii) Y=1.5", p2)]:
        print(f"{name}: omega_S={p.omega_S:.6f}, lambda_0={p.lambda_0:.6f}")
        assert abs(p.char_exp(0.0)) < 1e-12, f"Psi(0) != 0: {p.char_exp(0.0)}"
        print(f"  char_exp(-1j) = {p.char_exp(-1j):.6f}  (should equal omega_S via martingale)")

    print("PASS: CGMYParams self-test")
