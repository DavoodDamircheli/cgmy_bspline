"""
European put convergence study: B-spline CN solver vs COS reference.

Fills Tables 4 and 5 of the paper by pricing a European put with PARAMS_I
(Y=0.5) and PARAMS_II (Y=1.5) and comparing against the spectral-accuracy
COS pricer of Fang and Oosterlee (2008).
"""
import numpy as np
from scipy.special import gamma as sc_gamma

from cgmy_bspline.tests.test_parameters import PARAMS_I, PARAMS_II
from cgmy_bspline.parameters import omega_stock
from cgmy_bspline.solver_cn import solve_cn
from cgmy_bspline.projection import eval_solution
from cgmy_bspline.grid import make_grid


# ── COS reference pricer ─────────────────────────────────────────────────────

def cgmy_characteristic_function(u, params, T):
    """
    Characteristic function of log(S_T / S_0) under risk-neutral Q:
      phi(u) = exp(T * Psi(u))
    where
      Psi(u) = C*Gamma(-Y)*[(M-i*u)^Y - M^Y + (G+i*u)^Y - G^Y] + i*u*(r - omega_S)

    u may be a real numpy array. Return complex numpy array.
    Uses the principal branch for (M - 1j*u)**Y and (G + 1j*u)**Y.
    """
    oS = omega_stock(params)
    gY = sc_gamma(-params.Y)
    u = np.asarray(u, dtype=complex)
    psi = (
        params.C * gY * (
            (params.M - 1j * u) ** params.Y - params.M ** params.Y
            + (params.G + 1j * u) ** params.Y - params.G ** params.Y
        )
        + 1j * u * (params.r - oS)
    )
    return np.exp(T * psi)


def cos_price_put(params, S0, n_cos=1024):
    """
    COS method for the European put (Fang and Oosterlee, 2008).

    Steps:
    1. Truncation interval from the first two cumulants of log(S_T/S_0):
         c1 = (r - omega_S) * T
         c2 = C * |Gamma(-Y)| * Y*(Y-1) * (M^{Y-2} + G^{Y-2}) * T
         a  = c1 - 12 * sqrt(|c2|),  b = c1 + 12 * sqrt(|c2|)

    2. Closed-form cosine coefficients of the put payoff on [a, b]:
         V_k = (2/(b-a)) * [K * Chi_k - S0 * Psi_k]
       where Chi_k and Psi_k are the cosine integrals of 1 and exp(x)
       from a to min(log(K/S0), b).

    3. COS sum:
         price = exp(-r*T) * sum_{k=0}^{n_cos-1} ' Re[phi(k*pi/(b-a))
                 * exp(-i*k*pi*a/(b-a))] * V_k
       where ' means the k=0 term is halved.

    Return: put price (float).
    """
    K, T, r = params.K, params.T, params.r
    C, G, M, Y = params.C, params.G, params.M, params.Y

    oS = omega_stock(params)

    # Step 1: truncation interval
    c1 = (r - oS) * T
    c2 = C * abs(sc_gamma(-Y)) * Y * (Y - 1) * (M ** (Y - 2) + G ** (Y - 2)) * T
    a = c1 - 12.0 * np.sqrt(abs(c2))
    b = c1 + 12.0 * np.sqrt(abs(c2))
    ba = b - a

    # Step 2: payoff coefficients
    # Put payoff (K - S0*e^x)^+ is nonzero for x < log(K/S0)
    x_cut = min(np.log(K / S0), b)
    d = x_cut - a          # integration width [a, x_cut]

    k = np.arange(n_cos, dtype=float)
    omega_k = k[1:] * np.pi / ba  # frequencies for k >= 1

    # Chi_k = integral_a^{x_cut} cos(omega_k*(x-a)) dx
    Chi = np.empty(n_cos)
    Chi[0] = d
    Chi[1:] = np.sin(omega_k * d) / omega_k

    # Psi_k = integral_a^{x_cut} exp(x) * cos(omega_k*(x-a)) dx
    #   k=0: exp(x_cut) - exp(a)
    #   k>0: [(exp(x_cut)*cos(omega_k*d) - exp(a))
    #          + omega_k*exp(x_cut)*sin(omega_k*d)] / (1 + omega_k^2)
    ea, ec = np.exp(a), np.exp(x_cut)
    Psi = np.empty(n_cos)
    Psi[0] = ec - ea
    od = omega_k * d
    Psi[1:] = ((ec * np.cos(od) - ea) + omega_k * ec * np.sin(od)) / (1.0 + omega_k ** 2)

    V_k = (2.0 / ba) * (K * Chi - S0 * Psi)

    # Step 3: COS sum
    phi_vals = cgmy_characteristic_function(k * np.pi / ba, params, T)
    coeff = np.real(phi_vals * np.exp(-1j * k * np.pi * a / ba))
    coeff[0] *= 0.5    # k=0 term halved

    return float(np.exp(-r * T) * np.dot(coeff, V_k))


# ── Convergence table ─────────────────────────────────────────────────────────

p = 3
Ns = [64, 128, 256, 512]


def run_table(label, params, table_num):
    S0 = params.K          # at-the-money: S0 = K = 100
    x0 = np.log(S0)

    cos_ref = cos_price_put(params, S0, n_cos=2048)

    sep = '=' * 78
    print(f"\n{sep}")
    print(f"  Table {table_num}: {label}   (Y={params.Y})")
    print(sep)
    print(f"{'N':>6}  {'h':>10}  {'dt':>10}  {'price@S0':>12}  "
          f"{'err@S0':>12}  {'rate':>8}  {'CPU(s)':>8}")
    print(f"{'-' * 78}")

    prev_err = None
    for N in Ns:
        result = solve_cn(params, N=N, p=p, N_tau=N, bandwidth=2 * N)
        g = make_grid(params, N)
        V_at_S0 = eval_solution(result['c_final'], g, p, np.array([x0]))[0]
        err = abs(V_at_S0 - cos_ref)
        h   = g['h']
        dt  = params.T / N
        cpu = result['cpu_time']

        if prev_err is not None and prev_err > 0 and err > 0:
            rate = np.log2(prev_err / err)
        else:
            rate = float('nan')

        print(f"{N:>6}  {h:>10.4e}  {dt:>10.4e}  {V_at_S0:>12.6f}  "
              f"{err:>12.2e}  {rate:>8.3f}  {cpu:>8.2f}")
        prev_err = err

    print(f"\n  COS reference price: {cos_ref:.6f}")


if __name__ == '__main__':
    run_table("PARAMS_I ", PARAMS_I, 4)
    run_table("PARAMS_II", PARAMS_II, 5)
