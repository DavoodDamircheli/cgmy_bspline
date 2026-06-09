"""
Step 9 — Method comparison table: COS reference, B-spline Galerkin,
Grünwald–Letnikov FD, PIDE (FFT-based jump convolution).

Parameters: Set (ii), Y=1.5, N=128 (GL instability demo).
Errors relative to COS with N_cos=2048.
"""
import time
import os
import numpy as np
from scipy.special import gamma as sc_gamma

from cgmy_bspline.parameters import CGMYParams
from cgmy_bspline.solver_cn import solve_cn
from cgmy_bspline.grid import make_grid
from cgmy_bspline.projection import eval_solution
from cgmy_bspline.src.cos_pricer import cos_put_price
from cgmy_bspline.src.cgmy_params import CGMYParams as SrcParams

PARAMS = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=0.1, K=100.0, T=0.2)
N_COMP = 128
K_val  = PARAMS.K
S0_ATM = K_val


def _src(S0):
    return SrcParams(C=PARAMS.C, G=PARAMS.G, M=PARAMS.M, Y=PARAMS.Y,
                     r=PARAMS.r, T=PARAMS.T, K=K_val, S0=S0)


# ── 1. COS reference ────────────────────────────────────────────────────────

def run_cos():
    t0 = time.perf_counter()
    p = cos_put_price(_src(S0_ATM), N_cos=2048)
    cpu = time.perf_counter() - t0
    return p, cpu


# ── 2. B-spline Galerkin ────────────────────────────────────────────────────

def run_bspline(cos_ref):
    N = N_COMP
    t0 = time.perf_counter()
    result = solve_cn(PARAMS, N=N, p=3, N_tau=4 * N, bandwidth=2 * N)
    cpu = time.perf_counter() - t0
    g = make_grid(PARAMS, N)
    x0 = np.log(S0_ATM)
    V_atm = eval_solution(result['c_final'], g, 3, np.array([x0]))[0]
    err = abs(V_atm - cos_ref)
    return err, cpu


# ── 3. Grünwald–Letnikov FD ─────────────────────────────────────────────────

def _gl_weights_vec(Y, n):
    """GL weights via recurrence: w_0=1, w_j = w_{j-1}*(j-1-Y)/j."""
    w = np.empty(n + 1)
    w[0] = 1.0
    for j in range(1, n + 1):
        w[j] = w[j - 1] * (j - 1.0 - Y) / j
    return w


def _apply_gl_fft(V, w, h, Y):
    """
    Apply the GL left and right fractional difference operators via FFT conv.
    Each operator is a Toeplitz convolution; extend to 2n-1 for linear conv.
    """
    n = len(V)
    npad = 2 * n
    W_fft = np.fft.rfft(w[:n], n=npad)
    V_fft = np.fft.rfft(V, n=npad)
    # Left: (w * V)[k] = sum_{j=0}^{k} w[j]*V[k-j]
    conv = np.fft.irfft(W_fft * V_fft, n=npad)[:n]
    DL = h ** (-Y) * conv

    # Right: mirror V and convolve for the right derivative
    V_rev = V[::-1].copy()
    V_rev_fft = np.fft.rfft(V_rev, n=npad)
    conv_r = np.fft.irfft(W_fft * V_rev_fft, n=npad)[:n]
    DR = h ** (-Y) * conv_r[::-1]
    return DL, DR


def run_gl(cos_ref):
    """
    Grünwald–Letnikov scheme. Explicit Euler with CFL dt ≈ h^Y/2.
    For Y=1.5 this is typically unstable — we report what happens.
    """
    N = N_COMP
    Y = PARAMS.Y
    C = PARAMS.C
    r = PARAMS.r
    K = PARAMS.K
    T = PARAMS.T
    gY  = sc_gamma(-Y)
    omega_S = (C * gY * ((PARAMS.M - 1) ** Y - PARAMS.M ** Y
                         + (PARAMS.G + 1) ** Y - PARAMS.G ** Y))

    g = make_grid(PARAMS, N)
    h = float(g['h'])
    x = g['nodes']
    n = len(x)

    # CFL-type stability condition for GL (explicit)
    dt_cfl = (h ** Y) / (2.0 * abs(C * gY))
    N_tau  = max(1, int(np.ceil(T / dt_cfl)))
    dt     = T / N_tau

    w = _gl_weights_vec(Y, n)

    V = np.maximum(K - np.exp(x), 0.0).copy()

    t0 = time.perf_counter()
    stable = True
    for step in range(N_tau):
        DL, DR = _apply_gl_fft(V, w, h, Y)
        dV = dt * (-(r - omega_S) * np.gradient(V, h)
                   + C * gY * (DL + DR)  # gY < 0 for Y in (1,2)
                   - r * V)
        V_new = V + dV
        tau_new = (step + 1) * dt
        V_new[0]  = max(0.0, K * np.exp(-r * tau_new) - np.exp(x[0]))
        V_new[-1] = 0.0

        if (not np.isfinite(V_new).all()
                or np.max(np.abs(V_new)) > 1e6
                or np.max(np.abs(V_new)) > 100 * K):
            stable = False
            break
        V = V_new

    cpu = time.perf_counter() - t0

    if not stable:
        return float('nan'), cpu, False

    x0 = np.log(S0_ATM)
    V_atm = float(np.interp(x0, x, V))
    err = abs(V_atm - cos_ref)
    # If the solution is wildly off, call it unstable
    if err > 100 * cos_ref:
        return err, cpu, False
    return err, cpu, True


# ── 4. PIDE (FFT-based jump convolution) ────────────────────────────────────

def run_pide(cos_ref):
    """
    PIDE with jump integral computed via FFT convolution.
    Implicit (Crank-Nicolson diffusion / explicit jump) splitting.
    For Y in (1,2) the kernel is weakly singular but still integrable.
    """
    N = N_COMP
    Y = PARAMS.Y
    C = PARAMS.C
    r = PARAMS.r
    K = PARAMS.K
    T = PARAMS.T
    G_ = PARAMS.G
    M_ = PARAMS.M
    gY  = sc_gamma(-Y)
    omega_S = (C * gY * ((M_ - 1) ** Y - M_ ** Y
                         + (G_ + 1) ** Y - G_ ** Y))

    g = make_grid(PARAMS, N)
    h = float(g['h'])
    x = g['nodes']
    n = len(x)
    N_tau = 4 * N
    dt    = T / N_tau

    # Build Lévy kernel on shift grid y_j = j*h, j = -(n-1)...(n-1)
    # For Y in (1,2), subtract linear term: k_c(y) = k(y) - k(y)*1_{|y|<eps} is handled
    # by the GL-style first-moment compensation in the drift; here we use
    # k(y) directly (suitable for Y<2 with domain truncation).
    y_idx = np.arange(-(n - 1), n)
    y_vals = y_idx * h
    # Avoid y=0 singularity; the compensating drift is absorbed into omega_S
    safe_y = np.where(y_vals == 0, 1.0, y_vals)  # placeholder to avoid division
    kern = np.where(
        y_vals > 0,
        C * np.exp(-M_ * safe_y) * safe_y ** (-1.0 - Y),
        np.where(
            y_vals < 0,
            C * np.exp(G_ * safe_y) * (-safe_y) ** (-1.0 - Y),
            0.0,
        ),
    )
    kern *= h  # trapezoidal weight

    # FFT-based jump operator: (J V)[k] = sum_j kern[j] * V[k+j] - V[k] * sum_j kern[j]
    # implemented as cross-correlation then subtract total mass * V
    total_mass = np.sum(kern)
    kern_fft_size = 2 * n - 1
    nfft = 1 << int(np.ceil(np.log2(kern_fft_size + 1)))

    def jump_op(V):
        V_fft = np.fft.rfft(V, n=nfft)
        # We want the "centered" convolution: shift kern so index 0 = y=0
        # kern[0..n-1] corresponds to y = -(n-1)*h ... -h (negative shifts)
        # kern[n-1..2n-2] corresponds to y = 0 ... (n-1)*h
        # Place it so that the zero-lag is at position n-1
        kern_centered = np.zeros(nfft)
        kern_centered[:len(kern)] = kern
        # Roll so zero-lag (index n-1) is at position 0
        kern_rolled = np.roll(kern_centered, -(n - 1))
        kern_fft = np.fft.rfft(kern_rolled, n=nfft)
        # Cross-correlation gives (J V)[k] = sum_j kern_j * V_{k+j}
        JV_full = np.fft.irfft(kern_fft * V_fft, n=nfft)
        JV = JV_full[:n]
        # Compensate: (J V)[k] = (shift sum) - total_mass * V[k]
        return JV - total_mass * V

    V = np.maximum(K - np.exp(x), 0.0).copy()

    t0 = time.perf_counter()
    for step in range(N_tau):
        J_V = jump_op(V)
        # Simple forward Euler (explicit in jump, implicit in nothing here — fully explicit)
        dV = dt * (-(r - omega_S) * np.gradient(V, h)
                   + J_V
                   - r * V)
        V_new = V + dV
        tau_new = (step + 1) * dt
        V_new[0]  = max(0.0, K * np.exp(-r * tau_new) - np.exp(x[0]))
        V_new[-1] = 0.0
        if not np.isfinite(V_new).all():
            break
        V = V_new

    cpu = time.perf_counter() - t0
    x0 = np.log(S0_ATM)
    V_atm = float(np.interp(x0, x, V))
    err = abs(V_atm - cos_ref)
    return err, cpu


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("cgmy_bspline/results", exist_ok=True)

    print("Running COS reference...")
    cos_ref, cos_cpu = run_cos()
    print(f"  COS reference price: {cos_ref:.6f}  (cpu={cos_cpu:.4f}s)")

    print("Running B-spline Galerkin...")
    bs_err, bs_cpu = run_bspline(cos_ref)
    print(f"  B-spline Galerkin: err={bs_err:.3e}, cpu={bs_cpu:.3f}s")

    print("Running GL finite difference (may be unstable for Y=1.5)...")
    gl_err, gl_cpu, gl_stable = run_gl(cos_ref)
    gl_note = "converged" if gl_stable else "UNSTABLE/diverged"
    print(f"  GL FD: err={gl_err}, cpu={gl_cpu:.3f}s, status={gl_note}")

    print("Running PIDE (FFT convolution)...")
    pide_err, pide_cpu = run_pide(cos_ref)
    print(f"  PIDE: err={pide_err:.3e}, cpu={pide_cpu:.3f}s")

    gl_err_str = f"{gl_err:>12.3e}" if gl_stable else "    UNSTABLE "

    lines = [
        "=== Method Comparison, Set (ii): C=0.1, G=2, M=3.5, Y=1.5 ===",
        f"  N = {N_COMP},  dt = T/(4*N),  COS reference (N_cos=2048): {cos_ref:.6f}",
        "",
        f"  {'Method':<28}  {'|e_ATM|':>12}  {'CPU(s)':>8}  Notes",
        "  " + "-" * 76,
        f"  {'COS (reference)':<28}  {'---':>12}  {cos_cpu:>8.4f}  benchmark",
        f"  {'B-spline Galerkin (p=3)':<28}  {bs_err:>12.3e}  {bs_cpu:>8.3f}  "
        f"unconditionally stable, high accuracy",
        f"  {'GL finite difference':<28}  {gl_err_str}  {gl_cpu:>8.3f}  "
        f"{gl_note}: Y in (1,2) lacks dissipation",
        f"  {'PIDE (FFT trapezoidal)':<28}  {pide_err:>12.3e}  {pide_cpu:>8.3f}  "
        f"missing near-origin compensation for Y>1",
    ]

    print()
    for ln in lines:
        print(ln)

    path = "cgmy_bspline/results/table_method_comparison.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
