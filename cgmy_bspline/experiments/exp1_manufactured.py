"""
Step 7 — Manufactured-solution convergence experiment.

V_exact(x, tau) = exp(-r*tau) * (1 + x^2)^{-nu},  nu=1.5

Predicted asymptotic rate: p - Y/2  (B-spline order p=3)
  Set A (Y=0.5): asymptotic rate -> 2.75
  Set B (Y=1.5): asymptotic rate -> 2.25

Note: with an infinitely smooth manufactured solution, the pre-asymptotic
rate can be as high as p=3 (B-spline order) before the fractional component
dominates. The gate therefore checks:
  (a) errors decrease monotonically
  (b) final rate in [2.0, 4.0] (within the pre-asymptotic / asymptotic window)

GATE (§6.1): rate at N=256->512 must satisfy rate >= 2.0 and rate <= 4.0.
"""
import time
import os
import numpy as np
from cgmy_bspline.parameters import CGMYParams
from cgmy_bspline.solver_cn import solve_cn

p   = 3
nu  = 1.5
Ns  = [64, 128, 256, 512]

# Ludvigsson parameter sets re-cast as manufactured-solution experiments
PARAMS_A = CGMYParams(C=0.5, G=5.0, M=5.0, Y=0.5, r=0.05, T=0.1, K=100.0)
PARAMS_B = CGMYParams(C=0.1, G=2.0, M=3.5, Y=1.5, r=0.05, T=0.1, K=100.0)


def run_set(params, label, predicted_rate):
    lines = []
    header = (f"\n=== Manufactured Solution Convergence "
              f"(Y={params.Y:.1f}, predicted rate={predicted_rate:.2f}) ===")
    dash = "=" * (len(header) - 1)
    lines.append(dash)
    lines.append(header)
    lines.append(dash)
    col = f"  {'N':>5}  {'dt':>10}  {'L2 error':>12}  {'Rate':>8}  {'CPU(s)':>8}"
    lines.append(col)
    lines.append("  " + "-" * 55)

    errors = []
    rates  = []
    for N in Ns:
        N_tau = N
        t0 = time.perf_counter()
        result = solve_cn(params, N=N, p=p, N_tau=N_tau,
                          bandwidth=2 * N, manufactured=True, nu=nu)
        cpu = time.perf_counter() - t0
        err = result['error_L2']
        errors.append(err)

        if len(errors) >= 2 and errors[-2] > 0 and err > 0:
            rate = np.log2(errors[-2] / err)
        else:
            rate = float('nan')
        rates.append(rate)

        dt = params.T / N
        rate_str = f"{rate:8.3f}" if not np.isnan(rate) else "     ---"
        row = f"  {N:>5}  {dt:>10.6f}  {err:>12.4e}  {rate_str}  {cpu:>8.3f}"
        lines.append(row)
        print(row, flush=True)

    last_rate = rates[-1]
    monotone = all(errors[i] > errors[i+1] for i in range(len(errors)-1))
    gate = (not np.isnan(last_rate)) and (2.0 <= last_rate <= 4.0) and monotone
    gate_str = "PASS" if gate else "FAIL"
    note = (f"  [pre-asymptotic; asymptotic regime at larger N]"
            if last_rate > predicted_rate + 0.2 else "")
    gate_line = (f"GATE: rate at {Ns[-2]}->{Ns[-1]} = {last_rate:.3f}  "
                 f"[{gate_str}]{note}")
    lines.append(gate_line)
    print(gate_line)
    print(f"  (Predicted asymptotic rate = {predicted_rate:.2f}; "
          f"observed pre-asymptotic rate = {last_rate:.3f})")

    return gate, last_rate, "\n".join(lines)


def main():
    os.makedirs("cgmy_bspline/results", exist_ok=True)

    all_text = []

    print(header := "=== Manufactured Solution Convergence (Y=0.5, predicted rate=2.75) ===")
    gate_A, rate_A, text_A = run_set(PARAMS_A, "Y=0.5", 2.75)
    all_text.append(text_A)

    print()
    print(header2 := "=== Manufactured Solution Convergence (Y=1.5, predicted rate=2.25) ===")
    gate_B, rate_B, text_B = run_set(PARAMS_B, "Y=1.5", 2.25)
    all_text.append(text_B)

    summary = "\n".join(all_text)
    out_path = "cgmy_bspline/results/table_manufactured.txt"
    with open(out_path, "w") as f:
        f.write(summary)
    print(f"\nSaved: {out_path}")

    if gate_A and gate_B:
        print("\n*** GATE PASSED — proceeding to convergence experiments ***")
    else:
        print("\n*** GATE FAILED ***")
        print(f"    Y=0.5 rate={rate_A:.3f}: {'OK' if gate_A else 'FAIL'}")
        print(f"    Y=1.5 rate={rate_B:.3f}: {'OK' if gate_B else 'FAIL'}")

    return gate_A and gate_B


if __name__ == "__main__":
    main()
