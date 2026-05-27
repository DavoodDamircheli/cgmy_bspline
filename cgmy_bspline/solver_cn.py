import numpy as np
import scipy.sparse.linalg as spla
import time
from scipy.integrate import fixed_quad

from cgmy_bspline import grid as grid_module, matrices, projection, boundary
from cgmy_bspline.bspline_basis import phi, active_indices


def manufactured_exact(x, tau, params, nu=1.5):
    """V_exact(x, tau) = exp(-r*tau) * (1 + x^2)^{-nu}."""
    return np.exp(-params.r * tau) * (1.0 + np.asarray(x, dtype=float) ** 2) ** (-nu)


def _project_function(fn, N, p, h, x_min, M_h):
    """L2-project fn onto V_h. Returns coefficient vector (no kink splitting)."""
    ks = active_indices(N, p)
    b = np.zeros(N + p - 1)
    for i, k in enumerate(ks):
        j_lo = max(0, k)
        j_hi = min(N - 1, k + p - 1)
        val = 0.0
        for j in range(j_lo, j_hi + 1):
            a_elem = x_min + j * h
            b_elem = x_min + (j + 1) * h
            v, _ = fixed_quad(
                lambda x_, _k=k: fn(x_) * phi(x_, _k, p, h, x_min),
                a_elem, b_elem, n=10,
            )
            val += v
        b[i] = val
    return spla.spsolve(M_h, b)


def manufactured_source_discrete(x_nodes, tau, params, N, p, h, x_min,
                                  matrices_dict, nu=1.5):
    """
    Discrete source load vector for the manufactured solution at time tau.

    F_k = integral (partial_tau V_exact + A_h[V_exact_h]) phi_k dx
        = ((A_h - r * M_h) @ c_exact)_k

    Since partial_tau V_exact = -r * V_exact, the load vector simplifies to
    (A_h - r*M_h) applied to the coefficient vector of the L2 projection of V_exact.

    Returns the load vector F (not a coefficient vector).
    """
    M_h = matrices_dict['M_h']
    A_h = matrices_dict['A_h']
    c_exact = _project_function(
        lambda x: manufactured_exact(x, tau, params, nu),
        N, p, h, x_min, M_h,
    )
    return (A_h - params.r * M_h) @ c_exact


def solve_cn(params, N, p, N_tau, bandwidth=64, manufactured=False, nu=1.5):
    """
    Crank-Nicolson B-spline Galerkin solver for the CGMY FPDE.

    Crank-Nicolson scheme (eq. 81):
      (M_h + dt/2 * A_h) c^{n+1} = (M_h - dt/2 * A_h) c^n + dt/2 * (F^{n+1} + F^n)

    A single LU factorisation of (M_h + dt/2 * A_h) is reused at every step.

    Parameters
    ----------
    params       : CGMYParams
    N            : number of spatial intervals
    p            : B-spline order (use 3 throughout)
    N_tau        : number of time steps
    bandwidth    : truncation bandwidth for fractional matrices
    manufactured : if True, add manufactured-solution source term
    nu           : exponent for manufactured solution (requires nu > Y/2)

    Returns
    -------
    dict with keys: c_final, x_nodes, V_final, error_L2, cpu_time
    """
    t_start = time.time()

    g = grid_module.make_grid(params, N)
    x_min, x_max, h = g['x_min'], g['x_max'], g['h']
    x_nodes = g['nodes']

    mats = matrices.operator_matrix(N, p, h, x_min, params, bandwidth=bandwidth)
    M_h = mats['M_h']
    A_h = mats['A_h']

    dt = params.T / N_tau
    F_mat = (M_h + (dt / 2.0) * A_h).tocsc()
    B_mat = (M_h - (dt / 2.0) * A_h).tocsr()

    # ── Dirichlet BC setup (option pricing only) ──────────────────────────────
    # For p=3 the p-1=2 "ghost" basis functions at each boundary end are the
    # only ones with nonzero values at x_min / x_max. At x_min both have
    #   phi_i(x_min) = h^{-1/2} * 1/2,
    # so enforcing V(x_min, tau) = g_L requires c_0 = c_1 = g_L * h^{1/2}.
    # We pin those DOFs by replacing their rows in F_mat with identity rows and
    # overriding RHS[bc_dofs] at every time step.
    if not manufactured:
        n_dof = N + p - 1
        bc_left  = list(range(p - 1))               # DOFs 0 .. p-2
        bc_right = list(range(n_dof - p + 1, n_dof)) # DOFs n-(p-1) .. n-1
        bc_dofs  = bc_left + bc_right

        F_lil = F_mat.tolil()
        for i in bc_dofs:
            F_lil[i, :] = 0.0
            F_lil[i, i] = 1.0
        F_mat = F_lil.tocsc()

    lu = spla.splu(F_mat)

    if manufactured:
        # Exploit separable structure: V_exact(x, tau) = exp(-r*tau) * g(x)
        # Precompute c_g = proj_h(g) and F_base = (A_h - r*M_h) @ c_g once.
        # Source load vector: F(tau) = exp(-r*tau) * F_base.
        c_g = _project_function(lambda x: (1.0 + x ** 2) ** (-nu), N, p, h, x_min, M_h)
        F_base = (A_h - params.r * M_h) @ c_g
        c = c_g.copy()
    else:
        c = projection.l2_project_payoff(params, g, p, mats)
        # Pin boundary DOFs in the initial condition (tau = 0).
        c_L0 = params.K * np.sqrt(h)   # K * exp(-r*0) * h^{1/2}
        for i in bc_left:
            c[i] = c_L0
        for i in bc_right:
            c[i] = 0.0

    tau = 0.0
    for n in range(N_tau):
        tau_new = (n + 1) * dt

        f_old = boundary.boundary_forcing_vector(N, p, h, x_min, params, tau, mats)
        f_new = boundary.boundary_forcing_vector(N, p, h, x_min, params, tau_new, mats)

        rhs = B_mat @ c + (dt / 2.0) * (f_new + f_old)

        if manufactured:
            rhs += (dt / 2.0) * (
                np.exp(-params.r * tau_new) + np.exp(-params.r * tau)
            ) * F_base
        else:
            # Enforce Dirichlet BCs: V(x_min) = K*exp(-r*tau), V(x_max) = 0.
            c_L = params.K * np.exp(-params.r * tau_new) * np.sqrt(h)
            for i in bc_left:
                rhs[i] = c_L
            for i in bc_right:
                rhs[i] = 0.0

        c = lu.solve(rhs)
        tau = tau_new

    V_final = projection.eval_solution(c, g, p, x_nodes)

    error_L2 = None
    if manufactured:
        V_exact = manufactured_exact(x_nodes, params.T, params, nu)
        error_L2 = np.sqrt(h * np.sum((V_final - V_exact) ** 2))

    return {
        'c_final': c,
        'x_nodes': x_nodes,
        'V_final': V_final,
        'error_L2': error_L2,
        'cpu_time': time.time() - t_start,
    }
