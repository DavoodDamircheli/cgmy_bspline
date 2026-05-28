import numpy as np
import scipy.sparse.linalg as spla
import time
from scipy.integrate import fixed_quad
from scipy.special import gamma as sc_gamma

from cgmy_bspline import grid as grid_module, matrices, projection, boundary
from cgmy_bspline.bspline_basis import phi, active_indices
from cgmy_bspline.parameters import omega_stock, lambda0
from cgmy_bspline.toeplitz import build_fractional_toeplitz


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


def solve_cn(params, N, p, N_tau, bandwidth=64, mode='banded', manufactured=False, nu=1.5):
    """
    Crank-Nicolson B-spline Galerkin solver for the CGMY FPDE.

    Crank-Nicolson scheme (eq. 81):
      (M_h + dt/2 * A_h) c^{n+1} = (M_h - dt/2 * A_h) c^n + dt/2 * (F^{n+1} + F^n)

    Parameters
    ----------
    params       : CGMYParams
    N            : number of spatial intervals
    p            : B-spline order (use 3 throughout)
    N_tau        : number of time steps
    bandwidth    : truncation half-bandwidth for mode='banded'; also used as the
                   banded-preconditioner bandwidth for mode='toeplitz_precond'
    mode         : 'banded'           -- sparse LU direct solve (Version 1)
                   'toeplitz'         -- matrix-free GMRES with full Toeplitz matvec
                   'toeplitz_precond' -- same but with banded LU preconditioner
    manufactured : if True, add manufactured-solution source term
    nu           : exponent for manufactured solution (requires nu > Y/2)

    Returns
    -------
    dict with keys:
      c_final      -- final coefficient vector
      x_nodes      -- spatial nodes
      V_final      -- option values at x_nodes
      error_L2     -- L2 error vs manufactured solution (None for option pricing)
      cpu_time     -- wall-clock time (s)
      gmres_iters  -- average GMRES iterations per time step (None for mode='banded')
    """
    t_start = time.time()

    g = grid_module.make_grid(params, N)
    x_min, x_max, h = g['x_min'], g['x_max'], g['h']
    x_nodes = g['nodes']
    n_dof = N + p - 1
    dt = params.T / N_tau

    # ── Mode: banded (sparse direct solve) ──────────────────────────────────────
    if mode == 'banded':
        mats = matrices.operator_matrix(N, p, h, x_min, params, bandwidth=bandwidth)
        M_h = mats['M_h']
        A_h = mats['A_h']

        F_mat = (M_h + (dt / 2.0) * A_h).tocsc()
        B_mat = (M_h - (dt / 2.0) * A_h).tocsr()

        if not manufactured:
            bc_left  = list(range(p - 1))
            bc_right = list(range(n_dof - p + 1, n_dof))
            bc_dofs  = bc_left + bc_right

            F_lil = F_mat.tolil()
            for i in bc_dofs:
                F_lil[i, :] = 0.0
                F_lil[i, i] = 1.0
            F_mat = F_lil.tocsc()

        lu = spla.splu(F_mat)

        if manufactured:
            c_g = _project_function(lambda x: (1.0 + x ** 2) ** (-nu), N, p, h, x_min, M_h)
            F_base = (A_h - params.r * M_h) @ c_g
            c = c_g.copy()
        else:
            c = projection.l2_project_payoff(params, g, p, mats)
            c_L0 = params.K * np.sqrt(h)
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
                c_L = params.K * np.exp(-params.r * tau_new) * np.sqrt(h)
                for i in bc_left:
                    rhs[i] = c_L
                for i in bc_right:
                    rhs[i] = 0.0
            c = lu.solve(rhs)
            tau = tau_new

        gmres_iters = None

    # ── Mode: toeplitz / toeplitz_precond (matrix-free Krylov) ──────────────────
    else:
        M_h = matrices.mass_matrix(N, p, h, x_min)
        D_h = matrices.derivative_matrix(N, p, h, x_min)
        mats = {'M_h': M_h}

        r     = params.r
        C     = params.C
        Y     = params.Y
        gY    = sc_gamma(-Y)
        om_S  = omega_stock(params)
        lam0  = lambda0(params)

        # Full Toeplitz operators (no bandwidth truncation): O(n log n) matvec.
        K_G_top = build_fractional_toeplitz(n_dof, h, params.G * h, p, Y, 'minus')
        K_M_top = build_fractional_toeplitz(n_dof, h, params.M * h, p, Y, 'plus')

        def _A_matvec(v):
            return (
                -(r - om_S) * (D_h @ v)
                - C * gY * K_G_top.matvec(v)
                - C * gY * K_M_top.matvec(v)
                + lam0 * (M_h @ v)
            )

        def _B_matvec(v):
            return M_h @ v - (dt / 2.0) * _A_matvec(v)

        if not manufactured:
            bc_left  = list(range(p - 1))
            bc_right = list(range(n_dof - p + 1, n_dof))
            bc_dofs  = bc_left + bc_right

            def _F_matvec(v):
                w = M_h @ v + (dt / 2.0) * _A_matvec(v)
                for i in bc_dofs:
                    w[i] = v[i]
                return w
        else:
            bc_left = bc_right = bc_dofs = []

            def _F_matvec(v):
                return M_h @ v + (dt / 2.0) * _A_matvec(v)

        F_op = spla.LinearOperator((n_dof, n_dof), matvec=_F_matvec, dtype=float)

        # Optional banded preconditioner
        precond = None
        if mode == 'toeplitz_precond':
            mats_bnd = matrices.operator_matrix(N, p, h, x_min, params, bandwidth=bandwidth)
            F_bnd = (mats_bnd['M_h'] + (dt / 2.0) * mats_bnd['A_h']).tocsc()
            if not manufactured:
                F_lil = F_bnd.tolil()
                for i in bc_dofs:
                    F_lil[i, :] = 0.0
                    F_lil[i, i] = 1.0
                F_bnd = F_lil.tocsc()
            bnd_lu = spla.splu(F_bnd)
            precond = spla.LinearOperator(
                (n_dof, n_dof), matvec=lambda v: bnd_lu.solve(v)
            )

        if manufactured:
            c_g = _project_function(lambda x: (1.0 + x ** 2) ** (-nu), N, p, h, x_min, M_h)
            F_base = _A_matvec(c_g) - params.r * (M_h @ c_g)
            c = c_g.copy()
        else:
            c = projection.l2_project_payoff(params, g, p, mats)
            c_L0 = params.K * np.sqrt(h)
            for i in bc_left:
                c[i] = c_L0
            for i in bc_right:
                c[i] = 0.0

        total_iters = 0
        tau = 0.0
        for n_step in range(N_tau):
            tau_new = (n_step + 1) * dt
            f_old = boundary.boundary_forcing_vector(N, p, h, x_min, params, tau, mats)
            f_new = boundary.boundary_forcing_vector(N, p, h, x_min, params, tau_new, mats)
            rhs = _B_matvec(c) + (dt / 2.0) * (f_new + f_old)
            if manufactured:
                rhs += (dt / 2.0) * (
                    np.exp(-params.r * tau_new) + np.exp(-params.r * tau)
                ) * F_base
            else:
                c_L = params.K * np.exp(-params.r * tau_new) * np.sqrt(h)
                for i in bc_left:
                    rhs[i] = c_L
                for i in bc_right:
                    rhs[i] = 0.0

            step_iters = [0]

            def _cb(xk, _ctr=step_iters):
                _ctr[0] += 1

            c, info = spla.gmres(
                F_op, rhs, M=precond, rtol=1e-10, restart=50,
                callback=_cb, callback_type='pr_norm',
            )
            if info != 0:
                import warnings
                warnings.warn(f"GMRES did not converge at step {n_step + 1}: info={info}")
            total_iters += step_iters[0]
            tau = tau_new

        gmres_iters = total_iters / N_tau

    # ── Common output ────────────────────────────────────────────────────────────
    V_final = projection.eval_solution(c, g, p, x_nodes)

    error_L2 = None
    if manufactured:
        V_exact = manufactured_exact(x_nodes, params.T, params, nu)
        error_L2 = np.sqrt(h * np.sum((V_final - V_exact) ** 2))

    return {
        'c_final':     c,
        'x_nodes':     x_nodes,
        'V_final':     V_final,
        'error_L2':    error_L2,
        'cpu_time':    time.time() - t_start,
        'gmres_iters': gmres_iters,
    }
