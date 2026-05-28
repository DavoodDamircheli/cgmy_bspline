"""
CGMY calibration to an implied-volatility surface.

Speed strategy
--------------
M_h and D_h depend only on the grid (N, h, x_min, p) and are built once in
__init__. The initial coefficient vector c_init (L2 projection of the put
payoff) also does not depend on CGMY parameters and is cached the same way.

The Toeplitz sparsity structure (rows, cols, lags) is precomputed with numpy
in __init__ and reused for every (C, G, M, Y) evaluation: only the connection-
coefficient values (from the IFFT) change. This vectorised fill is 10-50x
faster than the Python double-loop in matrices.fractional_stiffness_matrices.

Cache invalidation: if |Y - Y_cached| > 0.01 the IFFT arrays are flushed and
recomputed; otherwise the cached IFFT result is reused (accepting O(dY)
perturbation to the symbol). Changing G or M always triggers a fresh IFFT for
the corresponding array.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import gamma as sc_gamma
from scipy.optimize import differential_evolution, minimize

from cgmy_bspline.parameters import CGMYParams, omega_stock, lambda0
from cgmy_bspline.connection_coefficients import tempered_coeffs_via_ifft
from cgmy_bspline import matrices, projection
from cgmy_bspline.grid import make_grid_manual
from cgmy_bspline.greeks import bs_implied_vol
from cgmy_bspline.projection import assemble_put_rhs, eval_solution


class CGMYCalibrator:
    """
    Calibrates CGMY parameters to an implied-volatility surface.

    Calibration objective (eq. 138):
      L(theta) = sum_{i,j} w_{ij} * (sigma_model - sigma_market)^2

    Parameters
    ----------
    market_ivs  : array (n_maturities, n_strikes) of market implied vols
    strikes     : array (n_strikes,) — interpreted as spot prices S0 evaluated
                  against the fixed option strike K_ref
    maturities  : sequence of maturities T_i
    r           : risk-free rate
    S0          : current stock price (used only for reference)
    K_ref       : strike used as K in CGMYParams; also sets the grid centre
    weights     : optional (n_maturities, n_strikes); default uniform
    N           : spatial grid intervals (default 128)
    bandwidth   : IFFT points / Toeplitz bandwidth for fractional matrices
    """

    def __init__(self, market_ivs, strikes, maturities, r, S0, K_ref,
                 weights=None, N=128, bandwidth=64):
        self.market_ivs = np.asarray(market_ivs, dtype=float)
        self.strikes    = np.asarray(strikes,    dtype=float)
        self.maturities = list(maturities)
        self.r          = float(r)
        self.S0         = float(S0)
        self.K_ref      = float(K_ref)
        self.N          = int(N)
        self.p          = 3
        self.bandwidth  = int(bandwidth)
        self.weights    = (np.ones_like(self.market_ivs)
                           if weights is None else np.asarray(weights, dtype=float))

        n_dof   = N + self.p - 1
        log_K   = np.log(K_ref)
        x_min   = log_K - 5.0
        x_max   = log_K + 3.0

        # Fixed grid — never changes during optimisation.
        self._g    = make_grid_manual(x_min, x_max, N)
        self._h    = self._g['h']
        self._xmin = x_min
        self._x_eval = np.log(self.strikes)

        # ── Permanent cache: M_h, D_h (expensive quad-based assembly) ─────────
        self._M_h = matrices.mass_matrix(N, self.p, self._h, x_min)
        self._D_h = matrices.derivative_matrix(N, self.p, self._h, x_min)

        # ── Permanent cache: payoff initial condition ──────────────────────────
        # max(K_ref - exp(x), 0) is theta-independent; project it once.
        _p0 = CGMYParams(C=1.0, G=5.0, M=5.0, Y=0.5, r=r, K=K_ref, T=1.0)
        b0  = assemble_put_rhs(_p0, self._g, self.p)
        c0  = spla.spsolve(self._M_h, b0)
        bc_left  = list(range(self.p - 1))
        bc_right = list(range(n_dof - self.p + 1, n_dof))
        c0[bc_left]  = K_ref * np.sqrt(self._h)   # left BC at tau=0
        c0[bc_right] = 0.0
        self._c_init    = c0.copy()
        self._bc_left   = bc_left
        self._bc_right  = bc_right
        self._bc_dofs   = bc_left + bc_right

        # ── Permanent cache: Toeplitz sparsity structure ───────────────────────
        # Rows, cols, and lags (m = i-j) for the banded Toeplitz pattern.
        # Depends only on (N, bandwidth) — completely Y-independent.
        half_bw = bandwidth // 2
        I, J    = np.meshgrid(np.arange(n_dof), np.arange(n_dof), indexing='ij')
        M_arr   = I - J
        mask    = (M_arr >= -half_bw) & (M_arr <= half_bw - 1)
        self._K_rows = I[mask].astype(np.int32)
        self._K_cols = J[mask].astype(np.int32)
        self._K_lags = M_arr[mask]      # shape (nnz,)

        # ── Y-indexed IFFT cache ───────────────────────────────────────────────
        self._Y_lam   = None   # Y value at which lam arrays were computed
        self._lam_G   = None   # cached tempered_coeffs for K_G
        self._lam_M   = None   # cached tempered_coeffs for K_M
        self._G_lam   = None
        self._M_lam   = None

        # ── Number of time steps per maturity (fixed, set at construction) ────
        N_tau_base      = 50
        self._N_taus    = [max(10, round(N_tau_base * T)) for T in self.maturities]
        self._n_evals   = 0

    # ──────────────────────────────────────────────────────────────────────────

    def _get_lam(self, G, M, Y):
        """
        Return (lam_G, lam_M) from cache or recompute via IFFT.

        Cache key: Y (invalidated when |Y - Y_cached| > 0.01 or G / M change).
        """
        Y_stale = (self._Y_lam is None or abs(Y - self._Y_lam) > 0.01)
        G_stale = (self._G_lam != G)
        M_stale = (self._M_lam != M)

        h = self._h
        if Y_stale or G_stale:
            self._lam_G = (tempered_coeffs_via_ifft(G * h, Y, self.p, 'minus',
                                                    L=self.bandwidth)
                           * h ** (-Y))
            self._G_lam = G
        if Y_stale or M_stale:
            self._lam_M = (tempered_coeffs_via_ifft(M * h, Y, self.p, 'plus',
                                                    L=self.bandwidth)
                           * h ** (-Y))
            self._M_lam = M
        if Y_stale:
            self._Y_lam = Y

        return self._lam_G, self._lam_M

    def _build_fractional_matrices(self, G, M, Y):
        """Assemble K_G and K_M using the cached sparsity structure."""
        n_dof  = self.N + self.p - 1
        lam_G, lam_M = self._get_lam(G, M, Y)

        idx    = self._K_lags % self.bandwidth
        vals_G = lam_G[idx]
        vals_M = lam_M[idx]

        K_G = sp.csr_matrix((vals_G, (self._K_rows, self._K_cols)),
                            shape=(n_dof, n_dof))
        K_M = sp.csr_matrix((vals_M, (self._K_rows, self._K_cols)),
                            shape=(n_dof, n_dof))
        return K_G, K_M

    # ──────────────────────────────────────────────────────────────────────────

    def _price_puts(self, C, G, M, Y):
        """
        Price European puts for all (maturity, strike) pairs.

        One FPDE solve per maturity; evaluate at self._x_eval = log(strikes).
        Returns array of shape (n_maturities, n_strikes).
        """
        gY   = sc_gamma(-Y)
        tmp  = CGMYParams(C=C, G=G, M=M, Y=Y, r=self.r, K=self.K_ref, T=1.0)
        om_S = omega_stock(tmp)
        lam0 = lambda0(tmp)

        K_G, K_M = self._build_fractional_matrices(G, M, Y)
        A_h = (
            -(self.r - om_S) * self._D_h
            - C * gY * K_G
            - C * gY * K_M
            + lam0 * self._M_h
        ).tocsr()

        prices = np.empty((len(self.maturities), len(self.strikes)))

        for i, (T_i, N_tau) in enumerate(zip(self.maturities, self._N_taus)):
            dt    = T_i / N_tau
            F_mat = (self._M_h + (dt / 2.0) * A_h).tocsc()
            B_mat = (self._M_h - (dt / 2.0) * A_h).tocsr()

            F_lil = F_mat.tolil()
            for k in self._bc_dofs:
                F_lil[k, :] = 0.0
                F_lil[k, k] = 1.0
            lu = spla.splu(F_lil.tocsc())

            c   = self._c_init.copy()
            tau = 0.0
            for _ in range(N_tau):
                tau_new = tau + dt
                rhs     = B_mat @ c
                c_L     = self.K_ref * np.exp(-self.r * tau_new) * np.sqrt(self._h)
                for k in self._bc_left:
                    rhs[k] = c_L
                for k in self._bc_right:
                    rhs[k] = 0.0
                c   = lu.solve(rhs)
                tau = tau_new

            prices[i] = eval_solution(c, self._g, self.p, self._x_eval)

        return prices

    # ──────────────────────────────────────────────────────────────────────────

    def objective(self, theta):
        """
        Evaluate L(theta) for theta = [C, G, M, Y].

        Returns a large penalty (1e6) if parameters are out of the valid region.
        """
        C, G, M, Y = float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])
        self._n_evals += 1

        if (C <= 0 or G <= 0 or M <= 1.0
                or not (0.0 < Y < 2.0) or abs(Y - 1.0) < 1e-3):
            return 1e6

        try:
            prices = self._price_puts(C, G, M, Y)
        except Exception:
            return 1e6

        loss = 0.0
        for i, T_i in enumerate(self.maturities):
            for j, K_j in enumerate(self.strikes):
                iv_m = bs_implied_vol(prices[i, j], S=K_j, K=self.K_ref,
                                      T=T_i, r=self.r)
                w = self.weights[i, j]
                if np.isnan(iv_m):
                    loss += w * 1.0    # soft penalty for unconvertible prices
                else:
                    loss += w * (iv_m - self.market_ivs[i, j]) ** 2

        return float(loss)

    # ──────────────────────────────────────────────────────────────────────────

    def calibrate(self, de_kwargs=None, lbfgs_kwargs=None):
        """
        Two-phase calibration:
          Phase 1 — differential_evolution (global search)
          Phase 2 — L-BFGS-B (local refinement from DE result)

        Parameters
        ----------
        de_kwargs    : dict of keyword overrides for differential_evolution
                       (e.g. {'popsize': 8, 'maxiter': 100} for a quick demo)
        lbfgs_kwargs : dict of keyword overrides for minimize (L-BFGS-B options)

        Returns dict: theta_star, C, G, M, Y, rmse_vol, n_func_evals, cpu_time.
        """
        t0            = time.time()
        self._n_evals = 0

        bounds = [(0.01, 5.0), (0.5, 50.0), (1.01, 50.0), (0.01, 1.99)]

        # Phase 1: differential evolution
        _de_kw = dict(popsize=15, maxiter=300, tol=1e-5, seed=42, polish=False)
        if de_kwargs:
            _de_kw.update(de_kwargs)

        de = differential_evolution(self.objective, bounds=bounds, **_de_kw)

        # Phase 2: L-BFGS-B refinement from DE result
        _lb_opts = dict(ftol=1e-12, gtol=1e-9, maxiter=500)
        if lbfgs_kwargs:
            _lb_opts.update(lbfgs_kwargs)

        lb = minimize(
            self.objective, de.x,
            method='L-BFGS-B', bounds=bounds,
            options=_lb_opts,
        )

        theta_star = lb.x
        C, G, M, Y = theta_star

        # Final in-sample RMSE
        prices = self._price_puts(C, G, M, Y)
        sq = []
        for i, T_i in enumerate(self.maturities):
            for j, K_j in enumerate(self.strikes):
                iv_m = bs_implied_vol(prices[i, j], S=K_j, K=self.K_ref,
                                      T=T_i, r=self.r)
                if not np.isnan(iv_m):
                    sq.append((iv_m - self.market_ivs[i, j]) ** 2)
        rmse = float(np.sqrt(np.mean(sq))) if sq else float('nan')

        return {
            'theta_star':   theta_star,
            'C': float(C), 'G': float(G), 'M': float(M), 'Y': float(Y),
            'rmse_vol':     rmse,
            'n_func_evals': self._n_evals,
            'cpu_time':     time.time() - t0,
        }
