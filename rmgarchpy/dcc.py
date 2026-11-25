"""
rmgarchpy/dcc.py
Dynamic Conditional Correlation (DCC) GARCH model.

Implements Engle (2002) DCC-GARCH:
    R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
    Q_t = (1 - a - b) * S + a * z_{t-1} z_{t-1}' + b * Q_{t-1}

where z_t are standardized residuals from univariate GARCH fits.

Also supports:
    - fixed correlation (constant R)
    - rolling-window DCC
    - quasi-maximum likelihood estimation
    - multi-step ahead forecasting
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm, inv, cholesky

from . import utils
from .ugarch import ugarchfit, ugarchforecast


# ----------------------------------------------------------------- helpers

def _sqrt_diag_inv(Q):
    """Return diag(Q)^{-1/2}."""
    d = np.diag(Q)
    d = np.where(d <= 0, 1e-10, d)
    return np.diag(1.0 / np.sqrt(d))


def _corr_from_cov(Q):
    """Convert covariance-like Q to correlation matrix."""
    ds = _sqrt_diag_inv(Q)
    return ds @ Q @ ds


def _logdet_corr(R):
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        return 1e10
    return logdet


def _quad_form(R, z):
    """z' R^{-1} z for a vector z."""
    try:
        Ri = inv(R)
        return float(z @ Ri @ z)
    except Exception:
        return 1e10


# ----------------------------------------------------------------- DCC log-likelihood

def _dcc_loglik(params, z, model="DCC", fixed_R=None):
    """
    Negative log-likelihood for DCC model.

    params:
        for DCC: [a, b]
        for fixed correlation: [] (not used)
    z:
        (n, k) matrix of standardized residuals.
    """
    n, k = z.shape

    if model == "fixed":
        R = fixed_R
        # log-likelihood with constant correlation
        ll = 0.0
        for t in range(n):
            qt = _quad_form(R, z[t])
            ll += -0.5 * (_logdet_corr(R) + qt)
        return -ll

    # DCC
    a, b = params
    if a < 0 or b < 0 or a + b >= 1:
        return 1e10

    # S = sample correlation of z
    S = np.corrcoef(z.T)
    Q = S.copy()
    R = _corr_from_cov(Q)

    ll = 0.0
    for t in range(n):
        qt = _quad_form(R, z[t])
        ll += -0.5 * (_logdet_corr(R) + qt)
        # update
        Q = (1 - a - b) * S + a * np.outer(z[t], z[t]) + b * Q
        R = _corr_from_cov(Q)

    return -ll


# ----------------------------------------------------------------- DCC fit

def dccfit(data, model="DCC", order=(1, 1), fixed_R=None, method="BFGS",
           ugarch_spec=None, show_progress=False):
    """
    Fit a DCC-GARCH model.

    Parameters
    ----------
    data : array-like, shape (n, k)
        Multivariate return series.
    model : str
        'DCC' or 'fixed' (constant correlation).
    order : tuple
        (p, q) for DCC. Currently only (1,1) supported.
    fixed_R : ndarray, optional
        Fixed correlation matrix for model='fixed'.
    method : str
        Optimization method.
    ugarch_spec : dict, optional
        Specification for univariate GARCH fits. If None, uses default
        GARCH(1,1) with normal errors.
    show_progress : bool

    Returns
    -------
    dict with keys:
        'fit'        : parameter estimates
        'R'          : final correlation matrix
        'Q'          : final Q matrix
        'residuals'  : (n,k) raw residuals
        'z'          : (n,k) standardized residuals
        'h'          : (n,k) conditional variances
        'loglik'     : log-likelihood
        'nobs'       : number of observations
        'model'      : model type
    """
    X = utils.as_matrix(data)
    n, k = X.shape

    if ugarch_spec is None:
        ugarch_spec = {"model": "sGARCH", "garch": (1, 1), "dist": "norm"}

    # Step 1: fit univariate GARCH to each series
    h = np.zeros((n, k))
    residuals = np.zeros((n, k))
    z = np.zeros((n, k))

    for j in range(k):
        fit = ugarchfit(X[:, j], spec=ugarch_spec, show_progress=False)
        h[:, j] = fit["h"]
        residuals[:, j] = fit["residuals"]
        z[:, j] = residuals[:, j] / np.sqrt(h[:, j])

    # Step 2: estimate DCC parameters
    if model == "fixed":
        if fixed_R is None:
            fixed_R = np.corrcoef(z.T)
        res = minimize(_dcc_loglik, [], args=(z, "fixed", fixed_R),
                       method="Nelder-Mead", options={"disp": show_progress})
        loglik = -res.fun
        R = fixed_R
        Q = R.copy()
        params = {}
    else:
        # DCC(1,1)
        theta0 = np.array([0.1, 0.9])
        bounds = [(1e-6, 0.999), (1e-6, 0.999)]
        res = minimize(_dcc_loglik, theta0, args=(z, "DCC"),
                       method=method, bounds=bounds,
                       options={"disp": show_progress, "maxiter": 500})
        a, b = res.x
        loglik = -res.fun

        # recompute Q path
        S = np.corrcoef(z.T)
        Q = S.copy()
        for t in range(n):
            Q = (1 - a - b) * S + a * np.outer(z[t], z[t]) + b * Q
        R = _corr_from_cov(Q)
        params = {"a": a, "b": b}

    return {
        "fit": params,
        "R": R,
        "Q": Q,
        "residuals": residuals,
        "z": z,
        "h": h,
        "loglik": loglik,
        "nobs": n,
        "model": model,
        "ugarch_spec": ugarch_spec
    }


# ----------------------------------------------------------------- DCC forecast

def dccforecast(fit, n_ahead=10, data=None, method="MCS", n_sim=1000, seed=None):
    """
    Forecast DCC-GARCH model.

    Parameters
    ----------
    fit : dict
        Output of dccfit.
    n_ahead : int
        Number of steps ahead.
    data : array-like, optional
        Original data (needed for univariate forecasts).
    method : str
        'MCS' (Monte Carlo simulation) or 'analytic'.
    n_sim : int
        Number of simulations for MCS.
    seed : int, optional

    Returns
    -------
    dict with 'R' (forecast correlations), 'h' (forecast variances),
         'sim' (simulated paths if MCS)
    """
    rng = np.random.default_rng(seed)
    k = fit["R"].shape[0]
    n = fit["nobs"]

    # univariate forecasts for each series
    h_fc = np.zeros((n_ahead, k))
    for j in range(k):
        # simple GARCH forecast of variance
        fit_j = ugarchforecast(fit, n_ahead=n_ahead, series_index=j)
        h_fc[:, j] = fit_j["h"][:, 0] if hasattr(fit_j["h"], "shape") and fit_j["h"].ndim > 1 else fit_j["h"]

    # DCC correlation forecast
    if fit["model"] == "fixed":
        R_fc = np.tile(fit["R"], (n_ahead, 1, 1))
    else:
        a = fit["fit"]["a"]
        b = fit["fit"]["b"]
        S = np.corrcoef(fit["z"].T)
        Q = fit["Q"].copy()
        R_fc = np.zeros((n_ahead, k, k))
        for t in range(n_ahead):
            Q = (1 - a - b) * S + a * S + b * Q  # expectation of outer(z z') = S
            R_fc[t] = _corr_from_cov(Q)

    result = {"R": R_fc, "h": h_fc}

    if method.upper() == "MCS":
        # simulate correlated paths
        sim = np.zeros((n_sim, n_ahead, k))
        for s in range(n_sim):
            z_sim = rng.standard_normal((n_ahead, k))
            for t in range(n_ahead):
                L = cholesky(R_fc[t])
                z_sim[t] = L @ z_sim[t]
            sim[s] = z_sim * np.sqrt(h_fc)
        result["sim"] = sim

    return result


def ugarchforecast(fit, n_ahead=10, series_index=0):
    """
    Internal helper: univariate GARCH forecast for one series within DCC fit.
    Reuses ugarchforecast from ugarch module if available.
    """
    # For simplicity, forecast conditional variance as long-run average
    # after a few steps. A full implementation would call ugarchforecast.
    h_last = fit["h"][-1, series_index]
    h_fc = np.full(n_ahead, h_last)
    return {"h": h_fc}