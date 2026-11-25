"""
rmgarchpy/cgarch.py
Dynamic Copula GARCH (cGARCH) model.

Implements the two-step dynamic copula estimation:
    Step 1: fit univariate GARCH models to each margin.
    Step 2: transform residuals to pseudo-observations via probability integral transform.
    Step 3: fit a (static or dynamic) copula to the pseudo-observations.

Supports:
    - static copula (constant parameters)
    - dynamic copula (time-varying parameters via rolling window or ARMA-style dynamics)
    - multiple copula families and rotations
    - simulation and forecasting
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as tdist

from . import utils
from .ugarch import ugarchfit
from . import mdcopula


# ----------------------------------------------------------------- pseudo-observations

def _pseudo_obs(residuals, h, dist="norm", nu=5):
    """
    Convert GARCH residuals to pseudo-observations in (0,1).

    u_t = F(F^{-1}(h_t); theta)  where F is the conditional CDF.
    """
    n, k = residuals.shape
    u = np.zeros((n, k))
    for j in range(k):
        if dist == "norm":
            u[:, j] = norm.cdf(residuals[:, j] / np.sqrt(h[:, j]))
        elif dist == "std":
            u[:, j] = tdist.cdf(residuals[:, j] / np.sqrt(h[:, j]), df=nu)
        else:
            raise ValueError(f"Unknown distribution: {dist}")
    # clip to avoid boundary issues
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return u


# ----------------------------------------------------------------- cGARCH fit

def cgarchfit(data, copula_family="clayton", rotation=0,
              ugarch_spec=None, copula_method="ML",
              dynamic=False, window=None, nu=5,
              show_progress=False):
    """
    Fit a dynamic copula GARCH model.

    Parameters
    ----------
    data : array-like, shape (n, k)
    copula_family : str
        'clayton', 'gumbel', 'frank', 'normal', 't'
    rotation : int
        0, 90, 180, 270
    ugarch_spec : dict, optional
        Univariate GARCH specification.
    copula_method : str
        'ML' or 'MoM'
    dynamic : bool
        If True, estimate time-varying copula parameters via rolling window.
    window : int, optional
        Window size for dynamic estimation. Defaults to n//4.
    nu : float
        df for t-copula / t-distributed errors.
    show_progress : bool

    Returns
    -------
    dict with:
        'ugarch'    : list of univariate GARCH fits
        'copula'    : copula parameter estimates (static or dynamic)
        'u'         : pseudo-observations
        'residuals' : raw residuals
        'h'         : conditional variances
        'loglik'    : total log-likelihood
        'nobs'      : number of observations
    """
    X = utils.as_matrix(data)
    n, k = X.shape

    if ugarch_spec is None:
        ugarch_spec = {"model": "sGARCH", "garch": (1, 1), "dist": "norm"}

    if window is None:
        window = max(50, n // 4)

    # Step 1: univariate GARCH fits
    ugarch_fits = []
    h = np.zeros((n, k))
    residuals = np.zeros((n, k))
    for j in range(k):
        fit = ugarchfit(X[:, j], spec=ugarch_spec, show_progress=False)
        ugarch_fits.append(fit)
        h[:, j] = fit["h"]
        residuals[:, j] = fit["residuals"]

    # Step 2: pseudo-observations
    u = _pseudo_obs(residuals, h, dist=ugarch_spec.get("dist", "norm"), nu=nu)

    # Step 3: copula estimation
    copula_params = {}
    if not dynamic:
        # static: pairwise estimation for each pair
        for i in range(k):
            for j in range(i + 1, k):
                est = mdcopula.mdCopulaParam(u[:, i], u[:, j],
                                             family=copula_family,
                                             rotation=rotation,
                                             method=copula_method,
                                             nu=nu)
                copula_params[(i, j)] = est
        loglik = sum(v["loglik"] for v in copula_params.values() if not np.isnan(v["loglik"]))
    else:
        # dynamic: rolling window estimation
        copula_params = {"theta": np.zeros((n, k, k)),
                         "window": window}
        for t in range(window, n):
            u_win = u[t - window:t]
            for i in range(k):
                for j in range(i + 1, k):
                    est = mdcopula.mdCopulaParam(u_win[:, i], u_win[:, j],
                                                 family=copula_family,
                                                 rotation=rotation,
                                                 method=copula_method,
                                                 nu=nu)
                    copula_params["theta"][t, i, j] = est["theta"]
                    copula_params["theta"][t, j, i] = est["theta"]
        loglik = np.nan  # not straightforward for dynamic

    return {
        "ugarch": ugarch_fits,
        "copula": copula_params,
        "u": u,
        "residuals": residuals,
        "h": h,
        "loglik": loglik,
        "nobs": n,
        "k": k,
        "family": copula_family,
        "rotation": rotation,
        "dynamic": dynamic
    }


# ----------------------------------------------------------------- cGARCH forecast

def cgarchforecast(fit, n_ahead=10, method="MCS", n_sim=1000, seed=None):
    """
    Forecast cGARCH model.

    Parameters
    ----------
    fit : dict
        Output of cgarchfit.
    n_ahead : int
    method : str
        'MCS' (Monte Carlo) or 'analytic'
    n_sim : int
    seed : int, optional

    Returns
    -------
    dict with 'sim' (n_sim, n_ahead, k), 'h' (n_ahead, k), 'u' (n_ahead, k)
    """
    rng = np.random.default_rng(seed)
    k = fit["k"]
    n = fit["nobs"]

    # univariate variance forecasts
    h_fc = np.zeros((n_ahead, k))
    for j in range(k):
        h_last = fit["h"][-1, j]
        h_fc[:, j] = h_last  # simplified: constant forecast

    # copula-based simulation
    if not fit["dynamic"]:
        # static copula: sample from fitted copula
        family = fit["family"]
        rotation = fit["rotation"]
        # for simplicity, use average pairwise theta
        thetas = [v["theta"] for v in fit["copula"].values()]
        theta_avg = np.mean(thetas) if thetas else 0.0

        u_sim = np.zeros((n_sim, n_ahead, k))
        for s in range(n_sim):
            # simulate from copula
            copula_sim = mdcopula.mdCopulaSim(n_ahead, family=family,
                                              theta=theta_avg, rotation=rotation,
                                              nu=nu, seed=rng)
            u_sim[s] = copula_sim
    else:
        # dynamic: use last estimated parameters
        u_sim = np.zeros((n_sim, n_ahead, k))
        for s in range(n_sim):
            # simplified: use last theta matrix
            theta_last = fit["copula"]["theta"][-1]
            # for now, fall back to independent simulation
            u_sim[s] = rng.random((n_ahead, k))

    # transform pseudo-observations to returns via inverse GARCH
    sim = np.zeros((n_sim, n_ahead, k))
    for s in range(n_sim):
        for j in range(k):
            dist = fit["ugarch"][j]["spec"].get("dist", "norm")
            if dist == "norm":
                z = norm.ppf(u_sim[s, :, j])
            else:
                z = tdist.ppf(u_sim[s, :, j], df=nu)
            sim[s, :, j] = z * np.sqrt(h_fc[:, j])

    return {
        "sim": sim,
        "h": h_fc,
        "u": u_sim
    }


# ----------------------------------------------------------------- utility: tail risk measures

def cgrisk(fit, alpha=0.05, horizon=1, n_sim=5000, seed=None):
    """
    Estimate conditional Value-at-Risk (VaR) and Expected Shortfall (ES)
    from a fitted cGARCH model.

    Parameters
    ----------
    fit : dict
    alpha : float
        Tail probability.
    horizon : int
        Forecast horizon.
    n_sim : int
    seed : int

    Returns
    -------
    dict with 'VaR' (k,), 'ES' (k,)
    """
    fc = cgarchforecast(fit, n_ahead=horizon, method="MCS", n_sim=n_sim, seed=seed)
    sim = fc["sim"]  # (n_sim, horizon, k)

    # aggregate over horizon (sum for simplicity)
    agg = np.sum(sim, axis=1)  # (n_sim, k)

    VaR = np.percentile(agg, alpha * 100, axis=0)
    ES = np.mean(agg[agg <= VaR], axis=0)

    return {"VaR": VaR, "ES": ES}