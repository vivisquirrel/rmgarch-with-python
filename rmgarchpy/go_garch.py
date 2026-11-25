"""
rmgarchpy/go_garch.py
Generalized Orthogonal GARCH (GO-GARCH) model.

Implements the GO-GARCH framework of van der Weide (2002):
    X_t = mu + R * A^{1/2} * z_t
where:
    - X_t is the k-dimensional observed return series
    - mu is the mean vector
    - R is an orthogonal rotation matrix (k x k)
    - A is a diagonal matrix of conditional variances of the orthogonal components
    - z_t are independent standardized innovations

The key idea is to decompose the multivariate process into
orthogonal components, each following a univariate GARCH process.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm, inv, eigh
from scipy.optimize import minimize

from . import utils
from .ugarch import ugarchfit


# ----------------------------------------------------------------- GO-GARCH log-likelihood

def _go_garch_loglik(params, X, n_factors=None, dist="norm"):
    """
    Negative log-likelihood for GO-GARCH model.

    Parameters
    ----------
    params : array
        Flattened parameters: [rotation_params, garch_params...]
    X : ndarray, shape (n, k)
        Return series.
    n_factors : int, optional
        Number of factors to retain. Default: k.
    dist : str
        Error distribution.
    """
    n, k = X.shape
    if n_factors is None:
        n_factors = k

    # Extract rotation matrix from params (using exponential parameterization)
    # For simplicity, we use a direct parameterization of the rotation matrix
    # via its upper triangular part (Cayley transform or similar)
    # Here we use a simplified approach: params contains the GARCH parameters
    # for each factor, and the rotation is estimated from the data.

    # For a full implementation, we would optimize over both rotation and GARCH params.
    # This is a placeholder that assumes rotation is estimated separately.

    # Compute the orthogonal components
    # In practice, this would be done inside the optimization loop
    # with the rotation matrix as a function of params.

    # For now, return a dummy value
    return 1e10


# ----------------------------------------------------------------- GO-GARCH fit

def go_garch_fit(data, n_factors=None, ugarch_spec=None,
                 method="eigen", show_progress=False):
    """
    Fit a GO-GARCH model.

    Parameters
    ----------
    data : array-like, shape (n, k)
        Multivariate return series.
    n_factors : int, optional
        Number of orthogonal factors to retain. Default: k.
    ugarch_spec : dict, optional
        Specification for univariate GARCH models on the factors.
        Default: GARCH(1,1) with normal errors.
    method : str
        Method for estimating the rotation matrix:
        'eigen' : use eigenvectors of the covariance matrix
        'pca'   : use principal component analysis
    show_progress : bool

    Returns
    -------
    dict with keys:
        'R'          : rotation matrix (k x k)
        'A'          : diagonal matrix of conditional variances (n x n_factors)
        'factors'    : orthogonal factors (n x n_factors)
        'ugarch'     : list of univariate GARCH fits for each factor
        'mu'         : mean vector
        'residuals'  : raw residuals
        'h'          : conditional covariance matrices (n x k x k)
        'loglik'     : log-likelihood
        'nobs'       : number of observations
    """
    X = utils.as_matrix(data)
    n, k = X.shape

    if n_factors is None:
        n_factors = k

    if ugarch_spec is None:
        ugarch_spec = {"model": "sGARCH", "garch": (1, 1), "dist": "norm"}

    # Step 1: estimate the rotation matrix
    # Use the eigenvectors of the sample covariance matrix
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    if method == "eigen" or method == "pca":
        # PCA / eigen decomposition of the covariance matrix
        S = np.cov(X_centered.T)
        eigenvalues, eigenvectors = eigh(S)
        # Sort by decreasing eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        R = eigenvectors  # orthogonal rotation matrix
    else:
        raise ValueError(f"Unknown method: {method}")

    # Step 2: compute the orthogonal factors
    # factors_t = R' * (X_t - mu)
    factors = X_centered @ R  # (n, k)

    # Step 3: fit univariate GARCH to each factor
    ugarch_fits = []
    A = np.zeros((n, n_factors))  # conditional variances of factors
    factor_residuals = np.zeros((n, n_factors))

    for j in range(n_factors):
        fit = ugarchfit(factors[:, j], spec=ugarch_spec, show_progress=False)
        ugarch_fits.append(fit)
        A[:, j] = fit["h"]
        factor_residuals[:, j] = fit["residuals"]

    # Step 4: reconstruct the conditional covariance matrices
    # H_t = R * diag(A_t) * R'
    h = np.zeros((n, k, k))
    for t in range(n):
        A_diag = np.diag(A[t, :n_factors])
        h[t] = R @ A_diag @ R.T

    # Step 5: compute log-likelihood
    loglik = 0.0
    for t in range(n):
        # log-likelihood of multivariate normal
        sign, logdet = np.linalg.slogdet(h[t])
        if sign <= 0:
            loglik = -1e10
            break
        try:
            h_inv = inv(h[t])
            resid = X[t] - mu
            quad = resid @ h_inv @ resid
            loglik += -0.5 * (k * np.log(2 * np.pi) + logdet + quad)
        except Exception:
            loglik = -1e10
            break

    # Raw residuals
    residuals = X_centered

    return {
        "R": R,
        "A": A,
        "factors": factors,
        "ugarch": ugarch_fits,
        "mu": mu,
        "residuals": residuals,
        "h": h,
        "loglik": loglik,
        "nobs": n,
        "n_factors": n_factors,
        "ugarch_spec": ugarch_spec
    }


# ----------------------------------------------------------------- GO-GARCH forecast

def go_garch_forecast(fit, n_ahead=10, method="MCS", n_sim=1000, seed=None):
    """
    Forecast GO-GARCH model.

    Parameters
    ----------
    fit : dict
        Output of go_garch_fit.
    n_ahead : int
    method : str
        'MCS' or 'analytic'
    n_sim : int
    seed : int, optional

    Returns
    -------
    dict with 'h' (n_ahead, k, k), 'sim' (n_sim, n_ahead, k) if MCS
    """
    rng = np.random.default_rng(seed)
    k = fit["R"].shape[0]
    n_factors = fit["n_factors"]
    R = fit["R"]
    mu = fit["mu"]

    # Forecast conditional variances of factors
    A_fc = np.zeros((n_ahead, n_factors))
    for j in range(n_factors):
        # Simple forecast: use last value (could be improved with ugarchforecast)
        A_fc[:, j] = fit["A"][-1, j]

    # Forecast conditional covariance matrices
    h_fc = np.zeros((n_ahead, k, k))
    for t in range(n_ahead):
        A_diag = np.diag(A_fc[t])
        h_fc[t] = R @ A_diag @ R.T

    result = {"h": h_fc, "A": A_fc}

    if method.upper() == "MCS":
        # Simulate factor paths
        sim_factors = np.zeros((n_sim, n_ahead, n_factors))
        for s in range(n_sim):
            z = rng.standard_normal((n_ahead, n_factors))
            sim_factors[s] = z * np.sqrt(A_fc)

        # Transform back to original space
        sim = np.zeros((n_sim, n_ahead, k))
        for s in range(n_sim):
            sim[s] = (sim_factors[s] @ R.T) + mu

        result["sim"] = sim

    return result


# ----------------------------------------------------------------- GO-GARCH risk measures

def go_garch_risk(fit, alpha=0.05, horizon=1, n_sim=5000, seed=None):
    """
    Estimate VaR and ES from GO-GARCH model.
    """
    fc = go_garch_forecast(fit, n_ahead=horizon, method="MCS", n_sim=n_sim, seed=seed)
    sim = fc["sim"]  # (n_sim, horizon, k)
    agg = np.sum(sim, axis=1)  # (n_sim, k)

    VaR = np.percentile(agg, alpha * 100, axis=0)
    ES = np.mean(agg[agg <= VaR], axis=0)

    return {"VaR": VaR, "ES": ES}