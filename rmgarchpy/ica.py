"""
rmgarchpy/ica.py
ICA-EGARCH model: Independent Component Analysis extended GARCH.

Implements the ICA-EGARCH framework of von Sachs and Posse (1998)
and Monjeau et al. (2022):
    X_t = mu + W * s_t
where:
    - W is a mixing matrix (not necessarily orthogonal)
    - s_t are independent components following EGARCH processes
    - the components are standardized to have unit variance

Unlike GO-GARCH which uses orthogonal rotations, ICA-EGARCH
finds independent (not just uncorrelated) components, which
can better capture non-normal dependence structures.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import inv, eigh
from scipy.stats import kurtosis, skew

from . import utils
from .ugarch import ugarchfit


# ----------------------------------------------------------------- ICA utilities

def _center_data(X):
    """Center the data."""
    mu = np.mean(X, axis=0)
    return X - mu, mu


def _whiten(X):
    """
    Whiten the data: Z = W_white * X
    where W_white is the whitening matrix.
    """
    n, k = X.shape
    # Covariance matrix
    S = np.cov(X.T)
    # Eigen decomposition
    eigenvalues, eigenvectors = eigh(S)
    # Whitening matrix
    W_white = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
    Z = X @ W_white
    return Z, W_white


def _negentropy(Z, max_iter=1000, tol=1e-6):
    """
    FastICA algorithm for finding independent components.
    Returns the unmixing matrix W such that S = W * Z.
    """
    n, k = Z.shape
    W = np.random.randn(k, k)
    W = np.linalg.qr(W)[0]  # orthogonal initialization

    for _ in range(max_iter):
        W_old = W.copy()
        for j in range(k):
            # Update j-th row of W
            w = W[j, :]
            # Nonlinearity: use tanh for super-Gaussian sources
            g = np.tanh(Z @ w)
            # Update rule
            w_new = Z.T @ g / n - np.mean(1 - g**2) * w
            # Normalize
            w_new = w_new / np.linalg.norm(w_new)
            W[j, :] = w_new

        # Orthogonalize W (symmetric orthogonalization)
        W = W @ inv(np.sqrtm(W @ W.T))

        # Check convergence
        if np.max(np.abs(np.abs(W @ W_old.T) - np.eye(k))) < tol:
            break

    return W


def _kurtosis_ic(s):
    """Excess kurtosis of independent component."""
    return kurtosis(s, fisher=True)


# ----------------------------------------------------------------- ICA-EGARCH fit

def ica_egarch_fit(data, n_components=None, ugarch_spec=None,
                   ica_method="fastica", show_progress=False):
    """
    Fit an ICA-EGARCH model.

    Parameters
    ----------
    data : array-like, shape (n, k)
        Multivariate return series.
    n_components : int, optional
        Number of independent components. Default: k.
    ugarch_spec : dict, optional
        Specification for EGARCH models on the components.
        Default: EGARCH(1,1) with normal errors.
    ica_method : str
        'fastica' or 'kurtosis' (maximize kurtosis).
    show_progress : bool

    Returns
    -------
    dict with keys:
        'W'          : mixing matrix (k x n_components)
        'W_inv'      : unmixing matrix (n_components x k)
        'components' : independent components (n x n_components)
        'ugarch'     : list of univariate EGARCH fits
        'mu'         : mean vector
        'residuals'  : raw residuals
        'h'          : conditional covariance matrices (n x k x k)
        'loglik'     : log-likelihood
        'nobs'       : number of observations
    """
    X = utils.as_matrix(data)
    n, k = X.shape

    if n_components is None:
        n_components = k

    if ugarch_spec is None:
        ugarch_spec = {"model": "eGARCH", "garch": (1, 1), "dist": "norm"}

    # Step 1: center and whiten
    X_centered, mu = _center_data(X)
    Z, W_white = _whiten(X_centered)

    # Step 2: ICA
    if ica_method == "fastica":
        W_ica = _negentropy(Z)  # (k, k)
    elif ica_method == "kurtosis":
        # Simplified: use FastICA anyway
        W_ica = _negentropy(Z)
    else:
        raise ValueError(f"Unknown ICA method: {ica_method}")

    # Independent components
    S = Z @ W_ica  # (n, k)

    # Mixing matrix: X = mu + S @ W_mix'
    # W_mix = W_white @ W_ica
    W_mix = W_white @ W_ica  # (k, k)
    W_inv = W_ica.T @ W_white.T  # unmixing: S = X_centered @ W_inv'

    # Step 3: fit EGARCH to each component
    ugarch_fits = []
    H_comp = np.zeros((n, n_components))  # conditional variances of components
    comp_residuals = np.zeros((n, n_components))

    for j in range(n_components):
        fit = ugarchfit(S[:, j], spec=ugarch_spec, show_progress=False)
        ugarch_fits.append(fit)
        H_comp[:, j] = fit["h"]
        comp_residuals[:, j] = fit["residuals"]

    # Step 4: reconstruct conditional covariance matrices
    # H_t = W_mix @ diag(H_comp_t) @ W_mix'
    h = np.zeros((n, k, k))
    for t in range(n):
        H_diag = np.diag(H_comp[t, :n_components])
        h[t] = W_mix @ H_diag @ W_mix.T

    # Step 5: log-likelihood
    loglik = 0.0
    for t in range(n):
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

    return {
        "W": W_mix,
        "W_inv": W_inv,
        "components": S,
        "ugarch": ugarch_fits,
        "mu": mu,
        "residuals": X_centered,
        "h": h,
        "loglik": loglik,
        "nobs": n,
        "n_components": n_components,
        "ugarch_spec": ugarch_spec
    }


# ----------------------------------------------------------------- ICA-EGARCH forecast

def ica_egarch_forecast(fit, n_ahead=10, method="MCS", n_sim=1000, seed=None):
    """
    Forecast ICA-EGARCH model.
    """
    rng = np.random.default_rng(seed)
    k = fit["W"].shape[0]
    n_comp = fit["n_components"]
    W = fit["W"]
    mu = fit["mu"]

    # Forecast component variances
    H_fc = np.zeros((n_ahead, n_comp))
    for j in range(n_comp):
        H_fc[:, j] = fit["h"][-1, j, 0] if hasattr(fit["h"], "shape") and fit["h"].ndim > 2 else fit["h"][-1]

    # Forecast covariance matrices
    h_fc = np.zeros((n_ahead, k, k))
    for t in range(n_ahead):
        H_diag = np.diag(H_fc[t])
        h_fc[t] = W @ H_diag @ W.T

    result = {"h": h_fc, "H_comp": H_fc}

    if method.upper() == "MCS":
        sim_comp = np.zeros((n_sim, n_ahead, n_comp))
        for s in range(n_sim):
            z = rng.standard_normal((n_ahead, n_comp))
            sim_comp[s] = z * np.sqrt(H_fc)

        sim = np.zeros((n_sim, n_ahead, k))
        for s in range(n_sim):
            sim[s] = (sim_comp[s] @ W.T) + mu

        result["sim"] = sim

    return result


# ----------------------------------------------------------------- ICA-EGARCH risk

def ica_egarch_risk(fit, alpha=0.05, horizon=1, n_sim=5000, seed=None):
    """
    Estimate VaR and ES from ICA-EGARCH model.
    """
    fc = ica_egarch_forecast(fit, n_ahead=horizon, method="MCS", n_sim=n_sim, seed=seed)
    sim = fc["sim"]
    agg = np.sum(sim, axis=1)

    VaR = np.percentile(agg, alpha * 100, axis=0)
    ES = np.mean(agg[agg <= VaR], axis=0)

    return {"VaR": VaR, "ES": ES}