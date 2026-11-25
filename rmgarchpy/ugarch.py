"""
rmgarchpy/ugarch.py
Univariate GARCH engine: GARCH(p,q), EGARCH, GJR-GARCH.

Provides:
    ugarchfit    - MLE estimation
    ugarchfilter - recompute conditional vol given estimated params
    ugarchsim    - simulate paths from a fitted model
    ugarchforecast - multi-step ahead forecast

Model specifications follow the R rugarch/rmgarch conventions:
    GARCH(p,q)   : sigma2_t = omega + sum alpha_i * r_{t-i}^2 + sum beta_j * sigma2_{t-j}
    GJR(p,q)     : sigma2_t = omega + sum (alpha_i + gamma_i * I(r_{t-i}<0)) * r_{t-i}^2
                   + sum beta_j * sigma2_{t-j}
    EGARCH(p,q)  : log(sigma2_t) = omega + sum alpha_i * g(z_{t-i})
                   + sum gamma_i * (|z_{t-i}| - E|z|) + sum beta_j * log(sigma2_{t-j})
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from . import utils


# ----------------------------------------------------------------- helpers

def _validate_order(p, q):
    p = int(p); q = int(q)
    if p < 0 or q < 0:
        raise ValueError("p and q must be non-negative")
    if p == 0 and q == 0:
        raise ValueError("At least one of p or q must be positive")
    return p, q


def _default_params(model, p, q):
    """Return a sensible starting parameter vector."""
    if model == "GARCH":
        omega = 0.05
        alpha = [0.1] * p
        beta = [0.8] * max(q - 1, 0)
        if q > 1:
            beta[-1] = 0.05
        return np.array([omega] + alpha + beta, dtype=float)
    if model == "GJR":
        omega = 0.05
        alpha = [0.08] * p
        gamma = [0.04] * p
        beta = [0.8] * max(q - 1, 0)
        if q > 1:
            beta[-1] = 0.05
        return np.array([omega] + alpha + gamma + beta, dtype=float)
    if model == "EGARCH":
        omega = 0.0
        alpha = [0.1] * p
        gamma = [0.05] * p
        beta = [0.95] * q
        return np.array([omega] + alpha + gamma + beta, dtype=float)
    raise ValueError(f"Unknown model: {model}")


def _param_names(model, p, q):
    if model == "GARCH":
        return (["omega"] + [f"alpha{i}" for i in range(1, p + 1)]
                + [f"beta{i}" for i in range(1, q + 1)])
    if model in ("GJR", "EGARCH"):
        return (["omega"] + [f"alpha{i}" for i in range(1, p + 1)]
                + [f"gamma{i}" for i in range(1, p + 1)]
                + [f"beta{i}" for i in range(1, q + 1)])
    raise ValueError(f"Unknown model: {model}")


def _unpack_params(model, params, p, q):
    omega = params[0]
    idx = 1
    alpha = params[idx:idx + p]; idx += p
    gamma = None
    if model in ("GJR", "EGARCH"):
        gamma = params[idx:idx + p]; idx += p
    beta = params[idx:idx + q]; idx += q
    return omega, alpha, gamma, beta


def _constraints(model, p, q):
    """Parameter bounds and stationarity constraints."""
    bounds = [(1e-8, None)]  # omega
    if model == "GARCH":
        bounds += [(0.0, 1.0)] * p
        bounds += [(0.0, 1.0)] * q
    elif model == "GJR":
        bounds += [(0.0, 1.0)] * p
        bounds += [(0.0, 1.0)] * p
        bounds += [(0.0, 1.0)] * q
    elif model == "EGARCH":
        bounds += [(None, None)] * (1 + 2 * p)
        bounds += [(-0.999, 0.999)] * q
    return bounds


def _check_stationarity(model, params, p, q):
    omega, alpha, gamma, beta = _unpack_params(model, params, p, q)
    if model == "GARCH":
        return np.sum(alpha) + np.sum(beta) < 1.0
    if model == "GJR":
        return np.sum(alpha + gamma / 2.0) + np.sum(beta) < 1.0
    if model == "EGARCH":
        # Nelson & Cao stationarity condition (sufficient approximation)
        return np.sum(np.abs(beta)) < 1.0
    return True


# ----------------------------------------------------------------- recursion

def _garch_sigma2(params, r, p, q, model="GARCH"):
    """Return sigma2 array (length n) and standardized residuals z."""
    omega, alpha, gamma, beta = _unpack_params(model, params, p, q)
    n = len(r)
    sigma2 = np.zeros(n)
    z = np.zeros(n)

    # initial values
    var0 = np.var(r)
    sigma2[:max(p, q)] = var0

    for t in range(n):
        if t == 0:
            sigma2[t] = var0
        else:
            s2 = omega
            for i in range(1, p + 1):
                if t - i >= 0:
                    lag_r2 = r[t - i] ** 2
                    if model == "GARCH":
                        s2 += alpha[i - 1] * lag_r2
                    elif model == "GJR":
                        s2 += (alpha[i - 1] + gamma[i - 1] * (r[t - i] < 0)) * lag_r2
            for j in range(1, q + 1):
                if t - j >= 0:
                    s2 += beta[j - 1] * sigma2[t - j]
            sigma2[t] = max(s2, 1e-12)
        z[t] = r[t] / np.sqrt(sigma2[t])
    return sigma2, z


def _egarch_sigma2(params, r, p, q):
    omega, alpha, gamma, beta = _unpack_params("EGARCH", params, p, q)
    n = len(r)
    sigma2 = np.zeros(n)
    z = np.zeros(n)
    log_sigma2 = np.zeros(n)
    Eabsz = np.sqrt(2 / np.pi)

    var0 = np.var(r)
    log_sigma2[:] = np.log(var0)

    for t in range(n):
        if t == 0:
            log_sigma2[t] = np.log(var0)
        else:
            ls = omega
            for i in range(1, p + 1):
                if t - i >= 0:
                    zlag = z[t - i]
                    ls += alpha[i - 1] * zlag
                    ls += gamma[i - 1] * (np.abs(zlag) - Eabsz)
            for j in range(1, q + 1):
                if t - j >= 0:
                    ls += beta[j - 1] * log_sigma2[t - j]
            log_sigma2[t] = ls
        sigma2[t] = np.exp(np.clip(log_sigma2[t], -20, 20))
        z[t] = r[t] / np.sqrt(sigma2[t])
    return sigma2, z


def _sigma2_recursion(params, r, p, q, model):
    if model == "EGARCH":
        return _egarch_sigma2(params, r, p, q)
    return _garch_sigma2(params, r, p, q, model=model)


# ----------------------------------------------------------------- likelihood

def _loglik(params, r, p, q, model, dist="normal"):
    try:
        sigma2, z = _sigma2_recursion(params, r, p, q, model)
    except Exception:
        return 1e10
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return 1e10
    if dist == "normal":
        ll = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2) - 0.5 * z ** 2
    elif dist == "t":
        nu = 5.0  # fixed for now; extended version can estimate nu
        ll = stats.t.logpdf(z, df=nu) - 0.5 * np.log(sigma2)
    else:
        raise ValueError(f"Unknown dist: {dist}")
    return -np.sum(ll)


# ----------------------------------------------------------------- public API

class UGARCHFitResult:
    def __init__(self, params, names, model, p, q, dist, data,
                 sigma2, z, loglik, nobs, converged, message,
                 hessian=None, std_errors=None):
        self.params = params
        self.names = names
        self.model = model
        self.p = p
        self.q = q
        self.dist = dist
        self.data = data
        self.sigma2 = sigma2
        self.z = z
        self.loglik = loglik
        self.nobs = nobs
        self.converged = converged
        self.message = message
        self.hessian = hessian
        self.std_errors = std_errors

    def __repr__(self):
        lines = [f"UGARCHFitResult(model={self.model}, p={self.p}, q={self.q}, dist={self.dist})"]
        for n, v in zip(self.names, self.params):
            lines.append(f"  {n:8s} = {v:.6f}")
        lines.append(f"  loglik   = {self.loglik:.4f}")
        lines.append(f"  converged= {self.converged}")
        return "\n".join(lines)

    def summary(self):
        print(self)
        if self.std_errors is not None:
            print("\nStandard errors:")
            for n, v, se in zip(self.names, self.params, self.std_errors):
                print(f"  {n:8s} = {v:.6f}  (se={se:.6f})")


def ugarchfit(data, model="GARCH", p=1, q=1, dist="normal",
              start_params=None, method="L-BFGS-B", maxiter=500):
    """
    Fit a univariate GARCH model by MLE.

    Parameters
    ----------
    data : array-like
        Return series (length n).
    model : str
        'GARCH', 'GJR', or 'EGARCH'.
    p, q : int
        ARCH and GARCH orders.
    dist : str
        'normal' or 't' (t uses fixed df=5 for now).
    start_params : array-like, optional
        Starting values.
    method : str
        scipy.optimize method.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    UGARCHFitResult
    """
    r = np.asarray(data, dtype=float).ravel()
    n = len(r)
    p, q = _validate_order(p, q)
    model = model.upper()

    x0 = start_params if start_params is not None else _default_params(model, p, q)
    x0 = np.asarray(x0, dtype=float)
    bounds = _constraints(model, p, q)

    # stationarity penalty
    def objective(params):
        ll = _loglik(params, r, p, q, model, dist)
        if not _check_stationarity(model, params, p, q):
            ll += 1e6
        return ll

    res = minimize(objective, x0, method=method, bounds=bounds,
                   options={"maxiter": maxiter, "disp": False})

    sigma2, z = _sigma2_recursion(res.x, r, p, q, model)
    loglik = -objective(res.x)

    # Hessian and standard errors via numeric differentiation
    hess = utils.num_hessian(objective, res.x)
    try:
        cov = np.linalg.inv(hess)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(res.x), np.nan)

    return UGARCHFitResult(
        params=res.x,
        names=_param_names(model, p, q),
        model=model, p=p, q=q, dist=dist,
        data=r, sigma2=sigma2, z=z,
        loglik=loglik, nobs=n,
        converged=res.success, message=res.message,
        hessian=hess, std_errors=se
    )


def ugarchfilter(data, fit):
    """
    Recompute conditional volatility and residuals using fitted parameters.

    Parameters
    ----------
    data : array-like
        Return series.
    fit : UGARCHFitResult
        Fitted model.

    Returns
    -------
    dict with keys 'sigma2', 'z', 'params'
    """
    r = np.asarray(data, dtype=float).ravel()
    sigma2, z = _sigma2_recursion(fit.params, r, fit.p, fit.q, fit.model)
    return {"sigma2": sigma2, "z": z, "params": fit.params}


def ugarchsim(fit, n=1000, n_start=100, seed=None):
    """
    Simulate paths from a fitted univariate GARCH model.

    Parameters
    ----------
    fit : UGARCHFitResult
    n : int
        Number of simulated observations.
    n_start : int
        Burn-in length discarded before returning.
    seed : int, optional

    Returns
    -------
    dict with 'r', 'sigma2', 'z'
    """
    rng = np.random.default_rng(seed)
    total = n + n_start
    omega, alpha, gamma, beta = _unpack_params(fit.model, fit.params, fit.p, fit.q)

    r = np.zeros(total)
    sigma2 = np.zeros(total)
    z = np.zeros(total)

    var0 = np.var(fit.data) if hasattr(fit, "data") else 0.01
    sigma2[0] = var0

    for t in range(1, total):
        if fit.model == "EGARCH":
            Eabsz = np.sqrt(2 / np.pi)
            ls = omega
            for i in range(1, fit.p + 1):
                if t - i >= 0:
                    zlag = z[t - i]
                    ls += alpha[i - 1] * zlag + gamma[i - 1] * (np.abs(zlag) - Eabsz)
            for j in range(1, fit.q + 1):
                if t - j >= 0:
                    ls += beta[j - 1] * np.log(sigma2[t - j])
            sigma2[t] = np.exp(np.clip(ls, -20, 20))
        else:
            s2 = omega
            for i in range(1, fit.p + 1):
                if t - i >= 0:
                    lag_r2 = r[t - i] ** 2
                    if fit.model == "GARCH":
                        s2 += alpha[i - 1] * lag_r2
                    elif fit.model == "GJR":
                        s2 += (alpha[i - 1] + gamma[i - 1] * (r[t - i] < 0)) * lag_r2
            for j in range(1, fit.q + 1):
                if t - j >= 0:
                    s2 += beta[j - 1] * sigma2[t - j]
            sigma2[t] = max(s2, 1e-12)

        if fit.dist == "normal":
            z[t] = rng.standard_normal()
        else:
            z[t] = rng.standard_t(df=5)
        r[t] = np.sqrt(sigma2[t]) * z[t]

    return {
        "r": r[n_start:],
        "sigma2": sigma2[n_start:],
        "z": z[n_start:]
    }


def ugarchforecast(fit, n_ahead=10, data=None):
    """
    Multi-step ahead conditional variance forecast.

    Parameters
    ----------
    fit : UGARCHFitResult
    n_ahead : int
    data : array-like, optional
        If provided, uses this series instead of fit.data for filtering.

    Returns
    -------
    dict with 'sigma2_forecast' (length n_ahead), 'last_sigma2'
    """
    r = np.asarray(data, dtype=float).ravel() if data is not None else fit.data
    sigma2, _ = _sigma2_recursion(fit.params, r, fit.p, fit.q, fit.model)
    last_sigma2 = sigma2[-1]

    omega, alpha, gamma, beta = _unpack_params(fit.model, fit.params, fit.p, fit.q)
    forecasts = np.zeros(n_ahead)
    hist = list(sigma2[-max(fit.p, fit.q):])
    last_r2 = r[-1] ** 2

    for h in range(n_ahead):
        if fit.model == "EGARCH":
            # approximate: E[log sigma2] recursion
            ls = omega
            for i in range(1, fit.p + 1):
                ls += alpha[i - 1] * 0.0  # E[z]=0
                ls += gamma[i - 1] * (np.sqrt(2 / np.pi) - np.sqrt(2 / np.pi))
            for j in range(1, fit.q + 1):
                prev = forecasts[h - j] if h - j >= 0 else last_sigma2
                ls += beta[j - 1] * np.log(prev)
            forecasts[h] = np.exp(np.clip(ls, -20, 20))
        else:
            s2 = omega
            for i in range(1, fit.p + 1):
                lag_r2 = last_r2 if h == 0 and i == 1 else forecasts[h - i] if h - i >= 0 else last_sigma2
                if fit.model == "GARCH":
                    s2 += alpha[i - 1] * lag_r2
                elif fit.model == "GJR":
                    s2 += (alpha[i - 1] + gamma[i - 1] * 0.5) * lag_r2
            for j in range(1, fit.q + 1):
                prev = forecasts[h - j] if h - j >= 0 else last_sigma2
                s2 += beta[j - 1] * prev
            forecasts[h] = max(s2, 1e-12)

    return {"sigma2_forecast": forecasts, "last_sigma2": last_sigma2}