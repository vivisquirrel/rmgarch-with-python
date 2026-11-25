"""
rmgarchpy/mdcopula.py
Multivariate copula toolbox (port of R rmgarch::mdcopula).

Provides:
    mdKendall        - Kendall's tau matrix
    mdTailDep        - upper/lower/middle tail dependence matrix
    mdCopulaParam    - pairwise copula parameter estimation (with rotations)
    mdCopulaGof      - goodness-of-fit test (CvM with parametric bootstrap)
    mdCopulaSim      - simulate from a copula
    mdStaticCopula   - static copula fit on pseudo-observations
    mdDynamicCopula  - dynamic copula (time-varying parameter via rolling window)

Supported families: normal, t, clayton, gumbel, frank (+ rotations 90/180/270).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize
from itertools import combinations

from . import utils


# ----------------------------------------------------------------- Kendall tau

def _kendall_tau_pair(x, y):
    """Kendall's tau for a pair (no ties correction beyond basic)."""
    n = len(x)
    if n < 3:
        return np.nan
    # Use scipy for speed and correctness
    tau, _ = stats.kendalltau(x, y)
    return tau


def mdKendall(data):
    """
    Kendall's tau matrix for multivariate data.

    Parameters
    ----------
    data : array-like, shape (n, k)

    Returns
    -------
    ndarray, shape (k, k)
    """
    X = utils.as_matrix(data)
    n, k = X.shape
    T = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            tau = _kendall_tau_pair(X[:, i], X[:, j])
            T[i, j] = T[j, i] = tau
    return T


# ----------------------------------------------------------------- Tail dependence

def _tail_dep_pair(x, y, tail="upper"):
    """
    Nonparametric tail dependence estimator for a pair.
    tail: 'upper', 'lower', or 'middle'
    """
    n = len(x)
    if n < 10:
        return np.nan
    # ranks -> empirical cdf
    rx = stats.rankdata(x) / (n + 1)
    ry = stats.rankdata(y) / (n + 1)

    if tail == "upper":
        # P(Y > F_Y^{-1}(1-1/n) | X > F_X^{-1}(1-1/n))
        ux = np.sum(rx > 1 - 1.0 / n)
        uy = np.sum(ry > 1 - 1.0 / n)
        uxy = np.sum((rx > 1 - 1.0 / n) & (ry > 1 - 1.0 / n))
        denom = min(ux, uy)
        return uxy / denom if denom > 0 else np.nan
    elif tail == "lower":
        lx = np.sum(rx <= 1.0 / n)
        ly = np.sum(ry <= 1.0 / n)
        lxy = np.sum((rx <= 1.0 / n) & (ry <= 1.0 / n))
        denom = min(lx, ly)
        return lxy / denom if denom > 0 else np.nan
    elif tail == "middle":
        # middle tail dependence: focus on (0.4,0.6) region
        mx = np.sum((rx > 0.4) & (rx < 0.6))
        my = np.sum((ry > 0.4) & (ry < 0.6))
        mxy = np.sum((rx > 0.4) & (rx < 0.6) & (ry > 0.4) & (ry < 0.6))
        denom = min(mx, my)
        return mxy / denom if denom > 0 else np.nan
    else:
        raise ValueError("tail must be 'upper', 'lower', or 'middle'")


def mdTailDep(data, tail="upper"):
    """
    Tail dependence matrix.

    Parameters
    ----------
    data : array-like, shape (n, k)
    tail : str
        'upper', 'lower', or 'middle'

    Returns
    -------
    ndarray, shape (k, k)
    """
    X = utils.as_matrix(data)
    n, k = X.shape
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            lam = _tail_dep_pair(X[:, i], X[:, j], tail=tail)
            D[i, j] = D[j, i] = lam
    return D


# ----------------------------------------------------------------- Copula families

def _clayton_cdf(u, v, theta):
    if theta <= 0:
        return u * v
    return (np.power(u, -theta) + np.power(v, -theta) - 1) ** (-1.0 / theta)


def _clayton_pdf(u, v, theta):
    if theta <= 0:
        return np.ones_like(u)
    return (1 + theta) * np.power(u * v, -(1 + theta)) * \
           np.power(np.power(u, -theta) + np.power(v, -theta) - 1, -(2 + 1.0 / theta))


def _gumbel_cdf(u, v, theta):
    if theta <= 1:
        return u * v
    a = -np.log(u)
    b = -np.log(v)
    return np.exp(-np.power(np.power(a, theta) + np.power(b, theta), 1.0 / theta))


def _gumbel_pdf(u, v, theta):
    if theta <= 1:
        return np.ones_like(u)
    a = -np.log(u)
    b = -np.log(v)
    s = np.power(a, theta) + np.power(b, theta)
    t = np.power(s, 1.0 / theta)
    return np.exp(-t) * np.power(u * v, -1) * \
           np.power(a * b, theta - 1) * \
           np.power(s, (2 - theta) / theta) * \
           (1 + (theta - 1) * t)


def _frank_cdf(u, v, theta):
    if np.abs(theta) < 1e-8:
        return u * v
    num = (1 - np.exp(-theta * u)) * (1 - np.exp(-theta * v))
    den = 1 - np.exp(-theta)
    return -np.log(1 - num / den) / theta


def _frank_pdf(u, v, theta):
    if np.abs(theta) < 1e-8:
        return np.ones_like(u)
    e = np.exp(-theta)
    num = -theta * e * (1 + (e - 1) * np.exp(-theta * (u + v)))
    den = np.power(1 - e * np.exp(-theta * u) - np.exp(-theta * v) + e * np.exp(-theta * (u + v)), 2)
    return num / den


def _normal_copula_cdf(u, v, rho):
    if np.abs(rho) >= 1:
        rho = np.clip(rho, -0.999999, 0.999999)
    from scipy.stats import multivariate_normal
    return multivariate_normal.cdf([np.array([stats.norm.ppf(u), stats.norm.ppf(v)])],
                                   mean=[0, 0], cov=[[1, rho], [rho, 1]])[0]


def _normal_copula_pdf(u, v, rho):
    if np.abs(rho) >= 1:
        rho = np.clip(rho, -0.999999, 0.999999)
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)
    det = 1 - rho ** 2
    quad = (x ** 2 + y ** 2 - 2 * rho * x * y) / det
    return np.exp(-0.5 * quad) / det


def _t_copula_cdf(u, v, rho, nu):
    if np.abs(rho) >= 1:
        rho = np.clip(rho, -0.999999, 0.999999)
    from scipy.stats import multivariate_t
    x = stats.t.ppf(u, df=nu)
    y = stats.t.ppf(v, df=nu)
    cov = [[1, rho], [rho, 1]]
    return multivariate_t.cdf([x, y], loc=[0, 0], cov=cov, df=nu)


def _t_copula_pdf(u, v, rho, nu):
    if np.abs(rho) >= 1:
        rho = np.clip(rho, -0.999999, 0.999999)
    from scipy.stats import multivariate_t
    x = stats.t.ppf(u, df=nu)
    y = stats.t.ppf(v, df=nu)
    cov = [[1, rho], [rho, 1]]
    return multivariate_t.pdf([x, y], loc=[0, 0], cov=cov, df=nu)


# ----------------------------------------------------------------- rotations

def _rotate_90(u, v):
    return 1 - u, v


def _rotate_180(u, v):
    return 1 - u, 1 - v


def _rotate_270(u, v):
    return u, 1 - v


def _apply_rotation(u, v, rotation):
    if rotation == 0:
        return u, v
    if rotation == 90:
        return _rotate_90(u, v)
    if rotation == 180:
        return _rotate_180(u, v)
    if rotation == 270:
        return _rotate_270(u, v)
    raise ValueError("rotation must be 0, 90, 180, or 270")


# ----------------------------------------------------------------- parameter estimation

def _kendall_to_param(family, tau, rotation=0):
    """Method-of-moments starting value from Kendall tau."""
    # Adjust tau for rotation
    if rotation == 90 or rotation == 270:
        tau = -tau
    elif rotation == 180:
        pass  # tau unchanged for 180 rotation of symmetric copulas

    if family == "clayton":
        if tau <= -1 or tau >= 1:
            return 1.0
        return 2 * tau / (1 - tau) if tau < 1 else 10.0
    if family == "gumbel":
        if tau >= 1:
            return 10.0
        return 1.0 / (1 - tau) if tau > 0 else 1.0
    if family == "frank":
        # approximate inversion
        if np.abs(tau) < 1e-6:
            return 0.0
        return 3 * tau  # crude start
    if family in ("normal", "t"):
        return np.sin(np.pi * tau / 2)
    raise ValueError(f"Unknown family: {family}")


def _copula_loglik(theta, u, v, family, rotation=0, nu=5):
    try:
        ur, vr = _apply_rotation(u, v, rotation)
        ur = np.clip(ur, 1e-10, 1 - 1e-10)
        vr = np.clip(vr, 1e-10, 1 - 1e-10)
        if family == "clayton":
            pdf = _clayton_pdf(ur, vr, theta[0])
        elif family == "gumbel":
            pdf = _gumbel_pdf(ur, vr, theta[0])
        elif family == "frank":
            pdf = _frank_pdf(ur, vr, theta[0])
        elif family == "normal":
            pdf = _normal_copula_pdf(ur, vr, theta[0])
        elif family == "t":
            pdf = _t_copula_pdf(ur, vr, theta[0], nu)
        else:
            raise ValueError(f"Unknown family: {family}")
        pdf = np.clip(pdf, 1e-300, 1e300)
        return -np.sum(np.log(pdf))
    except Exception:
        return 1e10


def mdCopulaParam(u, v, family="clayton", rotation=0, method="ML", nu=5):
    """
    Estimate pairwise copula parameter.

    Parameters
    ----------
    u, v : array-like
        Pseudo-observations in (0,1).
    family : str
        'clayton', 'gumbel', 'frank', 'normal', 't'
    rotation : int
        0, 90, 180, 270
    method : str
        'ML' (maximum likelihood) or 'MoM' (method of moments)
    nu : float
        Degrees of freedom for t-copula.

    Returns
    -------
    dict with 'theta', 'se', 'loglik', 'family', 'rotation'
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(u)

    tau = _kendall_tau_pair(u, v)
    theta0 = _kendall_to_param(family, tau, rotation)

    if method.upper() == "MOM":
        return {"theta": theta0, "se": np.nan, "loglik": np.nan,
                "family": family, "rotation": rotation}

    # ML estimation
    if family == "clayton":
        bounds = [(0.01, 50.0)]
    elif family == "gumbel":
        bounds = [(1.0, 50.0)]
    elif family == "frank":
        bounds = [(-20.0, 20.0)]
    elif family in ("normal", "t"):
        bounds = [(-0.999, 0.999)]
    else:
        raise ValueError(f"Unknown family: {family}")

    res = minimize(_copula_loglik, [theta0], args=(u, v, family, rotation, nu),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "disp": False})

    theta = res.x[0]
    loglik = -res.fun

    # standard error via numeric Hessian
    hess = utils.num_hessian(lambda th: _copula_loglik(th, u, v, family, rotation, nu), res.x)
    try:
        se = np.sqrt(1.0 / hess[0, 0]) if hess[0, 0] > 0 else np.nan
    except Exception:
        se = np.nan

    return {"theta": theta, "se": se, "loglik": loglik,
            "family": family, "rotation": rotation}


# ----------------------------------------------------------------- goodness of fit

def _empirical_copula(u, v):
    """Empirical copula C_n(u,v)."""
    n = len(u)
    def C(u0, v0):
        return np.mean((u <= u0) & (v <= v0))
    return C


def _cvm_statistic(u, v, family, theta, rotation=0, nu=5):
    """Cramer-von Mises statistic for copula fit."""
    ur, vr = _apply_rotation(u, v, rotation)
    ur = np.clip(ur, 1e-10, 1 - 1e-10)
    vr = np.clip(vr, 1e-10, 1 - 1e-10)
    n = len(u)

    # empirical copula at each point
    C_emp = np.array([np.mean((ur <= ur[i]) & (vr <= vr[i])) for i in range(n)])

    # theoretical copula
    if family == "clayton":
        C_th = _clayton_cdf(ur, vr, theta)
    elif family == "gumbel":
        C_th = _gumbel_cdf(ur, vr, theta)
    elif family == "frank":
        C_th = _frank_cdf(ur, vr, theta)
    elif family == "normal":
        C_th = np.array([_normal_copula_cdf(a, b, theta) for a, b in zip(ur, vr)])
    elif family == "t":
        C_th = np.array([_t_copula_cdf(a, b, theta, nu) for a, b in zip(ur, vr)])
    else:
        raise ValueError(f"Unknown family: {family}")

    return np.sum((C_emp - C_th) ** 2) / n


def mdCopulaGof(u, v, family="clayton", rotation=0, n_boot=199, nu=5):
    """
    Goodness-of-fit test for a copula family (CvM with parametric bootstrap).

    Parameters
    ----------
    u, v : array-like
    family : str
    rotation : int
    n_boot : int
        Number of bootstrap samples.
    nu : float
        df for t-copula.

    Returns
    -------
    dict with 'statistic', 'p_value', 'critical_values'
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(u)

    # estimate parameter
    est = mdCopulaParam(u, v, family=family, rotation=rotation, method="ML", nu=nu)
    theta = est["theta"]

    # observed statistic
    S_obs = _cvm_statistic(u, v, family, theta, rotation, nu)

    # parametric bootstrap
    rng = np.random.default_rng()
    S_boot = []
    for _ in range(n_boot):
        sim = mdCopulaSim(n, family=family, theta=theta, rotation=rotation, nu=nu, seed=rng)
        ub, vb = sim[:, 0], sim[:, 1]
        # re-estimate on bootstrap sample
        try:
            est_b = mdCopulaParam(ub, vb, family=family, rotation=rotation, method="ML", nu=nu)
            S_b = _cvm_statistic(ub, vb, family, est_b["theta"], rotation, nu)
            S_boot.append(S_b)
        except Exception:
            continue

    S_boot = np.array(S_boot)
    p_value = np.mean(S_boot >= S_obs) if len(S_boot) > 0 else np.nan

    return {
        "statistic": S_obs,
        "p_value": p_value,
        "critical_values": np.percentile(S_boot, [90, 95, 99]) if len(S_boot) > 0 else None
    }


# ----------------------------------------------------------------- simulation

def _sim_clayton(n, theta, rng):
    if theta <= 0:
        return rng.random((n, 2))
    V = rng.random(n)
    W = rng.random(n)
    U = np.power(1 + theta * (-np.log(V)) / (-np.log(W)), -1.0 / theta)
    # more stable: use conditional inversion
    U = np.zeros(n)
    V = np.zeros(n)
    for i in range(n):
        w = rng.random()
        v = rng.random()
        U[i] = np.power(1 + theta * (-np.log(v)) / (-np.log(w)), -1.0 / theta) if w > 0 else 0
        V[i] = w
    return np.column_stack([U, V])


def _sim_gumbel(n, theta, rng):
    if theta <= 1:
        return rng.random((n, 2))
    # Marshall-Olkin type algorithm for Gumbel
    U = np.zeros(n)
    V = np.zeros(n)
    for i in range(n):
        w = rng.random()
        v = rng.random()
        # conditional distribution
        U[i] = np.exp(-np.power(-np.log(v) / w, 1.0 / (theta - 1)) * np.power(-np.log(w), 1.0 / theta))
        V[i] = w
    return np.column_stack([U, V])


def _sim_frank(n, theta, rng):
    if np.abs(theta) < 1e-8:
        return rng.random((n, 2))
    U = np.zeros(n)
    V = np.zeros(n)
    for i in range(n):
        w = rng.random()
        v = rng.random()
        # inversion of conditional Frank
        U[i] = -np.log(1 - w * (1 - np.exp(-theta * v)) / (1 - np.exp(-theta))) / theta
        V[i] = v