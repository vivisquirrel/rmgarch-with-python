"""Shared utilities: matrix helpers, positive-definite projection,
numerical differentiation, standard normal / t helpers."""

import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, solve_triangular


# ----------------------------------------------------------------- matrix ops

def nearPD(A, max_iter=50, tol=1e-10):
    """Higham's nearest positive-definite correlation/covariance matrix."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    Y = A.copy()
    delta = np.zeros((n, n))
    for _ in range(max_iter):
        X = Y - delta
        R = X - np.diag(np.diag(X)) + np.eye(n) * np.diag(X)
        U, s, Vt = np.linalg.svd(R)
        Xnew = U @ np.diag(np.maximum(s, 1e-12)) @ Vt
        delta_new = Xnew - X
        if np.max(np.abs(delta_new)) < tol:
            break
        Y = Xnew
        delta = delta_new
    P = Y - delta
    return (P + P.T) / 2.0


def isPD(A, tol=1e-12):
    try:
        np.linalg.cholesky(np.asarray(A, dtype=float) + tol * np.eye(len(A)))
        return True
    except np.linalg.LinAlgError:
        return False


def sqrtm(A):
    """Symmetric matrix square root via eigen-decomposition."""
    A = np.asarray(A, dtype=float)
    w, V = np.linalg.eigh((A + A.T) / 2)
    w = np.clip(w, 1e-12, None)
    return (V * np.sqrt(w)) @ V.T


def inv_sqrtm(A):
    A = np.asarray(A, dtype=float)
    w, V = np.linalg.eigh((A + A.T) / 2)
    w = np.clip(w, 1e-12, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def chol_inv_sqrtm(A):
    """Lower-triangular L with L @ L.T = A (Cholesky-based square root)."""
    A = np.asarray(A, dtype=float)
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(nearPD(A))


def outer_product(v):
    v = np.asarray(v).reshape(-1, 1)
    return v @ v.T


def corr2cov(R, sd):
    sd = np.asarray(sd, dtype=float)
    D = np.diag(sd)
    return D @ R @ D


def cov2corr(C):
    C = np.asarray(C, dtype=float)
    d = np.sqrt(np.diag(C))
    d[d == 0] = 1.0
    D = np.diag(1.0 / d)
    return D @ C @ D


# ----------------------------------------------------------------- log-lik helpers

def dnorm_logpdf(x):
    return -0.5 * np.log(2 * np.pi) - 0.5 * x ** 2


def dt_logpdf(x, nu):
    return stats.t.logpdf(x, df=nu)


def safe_log(x, eps=1e-300):
    return np.log(np.maximum(x, eps))


# ----------------------------------------------------------------- numeric derivatives

def num_grad(f, x0, eps=1e-5):
    """Central-difference gradient of scalar function f at x0."""
    x0 = np.asarray(x0, dtype=float)
    g = np.zeros_like(x0)
    for i in range(len(x0)):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def num_hessian(f, x0, eps=1e-4):
    """Central-difference Hessian of scalar function f at x0."""
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    H = np.zeros((n, n))
    f0 = f(x0)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp = x0.copy(); xp[i] += eps
                xm = x0.copy(); xm[i] -= eps
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / eps ** 2
            else:
                xpp = x0.copy(); xpp[i] += eps; xpp[j] += eps
                xpm = x0.copy(); xpm[i] += eps; xpm[j] -= eps
                xmp = x0.copy(); xmp[i] -= eps; xmp[j] += eps
                xmm = x0.copy(); xmm[i] -= eps; xmm[j] -= eps
                H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps ** 2)
    return (H + H.T) / 2


# ----------------------------------------------------------------- misc

def as_matrix(data):
    """Coerce pandas/numpy input to float 2-D array (n x k)."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def rolling_mean(x, k):
    """Centered rolling mean, edges shrunk (for semiparametric kernels)."""
    x = np.asarray(x, dtype=float)
    out = np.array([x[i:i + k].mean() for i in range(len(x) - k + 1)])
    return out


def bandwidth_default(n):
    """Silverman-like bandwidth for semiparametric DCC."""
    return int(np.ceil(2 * np.sqrt(n)))