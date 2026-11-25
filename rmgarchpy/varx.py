"""
rmgarchpy/varx.py
Vector Autoregressive model with eXogenous variables (VARX).

Implements VARX(p, r) model:
    Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p}
          + B_1 X_t + ... + B_r X_{t-r}
          + e_t
where:
    - Y_t is a k-dimensional endogenous vector
    - X_t is an m-dimensional exogenous vector
    - A_i are k x k coefficient matrices
    - B_j are k x m coefficient matrices
    - e_t ~ N(0, H_t) where H_t can follow a GARCH process

Supports:
    - OLS estimation
    - lag selection via information criteria
    - impulse response functions
    - forecast error variance decomposition
    - integration with GARCH for conditional heteroskedasticity
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import inv, block_diag

from . import utils


# ----------------------------------------------------------------- VARX lag structure

def _build_lag_matrices(Y, X=None, lags_y=1, lags_x=1, constant=True):
    """
    Build the lagged design matrix for VARX.
    
    Parameters
    ----------
    Y : ndarray, shape (n, k)
        Endogenous variables.
    X : ndarray, shape (n, m), optional
        Exogenous variables.
    lags_y : int
        Number of lags for endogenous variables.
    lags_x : int
        Number of lags for exogenous variables.
    constant : bool
        Whether to include a constant term.
        
    Returns
    -------
    Z : ndarray, shape (n_max_lag, n - n_max_lag, k * lags_y + m * lags_x + 1)
        Design matrix with lags and constant.
    Y_target : ndarray, shape (n_max_lag, n - n_max_lag, k)
        Target values.
    """
    n, k = Y.shape
    if X is None:
        m = 0
        X = np.empty((n, 0))
    else:
        m = X.shape[1]
        
    n_max_lag = max(lags_y, lags_x)
    n_obs = n - n_max_lag
    
    if n_obs <= 0:
        raise ValueError("Not enough observations for the specified lags.")
        
    # Pre-allocate
    n_features = k * lags_y + m * lags_x + (1 if constant else 0)
    Z = np.zeros((n_obs, n_features))
    Y_target = np.zeros((n_obs, k))
    
    for t in range(n_max_lag, n):
        idx = t - n_max_lag
        
        # Target
        Y_target[idx] = Y[t]
        
        # Endogenous lags
        col_idx = 0
        for lag in range(1, lags_y + 1):
            Z[idx, col_idx:col_idx+k] = Y[t - lag]
            col_idx += k
            
        # Exogenous lags
        if m > 0:
            for lag in range(lags_x):
                Z[idx, col_idx:col_idx+m] = X[t - lag]
                col_idx += m
                
        # Constant
        if constant:
            Z[idx, col_idx] = 1.0
            
    return Z, Y_target


# ----------------------------------------------------------------- VARX estimation

def varx_fit(Y, X=None, lags_y=1, lags_x=1, constant=True, show_progress=False):
    """
    Fit a VARX model using OLS.
    
    Parameters
    ----------
    Y : array-like, shape (n, k)
        Endogenous variables (e.g., returns).
    X : array-like, shape (n, m), optional
        Exogenous variables.
    lags_y : int
        Number of lags for Y.
    lags_x : int
        Number of lags for X.
    constant : bool
        Include constant term.
    show_progress : bool
        
    Returns
    -------
    dict with keys:
        'coef'      : coefficient matrix (k x n_features)
        'residuals' : residuals (n_obs x k)
        'fitted'    : fitted values (n_obs x k)
        'sigma'     : residual covariance matrix (k x k)
        'loglik'    : log-likelihood
        'nobs'      : number of observations used
        'lags_y'    : lags for Y
        'lags_x'    : lags for X
        'constant'  : whether constant was included
    """
    Y = utils.as_matrix(Y)
    if X is not None:
        X = utils.as_matrix(X)
        
    n, k = Y.shape
    
    # Build lag matrices
    Z, Y_target = _build_lag_matrices(Y, X, lags_y, lags_x, constant)
    n_obs = Z.shape[0]
    
    # OLS estimation: Y' = Z @ B'  =>  B = (Z'Z)^{-1} Z'Y
    # Z is (n_obs, n_features), Y_target is (n_obs, k)
    # We want coef such that Y_target = Z @ coef.T
    
    ZT = Z.T
    ZTZ = ZT @ Z
    ZTY = ZT @ Y_target
    
    # Solve for coefficients
    try:
        coef_T = np.linalg.solve(ZTZ, ZTY)  # (n_features, k)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        coef_T = np.linalg.pinv(ZTZ) @ ZTY
        
    coef = coef_T.T  # (k, n_features)
    
    # Fitted values and residuals
    fitted = Z @ coef_T
    residuals = Y_target - fitted
    
    # Residual covariance
    sigma = (residuals.T @ residuals) / n_obs
    
    # Log-likelihood (assuming multivariate normal)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign > 0:
        loglik = -0.5 * n_obs * (k * np.log(2 * np.pi) + logdet + k)
    else:
        loglik = -1e10
        
    return {
        "coef": coef,
        "residuals": residuals,
        "fitted": fitted,
        "sigma": sigma,
        "loglik": loglik,
        "nobs": n_obs,
        "n_features": Z.shape[1],
        "lags_y": lags_y,
        "lags_x": lags_x,
        "constant": constant,
        "k": k
    }


# ----------------------------------------------------------------- VARX forecast

def varx_forecast(fit, Y, X=None, n_ahead=10, X_future=None):
    """
    Forecast VARX model.
    
    Parameters
    ----------
    fit : dict
        Output of varx_fit.
    Y : ndarray
        Historical endogenous variables.
    X : ndarray, optional
        Historical exogenous variables.
    n_ahead : int
        Forecast horizon.
    X_future : ndarray, optional
        Future exogenous variables (n_ahead x m).
        
    Returns
    -------
    dict with 'forecast' (n_ahead x k)
    """
    k = fit["k"]
    lags_y = fit["lags_y"]
    lags_x = fit["lags_x"]
    constant = fit["constant"]
    coef = fit["coef"]
    
    n_Y = Y.shape[0]
    if X is not None:
        m = X.shape[1]
    else:
        m = 0
        
    # Initialize forecast storage
    forecast = np.zeros((n_ahead, k))
    
    # We need to build the lag vector iteratively
    # Start with the last available lags
    
    # Create extended Y and X for forecasting
    Y_ext = Y.copy()
    if X is not None:
        X_ext = X.copy()
    else:
        X_ext = np.empty((n_Y, 0))
        
    if X_future is not None:
        X_ext = np.vstack([X_ext, X_future])
    else:
        X_ext = np.vstack([X_ext, np.zeros((n_ahead, m))])
        
    for h in range(n_ahead):
        t = n_Y + h
        
        # Build the feature vector for time t
        feat = np.zeros(fit["n_features"])
        col_idx = 0
        
        # Endogenous lags
        for lag in range(1, lags_y + 1):
            if t - lag < n_Y:
                feat[col_idx:col_idx+k] = Y_ext[t - lag]
            else:
                # Use forecasted values
                feat[col_idx:col_idx+k] = forecast[t - n_Y - lag]
            col_idx += k
            
        # Exogenous lags
        if m > 0:
            for lag in range(lags_x):
                feat[col_idx:col_idx+m] = X_ext[t - lag]
            col_idx += m
            
        # Constant
        if constant:
            feat[col_idx] = 1.0
            
        # Forecast
        forecast[h] = feat @ coef.T
        
    return {"forecast": forecast}


# ----------------------------------------------------------------- Impulse Response Function

def varx_irf(fit, n_ahead=10, orthogonalize=True):
    """
    Compute impulse response functions.
    
    Parameters
    ----------
    fit : dict
        Output of varx_fit.
    n_ahead : int
        Number of periods for IRF.
    orthogonalize : bool
        Use Cholesky decomposition for orthogonalized shocks.
        
    Returns
    -------
    dict with 'irf' (n_ahead x k x k) - response of k variables to k shocks
    """
    k = fit["k"]
    lags_y = fit["lags_y"]
    coef = fit["coef"]
    sigma = fit["sigma"]
    
    # Extract A matrices from coefficients
    # coef is (k, n_features), first k*lags_y columns are Y lags
    A = np.zeros((lags_y, k, k))
    for lag in range(lags_y):
        A[lag] = coef[:, lag*k:(lag+1)*k].T
        
    # Compute IRF using the recursive method
    # Y_t = A_1 Y_{t-1} + ... + A_p Y_{t-p} + e_t
    # In companion form: the IRF at horizon h is given by the first block row of A^h
    
    # Build companion matrix
    # For VAR(p): companion matrix is (k*p) x (k*p)
    # [A_1 A_2 ... A_{p-1} A_p]
    # [I   0   ... 0       0  ]
    # [0   I   ... 0       0  ]
    # ...
    comp = np.zeros((k * lags_y, k * lags_y))
    comp[:k, :] = A.reshape(k, k * lags_y)
    for i in range(1, lags_y):
        comp[k*i:k*i+k, k*(i-1):k*(i-1)+k] = np.eye(k)
        
    # Compute powers of companion matrix
    irf = np.zeros((n_ahead, k, k))
    
    # Orthogonalize if requested
    if orthogonalize:
        try:
            P = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            P = np.eye(k)
    else:
        P = np.eye(k)
        
    # IRF at horizon 0 is identity
    irf[0] = P
    
    # For horizon h, take the first k rows of comp^h applied to P
    comp_power = np.eye(k * lags_y)
    for h in range(1, n_ahead):
        comp_power = comp_power @ comp
        # First k rows give the response of Y_t to shocks
        irf[h] = comp_power[:k, :k] @ P
        
    return {"irf": irf}


# ----------------------------------------------------------------- Forecast Error Variance Decomposition

def varx_fevd(fit, n_ahead=10):
    """
    Forecast Error Variance Decomposition.
    
    Parameters
    ----------
    fit : dict
        Output of varx_fit.
    n_ahead : int
        
    Returns
    -------
    dict with 'fevd' (n_ahead x k x k) - contribution of each shock to forecast variance
    """
    irf_result = varx_irf(fit, n_ahead=n_ahead, orthogonalize=True)
    irf = irf_result["irf"]
    
    k = fit["k"]
    fevd = np.zeros((n_ahead, k, k))
    
    for h in range(n_ahead):
        # Variance decomposition at horizon h
        # Sum of squared IRFs up to horizon h
        for i in range(k):
            for j in range(k):
                fevd[h, i, j] = np.sum(irf[:h+1, i, j]**2)
                
    # Normalize to percentages
    for h in range(n_ahead):
        for i in range(k):
            total = np.sum(fevd[h, i, :])
            if total > 0:
                fevd[h, i, :] /= total
                
    return {"fevd": fevd}


# ----------------------------------------------------------------- Information Criteria for lag selection

def varx_select_lag(Y, X=None, max_lags=10, constant=True, criterion="AIC"):
    """
    Select optimal lag order using information criteria.
    
    Parameters
    ----------
    Y : ndarray
        Endogenous variables.
    X : ndarray, optional
        Exogenous variables.
    max_lags : int
        Maximum number of lags to test.
    constant : bool
        Include constant.
    criterion : str
        'AIC' or 'BIC'.
        
    Returns
    -------
    dict with 'optimal_lag', 'criteria' (dict of lag -> criterion value)
    """
    Y = utils.as_matrix(Y)
    if X is not None:
        X = utils.as_matrix(X)
        
    n, k = Y.shape
    if X is not None:
        m = X.shape[1]
    else:
        m = 0
        
    criteria = {}
    
    for p in range(1, max_lags + 1):
        try:
            fit = varx_fit(Y, X, lags_y=p, lags_x=min(p, 1), constant=constant)
            n_obs = fit["nobs"]
            n_params = k * (k * p + m * min(p, 1) + (1 if constant else 0))
            
            if criterion.upper() == "AIC":
                ic = -2 * fit["loglik"] + 2 * n_params
            elif criterion.upper() == "BIC":
                ic = -2 * fit["loglik"] + np.log(n_obs) * n_params
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
                
            criteria[p] = ic
        except Exception:
            criteria[p] = np.inf
            
    optimal_lag = min(criteria, key=criteria.get)
    
    return {"optimal_lag": optimal_lag, "criteria": criteria}