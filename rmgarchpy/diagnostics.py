"""
rmgarchpy/diagnostics.py
Diagnostic tools for time series models (VARX, GARCH, DCC, etc.).

This module provides functions to test the residuals of fitted models 
for normality, autocorrelation, heteroskedasticity (ARCH effects), 
and conditional moment violations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro, acorr_ljungbox
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.tsa.stattools import acf, pacf
import warnings

def test_normality(residuals, method="jarque_bera"):
    """
    检验残差的正态性。
    
    参数:
    residuals (array-like): 残差序列，可以是 1D 或 2D (n_obs, k)。
    method (str): 检验方法，可选 "jarque_bera", "shapiro", "normaltest" (D'Agostino-Pearson)。
    
    返回:
    dict: 包含统计量 (statistic) 和 p值 (pvalue)。如果是多维数据，返回每个维度的检验结果。
    """
    residuals = np.asarray(residuals)
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)
        
    n_vars = residuals.shape[1]
    results = {}
    
    for i in range(n_vars):
        res = residuals[:, i]
        if method == "jarque_bera":
            stat, pval, skew, kurtosis = jarque_bera(res)
            results[f"var_{i+1}"] = {
                "method": "Jarque-Bera",
                "statistic": stat,
                "pvalue": pval,
                "skewness": skew,
                "kurtosis": kurtosis
            }
        elif method == "shapiro":
            # Shapiro-Wilk 检验对大样本 (n > 5000) 可能会报错或性能较差
            if len(res) > 5000:
                warnings.warn("Shapiro-Wilk test is not recommended for n > 5000. Using D'Agostino-Pearson instead.")
                stat, pval = normaltest(res)
                results[f"var_{i+1}"] = {"method": "D'Agostino-Pearson", "statistic": stat, "pvalue": pval}
            else:
                stat, pval = shapiro(res)
                results[f"var_{i+1}"] = {"method": "Shapiro-Wilk", "statistic": stat, "pvalue": pval}
        elif method == "normaltest":
            stat, pval = normaltest(res)
            results[f"var_{i+1}"] = {"method": "D'Agostino-Pearson", "statistic": stat, "pvalue": pval}
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'jarque_bera', 'shapiro', 'normaltest'.")
            
    return results


def test_autocorrelation(residuals, lags=10, method="ljung_box"):
    """
    检验残差的自相关性 (Ljung-Box 检验)。
    
    参数:
    residuals (array-like): 残差序列。
    lags (int or list): 检验的滞后阶数。
    method (str): 目前仅支持 "ljung_box"。
    
    返回:
    dict: 包含各滞后阶数的统计量和 p值。
    """
    residuals = np.asarray(residuals)
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)
        
    n_vars = residuals.shape[1]
    results = {}
    
    for i in range(n_vars):
        res = residuals[:, i]
        if method == "ljung_box":
            lb_result = acorr_ljungbox(res, lags=lags, return_df=True)
            results[f"var_{i+1}"] = {
                "lags": lb_result.index,
                "statistic": lb_result["lb_stat"].values,
                "pvalue": lb_result["lb_pvalue"].values
            }
        else:
            raise ValueError(f"Unknown method: {method}.")
            
    return results


def test_arch_effects(residuals, lags=5):
    """
    检验残差中是否存在 ARCH 效应 (条件异方差性)。
    通过对残差平方序列进行 Ljung-Box 检验来实现。
    
    参数:
    residuals (array-like): 残差序列。
    lags (int): 检验的滞后阶数。
    
    返回:
    dict: 包含残差平方序列的自相关检验结果。
    """
    residuals = np.asarray(residuals)
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)
        
    n_vars = residuals.shape[1]
    results = {}
    
    for i in range(n_vars):
        # 对残差平方进行检验
        res_sq = residuals[:, i] ** 2
        lb_result = acorr_ljungbox(res_sq, lags=lags, return_df=True)
        results[f"var_{i+1}"] = {
            "test": "ARCH LM Test (via Ljung-Box on squared residuals)",
            "lags": lb_result.index,
            "statistic": lb_result["lb_stat"].values,
            "pvalue": lb_result["lb_pvalue"].values
        }
        
    return results


def test_heteroskedasticity(residuals, exog=None, method="breusch_pagan"):
    """
    检验残差的异方差性 (Breusch-Pagan 或 White 检验)。
    通常用于线性回归或 VAR 模型的残差。
    
    参数:
    residuals (array-like): 残差序列。
    exog (array-like, optional): 外生变量/解释变量。如果为 None，则使用常数项。
    method (str): "breusch_pagan" 或 "white"。
    
    返回:
    dict: 包含 LM 统计量、p值等。
    """
    residuals = np.asarray(residuals)
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)
        
    n_obs, n_vars = residuals.shape
    
    if exog is None:
        exog = np.ones((n_obs, 1)) # 仅包含截距项
    else:
        exog = np.asarray(exog)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
            
    results = {}
    
    for i in range(n_vars):
        res = residuals[:, i]
        if method == "breusch_pagan":
            # het_breuschpagan 返回: LM stat, LM p-value, F-stat, F p-value
            lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(res, exog)
            results[f"var_{i+1}"] = {
                "method": "Breusch-Pagan",
                "lm_statistic": lm_stat,
                "lm_pvalue": lm_pval,
                "f_statistic": f_stat,
                "f_pvalue": f_pval
            }
        elif method == "white":
            # het_white 返回: LM stat, LM p-value, F-stat, F p-value
            lm_stat, lm_pval, f_stat, f_pval = het_white(res, exog)
            results[f"var_{i+1}"] = {
                "method": "White",
                "lm_statistic": lm_stat,
                "lm_pvalue": lm_pval,
                "f_statistic": f_stat,
                "f_pvalue": f_pval
            }
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'breusch_pagan' or 'white'.")
            
    return results


def check_conditional_moments(standardized_residuals):
    """
    检查标准化残差 (z_t = e_t / sigma_t) 的条件矩。
    理想的标准化残差应服从 N(0, 1)，即均值为0，方差为1。
    
    参数:
    standardized_residuals (array-like): 标准化后的残差。
    
    返回:
    dict: 包含样本均值、样本方差以及针对均值和方差的 t 检验结果。
    """
    z = np.asarray(standardized_residuals)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
        
    n_obs, n_vars = z.shape
    results = {}
    
    for i in range(n_vars):
        zi = z[:, i]
        mean_z = np.mean(zi)
        var_z = np.var(zi, ddof=1)
        
        # t-test for mean = 0
        t_stat_mean, pval_mean = stats.ttest_1samp(zi, 0)
        
        # Chi-square test for variance = 1
        # (n-1)*s^2 ~ chi^2(n-1)
        chi2_stat = (n_obs - 1) * var_z
        pval_var = 1 - stats.chi2.cdf(chi2_stat, df=n_obs - 1)
        
        results[f"var_{i+1}"] = {
            "sample_mean": mean_z,
            "sample_variance": var_z,
            "mean_test": {"t_stat": t_stat_mean, "pvalue": pval_mean},
            "variance_test": {"chi2_stat": chi2_stat, "pvalue": pval_var}
        }
        
    return results


def plot_residual_diagnostics(residuals, lags=10, save_path=None):
    """
    绘制残差诊断图 (4合1)。
    包含：残差时序图、自相关图(ACF)、残差平方ACF、Q-Q图。
    
    参数:
    residuals (array-like): 残差序列 (建议使用第一个变量的残差或标准化残差)。
    lags (int): ACF 图的滞后阶数。
    save_path (str, optional): 图像保存路径。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Please install it.")
        
    residuals = np.asarray(residuals)
    if residuals.ndim == 2:
        residuals = residuals[:, 0] # 默认绘制第一个变量
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Residual Diagnostics', fontsize=16)
    
    # 1. 残差时序图
    ax1 = axes[0, 0]
    ax1.plot(residuals, color='blue', alpha=0.7)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_title('Residuals over Time')
    ax1.set_xlabel('Time')
    
    # 2. 残差 ACF
    ax2 = axes[0, 1]
    acf_vals = acf(residuals, nlags=lags)
    conf_interval = 1.96 / np.sqrt(len(residuals))
    ax2.bar(range(lags + 1), acf_vals, color='gray', alpha=0.7)
    ax2.axhline(conf_interval, color='red', linestyle='--', linewidth=1)
    ax2.axhline(-conf_interval, color='red', linestyle='--', linewidth=1)
    ax2.set_title('Autocorrelation of Residuals')
    ax2.set_xlabel('Lag')
    
    # 3. 残差平方 ACF (ARCH 效应)
    ax3 = axes[1, 0]
    acf_sq_vals = acf(residuals**2, nlags=lags)
    ax3.bar(range(lags + 1), acf_sq_vals, color='orange', alpha=0.7)
    ax3.axhline(conf_interval, color='red', linestyle='--', linewidth=1)
    ax3.axhline(-conf_interval, color='red', linestyle='--', linewidth=1)
    ax3.set_title('Autocorrelation of Squared Residuals (ARCH Effect)')
    ax3.set_xlabel('Lag')
    
    # 4. Q-Q 图
    ax4 = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Normal Q-Q Plot')
    ax4.get_lines()[0].set_markerfacecolor('blue')
    ax4.get_lines()[0].set_markersize(4)
    ax4.get_lines()[1].set_color('red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Diagnostic plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close(fig)