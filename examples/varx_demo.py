"""
examples/varx_demo.py
Demo for the VARX module.
"""

import numpy as np
import matplotlib.pyplot as plt
from rmgarchpy.varx import (
    varx_fit, 
    varx_forecast, 
    varx_irf, 
    varx_fevd, 
    varx_select_lag
)

def main():
    print("="*50)
    print("1. 生成模拟数据 (Simulating Data)")
    print("="*50)
    np.random.seed(42)
    n_obs = 500
    k = 2  # 两个内生变量
    m = 1  # 一个外生变量

    # 真实数据生成过程 (DGP)
    # Y_t = A_1 Y_{t-1} + B_1 X_t + e_t
    A1 = np.array([[0.5, 0.2], 
                   [0.3, 0.4]])
    B1 = np.array([[0.1], 
                   [0.2]])
    
    Y = np.zeros((n_obs, k))
    X = np.random.normal(0, 1, (n_obs, m)) # 外生变量
    
    for t in range(1, n_obs):
        e_t = np.random.multivariate_normal([0, 0], )
        Y[t] = A1 @ Y[t-1] + B1.flatten() * X[t] + e_t

    print(f"数据形状: Y={Y.shape}, X={X.shape}")

    print("\n" + "="*50)
    print("2. 滞后阶数选择 (Lag Selection)")
    print("="*50)
    # 使用 AIC 准则选择最优滞后阶数
    lag_selection = varx_select_lag(Y, X, max_lags=5, criterion="AIC")
    print(f"最优滞后阶数: {lag_selection['optimal_lag']}")
    print(f"各阶数 AIC 值: {lag_selection['criteria']}")

    print("\n" + "="*50)
    print("3. 模型拟合 (Model Fitting)")
    print("="*50)
    # 假设我们选择滞后阶数为 1
    lags_y = 1
    lags_x = 1
    fit_result = varx_fit(Y, X, lags_y=lags_y, lags_x=lags_x, constant=True)
    
    print(f"使用的观测值数量: {fit_result['nobs']}")
    print(f"对数似然值: {fit_result['loglik']:.4f}")
    print(f"估计的系数矩阵 (Transposed):\n{fit_result['coef']}")
    print(f"残差协方差矩阵:\n{fit_result['sigma']}")

    print("\n" + "="*50)
    print("4. 脉冲响应函数 (Impulse Response Function)")
    print("="*50)
    irf_result = varx_irf(fit_result, n_ahead=10, orthogonalize=True)
    irf = irf_result['irf']
    print(f"IRF 矩阵形状 (Horizon x Response x Shock): {irf.shape}")
    print("第 5 期的正交化脉冲响应矩阵:\n", irf[5])

    print("\n" + "="*50)
    print("5. 预测 (Forecasting)")
    print("="*50)
    # 预测未来 5 期，假设未来外生变量为 0
    n_ahead = 5
    forecast_result = varx_forecast(fit_result, Y, X, n_ahead=n_ahead, X_future=np.zeros((n_ahead, m)))
    print(f"未来 {n_ahead} 期的预测值:\n{forecast_result['forecast']}")

    # 绘图展示 (可选)
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # 1. 实际值与拟合值
        ax1 = axes[0, 0]
        ax1.plot(Y[1:, 0], label='Actual Y1', color='blue', alpha=0.6)
        ax1.plot(fit_result['fitted'][:, 0], label='Fitted Y1', color='red', linestyle='--')
        ax1.set_title('Actual vs Fitted (Y1)')
        ax1.legend()

        # 2. 残差
        ax2 = axes[0, 1]
        ax2.plot(fit_result['residuals'][:, 0], color='gray')
        ax2.set_title('Residuals (Y1)')

        # 3. IRF
        ax3 = axes[1, 0]
        ax3.plot(irf[:, 0, 0], label='Y1 response to Y1 shock')
        ax3.plot(irf[:, 1, 0], label='Y2 response to Y1 shock')
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.set_title('Orthogonalized IRF (Shock to Y1)')
        ax3.legend()

        # 4. Forecast
        ax4 = axes[1, 1]
        ax4.plot(Y[-50:, 0], label='Historical Y1')
        ax4.plot(np.arange(n_obs, n_obs + n_ahead), forecast_result['forecast'][:, 0], 'ro-', label='Forecast Y1')
        ax4.set_title('Forecast for Y1')
        ax4.legend()

        plt.tight_layout()
        plt.savefig('varx_demo_plot.png')
        print("\n图表已保存为 varx_demo_plot.png")
    except Exception as e:
        print(f"绘图失败 (可能未安装 matplotlib): {e}")

if __name__ == "__main__":
    main()