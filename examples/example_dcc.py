"""
examples/example_dcc.py
Demo for the DCC-GARCH model.
"""

import numpy as np
import matplotlib.pyplot as plt
from rmgarchpy.dcc import dcc_fit, dcc_forecast

def main():
    print("="*50)
    print("1. 加载模拟数据 (Loading Simulated Data)")
    print("="*50)
    np.random.seed(2026)
    n_obs = 1000
    k = 3  # 3个资产
    
    # 模拟带有时间变动相关性的收益率数据
    # 使用简单的 DCC(1,1) 过程生成数据
    Y = np.zeros((n_obs, k))
    Q = np.eye(k) * 0.01 # 初始条件协方差矩阵
    
    # DCC 参数
    omega = np.eye(k) * 0.001
    alpha = 0.1
    beta = 0.85
    
    A = np.array([[0.5, 0.2, 0.1],
                  [0.2, 0.6, 0.3],
                  [0.1, 0.3, 0.4]]) # 滞后系数
    
    for t in range(1, n_obs):
        # 更新 Q 矩阵 (简化版 DCC 过程)
        eps_prev = Y[t-1]
        Q = omega + alpha * np.outer(eps_prev, eps_prev) + beta * Q
        
        # 转换为相关矩阵 R
        D = np.diag(np.sqrt(np.diag(Q)))
        D_inv = np.linalg.inv(D)
        R = D_inv @ Q @ D_inv
        
        # 生成相关误差
        L = np.linalg.cholesky(R)
        z = np.random.randn(k)
        eps = L @ z
        
        # 生成收益率 (假设无条件方差为1)
        Y[t] = A @ Y[t-1] + eps
        
    print(f"数据形状: {Y.shape}")

    print("\n" + "="*50)
    print("2. 拟合 DCC-GARCH 模型 (Fitting DCC-GARCH)")
    print("="*50)
    # 假设先对每个资产拟合单变量 GARCH(1,1) 获取标准化残差
    # 这里为了演示，直接传入收益率，实际应用中应先做 univariate GARCH 过滤
    # 假设我们已经有了标准化残差 z_scores
    z_scores = Y[1:] # 仅作演示
    
    # 拟合 DCC(1,1) 模型
    # 注意：实际 rmgarchpy 的 dcc_fit 可能需要更复杂的参数配置
    try:
        dcc_result = dcc_fit(z_scores, order=(1, 1))
        
        print(f"模型估计成功！")
        print(f"Omega 矩阵:\n{dcc_result['omega']}")
        print(f"Alpha 参数: {dcc_result['alpha']:.4f}")
        print(f"Beta 参数: {dcc_result['beta']:.4f}")
        print(f"对数似然值: {dcc_result['loglik']:.4f}")
        
        # 提取动态相关矩阵序列
        R_t = dcc_result['R'] # shape: (n_obs-1, k, k)
        print(f"动态相关矩阵序列形状: {R_t.shape}")
        
        # 提取资产1和资产2之间的动态相关性
        corr_12 = R_t[:, 0, 1]
        
        print("\n" + "="*50)
        print("3. 绘图展示 (Plotting Results)")
        print("="*50)
        
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # 1. 收益率序列
        ax1 = axes[0]
        ax1.plot(Y[:, 0], label='Asset 1', alpha=0.5)
        ax1.plot(Y[:, 1], label='Asset 2', alpha=0.5)
        ax1.set_title('Asset Returns')
        ax1.legend()
        
        # 2. 动态相关性
        ax2 = axes[1]
        ax2.plot(corr_12, color='purple', label='Correlation (Asset 1 vs Asset 2)')
        ax2.axhline(np.mean(corr_12), color='red', linestyle='--', label='Mean Correlation')
        ax2.set_title('Dynamic Conditional Correlation (DCC)')
        ax2.set_ylim(-1, 1)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('example_dcc_plot.png')
        print("图表已保存为 example_dcc_plot.png")
        
    except Exception as e:
        print(f"DCC 拟合过程中出现错误 (可能是由于模块尚未完全实现): {e}")
        print("跳过绘图步骤。")

if __name__ == "__main__":
    main()