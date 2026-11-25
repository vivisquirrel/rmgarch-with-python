"""
examples/example_cgarch.py
Demo for the CGARCH (Asymmetric GARCH) model.
"""

import numpy as np
import matplotlib.pyplot as plt
from rmgarchpy.cgarch import cgarch_fit

def main():
    print("="*50)
    print("1. 加载模拟数据 (Loading Simulated Data)")
    print("="*50)
    np.random.seed(2026)
    n_obs = 1000
    
    # 模拟带有杠杆效应的收益率数据 (类似 EGARCH 过程)
    Y = np.zeros(n_obs)
    h = np.zeros(n_obs) # 条件方差
    
    # 真实参数
    omega = 0.05
    alpha = 0.1   #  ARCH 项
    beta = 0.85   # GARCH 项
    gamma = -0.2  # 杠杆效应项 (负值代表杠杆效应)
    
    h[0] = omega / (1 - alpha - beta)
    
    for t in range(1, n_obs):
        eps_prev = Y[t-1]
        # CGARCH/EGARCH 逻辑: 对数方差或包含非对称项的方差方程
        # 这里使用 GJR-GARCH 形式作为示例: h_t = omega + (alpha + gamma * I(eps<0)) * eps_{t-1}^2 + beta * h_{t-1}
        indicator = 1 if eps_prev < 0 else 0
        h[t] = omega + (alpha + gamma * indicator) * (eps_prev**2) + beta * h[t-1]
        Y[t] = np.sqrt(h[t]) * np.random.randn()
        
    print(f"数据形状: {Y.shape}")
    print(f"收益率标准差: {np.std(Y):.4f}")

    print("\n" + "="*50)
    print("2. 拟合 CGARCH 模型 (Fitting CGARCH)")
    print("="*50)
    # 拟合 CGARCH(1,1,1) 模型
    try:
        # 传入一维数组，指定阶数
        cgarch_result = cgarch_fit(Y, order=(1, 1, 1))
        
        print(f"模型估计成功！")
        print(f"Omega: {cgarch_result['omega']:.6f}")
        print(f"Alpha: {cgarch_result['alpha']:.4f}")
        print(f"Beta: {cgarch_result['beta']:.4f}")
        print(f"Gamma (杠杆效应): {cgarch_result['gamma']:.4f}")
        print(f"对数似然值: {cgarch_result['loglik']:.4f}")
        
        # 检查持久性 (Persistence)
        persistence = cgarch_result['alpha'] + cgarch_result['beta'] + cgarch_result['gamma']/2
        print(f"模型持久性 (Persistence): {persistence:.4f}")
        
        # 提取条件方差序列
        sigma_t = np.sqrt(cgarch_result['h'])
        
        print("\n" + "="*50)
        print("3. 绘图展示 (Plotting Results)")
        print("="*50)
        
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # 1. 收益率序列
        ax1 = axes[0]
        ax1.plot(Y, color='blue', alpha=0.6)
        ax1.set_title('Returns (with Leverage Effect)')
        
        # 2. 条件波动率
        ax2 = axes[1]
        ax2.plot(sigma_t, color='red', label='Conditional Volatility (CGARCH)')
        ax2.set_title('Conditional Volatility')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('example_cgarch_plot.png')
        print("图表已保存为 example_cgarch_plot.png")
        
    except Exception as e:
        print(f"CGARCH 拟合过程中出现错误 (可能是由于模块尚未完全实现): {e}")
        print("跳过绘图步骤。")

if __name__ == "__main__":
    main()