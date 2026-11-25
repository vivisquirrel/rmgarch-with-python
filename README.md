# rmgarchpy — R语言中 rmgarch 的 Python 实现

`rmgarchpy` 是 R 语言量化金融库 **rmgarch**（Alexios Galanos, github.com/alexiosg/rmgarch）
的 Python 实现，面向量化投资研究中的 **多变量 GARCH（Multivariate GARCH）** 建模需求。
它不依赖任何 R 运行时，仅使用 numpy / scipy / pandas / arch / statsmodels等常用Python库，直接 `import rmgarchpy` 集成。

## 能力总览（R -> Python 对照）

| R (rmgarch) | Python (rmgarchpy) | 说明 |
|---|---|---|
| `rugarch::ugarchfit`（依赖部分） | `fit_ugarch()` | 一元 GARCH(p,q) MLE 引擎（normal / 标准化 Student-t 新息），为多变量模型提供第一阶段 |
| `dccspec` / `dccfit` | `dcc_fit()`（`DCCSpec`） | 两阶段 DCC-GARCH；高斯似然与半参数（kernel）似然；OPG 标准误 |
| `dccfilter` | `dcc_filter()` | 固定参数对新一段样本重算动态相关路径 |
| `dccsim` | `dcc_sim()` | 从拟合模型模拟多元 DCC-GARCH 路径 |
| `dccforecast` | `DCCFit.forecast()` | 一步向前 Q / Rho / H 预测 |
| `cgarchspec` / `cgarchfit` | `cgarch_fit()` | 动态 copula GARCH（mvnormal / mvt；analytic / empirical 伪观测变换；动态或静态） |
| `mdcopula.*`（历史版本） | `MDcopula` 类 + `ktau` / `tailindex` / `copula_fit` / `copula_mvfit` / `copula_gof` / `copula_sample` | 依赖结构诊断：Kendall tau 矩阵、上/下/中尾相关指数、逐对 copula 参数估计（clayton/gumbel/frank/normal/t，含 90/180/270 旋转）、Cramer-von-Mises 拟合优度 + 参数化 bootstrap |
| `gogarchfit`（基础配置） | `gogarch_fit()` | GO-GARCH：FastICA 正交化 + 独立分量上一元 GARCH，重构时变协方差 |
| `fastica` / `radical` | `fastica()` / `radical()` | 独立成分分析（Hyvarinen 定点迭代 + Radical 成对角度优化） |
| `varxfit` | `varx_fit()` | 带外生变量的 VAR（OLS） |
| `DCCtest` / `MardiaTest` | `dcc_test()` / `mardia_test()` | Engle-Sheppard 动态相关 LM 检验；Mardia 多元偏度/峰度正态性检验 |
| `rshape` / `rskew` | `rshape()` / `rskew()` | 偏斜-正态族形状/偏度互换 |

## 安装

```bash
git clone <your-repo-url> && cd rmgarchpy
pip install -r requirements.txt
pip install .
```

或直接以源码目录形式放入项目（包名 `rmgarchpy`，无编译依赖）。

## 快速上手

```python
import numpy as np
import rmgarchpy as rmg

# returns: (T, m) 收益率矩阵, 例如 pd.DataFrame(...).values
fit = rmg.dcc_fit(returns, garch_order=(1, 1), uni_dist='std',
                  likelihood='gaussian')

print(fit.summary())          # 参数 + OPG 标准误
rho = fit.Rho                 # (T, m, m) 动态相关矩阵
Q = fit.Q                     # (T, m, m) 动态准协方差
H = fit.H                     # (T, m, m) 协方差 D rho D
fit.plot(pair=(0, 1))         # 相关路径图（需 matplotlib）

fc = fit.forecast()           # 下一步 Q/Rho/H
sim = rmg.dcc_sim(fit, n_sim=1000)   # 模拟路径
flt = rmg.dcc_filter(fit, new_returns)  # 新样本上重估动态相关

# 依赖结构诊断（mdcopula 风格）
md = rmg.MDcopula(fit.resmat)          # 用标准化残差
print(md.ktau())                       # Kendall tau 矩阵
print(md.tailindex('lower')['index'])  # 下尾相关指数
print(md.param(family='clayton', method='mle')['theta'])

# 动态 copula
cg = rmg.cgarch_fit(returns, dist='mvt', nu=10.0, dynamic=True)

# 检验
from rmgarchpy import dcc_test, mardia_test
print(dcc_test(cg.stdresid, n_lags=1))
print(mardia_test(returns, alpha=0.05))
```

详细教程见 `使用文档.md`；可直接运行的示例见 `examples/`。

## 数值验证（本仓库 tests/test_smoke.py 全部通过）

- 从已知参数的 DCC(1,1)-GARCH 数据生成 1500x3 模拟样本，`dcc_fit` 收敛并良好恢复：
  平均相关偏差 RMSE 约 0.02，A/B 矩阵元素偏差小于 0.05。
- copula 逐对 MLE：clayton / gumbel / frank / normal / t（含 90/180/270 旋转）
  在 3000 样本模拟数据上参数偏差 < 0.1。
- FastICA 在人为混合信号上恢复 |相关系数| > 0.99；Radical 收敛到同等精度。
- GO-GARCH 重构的 H_t 全部对称正定；A 矩阵严格正交。

## 与原 R 实现的差异

1. **一元第一阶段**用本库自带的 `fit_ugarch`（常数均值 GARCH(p,q)，normal/std 新息）
   替代 rugarch；ARMA 均值、fixed 参数、realGARCH 等未纳入。
2. **DCC 参数化**采用经典 fitDCC 形式
   `q_t = omega + A e_{t-1}e_{t-1}' A' + B q_{t-1} B'`
   （A 下三角含对角，B 严格下三角），等价于 rmgarch 的 masked-DCC；新版 rmgarch 的
   标量-lag 高阶（maxdccOrder>1）未移植。
3. **半参数似然**按 Galluccio 等人的思想实现为对白化残差的留一多元核密度估计，
   与 R 的具体加权形式在细节上可能有出入（高斯似然则与 R 完全一致）。
4. **cGARCH** 覆盖 mvnormal / mvt（动态与静态）；Archimedean 族走 mdcopula 逐对通道。
5. **GO-GARCH** 为固定旋转矩阵（IC A 估计）的基础版本，未包含 R 版完整的
   时变旋转 P-sigma 递归与 FFT 矩函数。
6. 标准误统一采用 **OPG**（外积梯度）数值实现；未移植 solver=solnp/gosolnp 选项。
7. 历史版本中的 `fitvDCC / fitGGX1 / fitGGX2`（vDCC、GGX 模型）在当前 GitHub master
   已被 dccfit / cgarchfit 取代，故本移植对齐 **master 现行能力面**。

## Reference

- 原算法与接口设计：Alexios Galanos, rmgarch, GPL-3
- DCC：Engle R. (2002) "Dynamic Conditional Correlation", Journal of Business & Economic Statistics.
- 半参数 DCC：Galluccio, Maffezzoli, Onorati (2012), Journal of Financial Econometrics.
- 动态 copula：Patton (2006); rmgarch cGARCH 文档。
