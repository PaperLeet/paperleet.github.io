# SmoothQuant 激活与权重等价平滑 — 题解

## 推导与实现

在样本轴提取激活最大值，在输出轴提取权重最大值，二者得到同一个输入通道向量。激活除以scale，权重乘以scale。校准统计无需梯度，等价矩阵乘法仍保留训练梯度。

$$
s_j=\max\left(\frac{(\max_i|X_{ij}|)^\alpha}{\max(\max_o|W_{jo}|,\epsilon)^{1-\alpha}},\epsilon\right),\quad \widehat X=X/s,\quad \widehat W=sW,\quad \widehat X\widehat W=XW.
$$

## 边界与梯度归属

x:(N,D)，weight:(D,O)，alpha在[0,1]；eps>0。权重通道绝对最大值先clamp到eps，最终scale再clamp到eps；零激活通道按该公式处理。scale脱离校准图，smooth_transform仍对x、weight、scale可微。输入不修改。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

用 17 个样本、一个幅度100的离群通道及反向缩小的权重，扫描alpha；对比直接INT8与平滑后INT8的重建误差。

校准最大值、离线scale与可微等价变换；实验用对称INT8模拟量化，未实现整数GEMM或LayerNorm融合。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| unsmoothed_mse | 0.052506032 |
| balanced_mse | 2.6201785e-05 |
| equivalence_error | 3.3306691e-16 |

## 常见错误

- m1_weight_wrong_direction: 权重也除以scale破坏等价变换
- m2_mean_instead_of_max: 校准激活使用均值而非最大值

[SmoothQuant](https://arxiv.org/abs/2211.10438) · Section 4, Equations (3) and (4); official smooth.py offline scale construction.
