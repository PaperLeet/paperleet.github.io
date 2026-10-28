# AWQ 激活感知缩放校准 — 题解

## 推导与实现

每个候选指数使用同一份原始权重，先乘通道scale、分组量化、再除回scale。用校准输入经过线性层的输出MSE选择候选，不能用权重自身MSE替代。

$$
a_j=\operatorname{mean}_i|X_{ij}|,\quad s_j=\frac{\max(a_j^\alpha,10^{-4})}{\sqrt{\max_j s^{raw}_j\min_j s^{raw}_j}},\quad \widehat W=Q(Ws)/s,\quad \alpha^*=\arg\min_{\alpha\in G}\operatorname{mean}(X\widehat W^T-XW^T)^2.
$$

## 边界与梯度归属

x:(N,D)，weight:(O,D)，D须被group_size整除。bits整数2..8，alphas非空且在[0,1]。组沿每行连续输入通道划分，scale=max((max-min)/(2^bits-1),1e-5)，zero=clip(round(-min/scale),0,2^bits-1)，ties-to-even。缩放raw先clamp1e-4，再除以sqrt(max*min)。按输入alpha顺序打破并列误差。返回(scale,反量化weight,alpha,float MSE)，离线校准不传梯度。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

16个确定性校准样本含8倍显著通道，3×4权重用3bit分组量化；扫描0到1的11个alpha并比较首个alpha=0基线和选中候选。

单线性层AWQ网格搜索与分组非对称量化，使用平均绝对激活统计；不包含权重clipping搜索或TinyChat内核。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| best_mse | 0.010094095 |
| baseline_mse | 0.10756917 |
| chosen_alpha | 0.2 |
| reported_error_difference | 0 |

## 常见错误

- m1_no_inverse_scale: 量化后未撤销通道缩放
- m2_peak_activation: 使用最大激活而不是平均绝对激活
- m3_weight_mse: 以权重MSE替代校准输出MSE

[AWQ](https://arxiv.org/abs/2306.00978) · Section 3.2: activation-aware scaling search; official auto_scale.py _search_module_scale.
