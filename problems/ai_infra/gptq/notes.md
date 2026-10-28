# GPTQ Cholesky 误差补偿量化 — 题解

## 推导与实现

拷贝工作权重，每列先保存网格化结果，再用归一化残差更新剩余权重。必须除以Cholesky对角而不是Hessian对角；补偿符号是减法。

$$
H^{-1}=U^TU,\qquad q_j=\Delta\,\operatorname{clip}(\operatorname{round}(w_j/\Delta),q_{min},q_{max}),\qquad e_j=(w_j-q_j)/U_{jj},\qquad W_{:,j:}\leftarrow W_{:,j:}-e_jU_{j,j:}.
$$

## 边界与梯度归属

weight:(O,D)，upper:(D,D)，scales:(O,)；upper严格上三角以下为零、对角>0，表示H^{-1}=U^T U。round为ties-to-even，整数界闭区间。从左到右量化，补偿包含当前列与所有后列。返回反量化float权重，不修改输入；离线PTQ不传梯度。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

使用两维相关校准输入和两行权重，扫描固定网格间距；比较GPTQ与独立逐元素round-to-nearest的输出重建MSE。

给定上三角Cholesky因子的单块GPTQ核心；量化网格固定，Hessian估计、阻尼与分解由题外提供。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| compensated_mse | 0.09065 |
| rtn_mse | 0.28265 |

## 常见错误

- m1_no_compensation: 量化后没有补偿后续列
- m2_wrong_error_sign: 误差补偿使用相反符号

[GPTQ](https://arxiv.org/abs/2210.17323) · Section 4, Algorithm 1; official gptq.py fasterquant inner column loop.
