# Diffusion-DPO：成对去噪偏好优化

实现 [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908) 的偏好训练核心。特征误差取均值、显式使用 `beta/2`，与[作者发布的训练代码](https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py)一致；小型实验会实际训练一个去噪器。

## 1. 偏好对共享前向噪声

每行包含偏好潜变量 $x^w$、非偏好潜变量 $x^l$、共享噪声 $\epsilon$ 和累计调度值 $\bar\alpha$。同一对必须使用相同时间步和噪声：

$$
x_t^w=\sqrt{\bar\alpha}\,x^w+\sqrt{1-\bar\alpha}\,\epsilon,
\qquad
x_t^l=\sqrt{\bar\alpha}\,x^l+\sqrt{1-\bar\alpha}\,\epsilon.
$$

实现 `paired_noising(winner, loser, noise, alpha_bar)`。前三个张量具有相同的非空浮点形状 `(B, ...)`，至少有一个特征维；`alpha_bar` 为 `(B,)`，取值在 `[0, 1]`。将调度值广播至所有特征维，返回 `(noisy_winner, noisy_loser)`。这些输出属于固定训练数据，函数内部对四个输入都执行 detach。端点零返回纯噪声，端点一返回干净潜变量。

## 2. 相对参考策略的偏好目标

给定当前模型的噪声预测 $\epsilon_\theta^w,\epsilon_\theta^l$ 和参考模型的预测 $\epsilon_{\rm ref}^w,\epsilon_{\rm ref}^l$，先按特征取均方误差：

$$
e_\theta^w=\frac1D\sum_{d=1}^D(\epsilon_{\theta,d}^w-\epsilon_d)^2,
\qquad
\Delta_\theta=e_\theta^w-e_\theta^l,
\qquad
\Delta_{\rm ref}=e_{\rm ref}^w-e_{\rm ref}^l.
$$

偏好间隔和损失定义为

$$
m=-\frac\beta2(\Delta_\theta-\Delta_{\rm ref}),
\qquad
L=\frac1B\sum_{i=1}^B\operatorname{softplus}(-m_i).
$$

实现 `diffusion_dpo_loss(model_w, model_l, ref_w, ref_l, noise, beta)`，返回标量张量。五个张量具有相同形状、dtype 和 device。误差均值覆盖所有非 batch 维。梯度只流入 `model_w` 与 `model_l`；即使调用者给参考预测和噪声开启梯度，也应在函数内冻结它们。用稳定的 softplus 或 log-sigmoid 计算损失。

`beta` 是有限正数，是吸收了论文中时间步和权重常数的有效尺度。本题保留发布代码中的 `1/2` 与特征均值约定，避免潜变量维度改变损失尺度。

## 3. 接口约定与示例

保持浮点 dtype/device，不修改输入。形状、dtype、device 不匹配，空张量、非浮点张量、非有限张量、非法 alpha 或 beta 都应抛出 `ValueError`。测试覆盖 CPU float32 与 float64。

当前预测与参考预测相同时，每个间隔都是零，损失为 $\log 2$。对一个标量偏好对，若噪声为零、当前预测为 `(0, 2)`、参考预测为 `(1, 1)`、`beta=2`，则间隔为 $4$，损失为 $\log(1+e^{-4})$。

## 4. 验证层次

T0 检查接口与输入不变性。T1 对照独立标量计算、调度端点与极端间隔。T2 检查解析梯度、冻结边界，以及复制特征或 batch 后损失不变。T3 用你的函数，在四个固定偏好对上训练两参数仿射去噪器 80 步；要求损失下降且各对相对参考模型的偏好间隔为正。

## 5. 实验范围

实验使用合成标量潜变量，并真实通过你的函数做梯度下降，无需图像模型和偏好数据集。偏好间隔增大不代表两侧绝对去噪误差都下降；目标优化的是相对参考模型的误差差值，因此图表展示 logistic 损失和各对间隔。

## 6. 怎么运行

```bash
paperleet init diffusion_rl/diffusion_dpo
paperleet test diffusion_rl/diffusion_dpo
```

## 7. 来源

- Wallace 等，[论文第 4 节](https://arxiv.org/html/2311.12908v1)。
- [官方 `train.py`](https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py)：成对共享噪声、时间步以及 DPO 损失。

本题代码是依据上述公式独立编写的精简教学实现，没有直接引入原项目训练脚本。
