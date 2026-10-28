# Diffusion Policy：动作去噪与滚动执行

## 1. 练习目标

实现 [Diffusion Policy](https://arxiv.org/abs/2303.04137) 的动作序列加噪、噪声预测损失和滚动执行切片。观测编码器与条件预测网络由调用方提供。

## 2. 接口与返回值

| 函数 | 输入 | 返回 |
|---|---|---|
| `add_action_noise(actions, noise, alpha_bar)` | 同形 `(B,T,D)` 归一化动作和噪声，`(B,)` 累积 alpha | 同形、同 dtype/device 的加噪动作 |
| `action_denoising_loss(prediction, noise)` | 同形 `(B,T,D)` 预测噪声和目标噪声 | 同 dtype/device 的零维标量 Tensor |
| `action_chunk(actions, observation_horizon, execution_horizon)` | `(B,T,D)` 动作预测，整数 `To`、`Ta` | `(B,Ta,D)` 执行片段 |

输入均为有限的 CPU float32/float64，三个维度非空，不允许修改输入。已提供的 `_sequence`、`_matching` 只做检查，成功返回 `None`，失败抛出 `ValueError`。

### 返回值要求

加噪函数保留动作、噪声和 alpha 的梯度。损失对所有样本、时间与动作坐标求均值，并在函数内部冻结目标噪声；预测值保持可微。切片保持所选动作的梯度，可以返回视图。

## 3. 核心公式

`alpha_bar` 是 DDPM 中 alpha 的累积乘积，不是单步系数。每个样本独立使用自己的值：

$$
A^k=\sqrt{\bar\alpha_k}A^0+\sqrt{1-\bar\alpha_k}\epsilon,\qquad L=\frac{1}{BTD}\sum_{b,t,d}(\hat\epsilon_{btd}-\operatorname{stopgrad}(\epsilon_{btd}))^2.
$$

要求 `0 < alpha_bar[b] < 1`。本练习仅给动作加噪；观测作为预测网络的条件输入。

采用官方代码的 `oa_step_convention=True` 对齐方式：当前时刻动作位于预测窗口的 `To-1`，执行区间为 `[To-1, To-1+Ta)`。`To`、`Ta` 必须是正整数，且 `To-1+Ta <= T`。执行完这一小段后重新规划。

## 4. 小例子

动作 `2`、噪声 `-1`、累积 alpha `.64` 得到 `.8*2+.6*(-1)=1`。预测 `[10,20,30,40,50]`，`To=2`、`Ta=3` 时，执行 `[20,30,40]`。

## 5. 确定性实验

给定目标位置与平滑时间轮廓，生成二维示范动作。只有两个参数的条件预测器从加噪动作和目标条件中学习噪声；你的函数完成加噪、损失训练和执行切片。图表展示损失、去噪动作以及移除目标条件后的对照。

这是条件动作扩散的教学改编，聚焦训练与时间对齐；没有训练视觉编码器、执行完整反向采样或复现机器人基准。

## 6. 运行实现

补全 `starter.py` 的三个函数后，通过 PaperLeet 运行。T0 检查接口，T1 使用独立数值例子，T2 检查梯度与不变性，T3 运行条件动作去噪实验。

## 7. 来源

- [论文和官方项目](https://diffusion-policy.cs.columbia.edu/)：RSS 2023，Section 2.3。
- [官方低维策略实现](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py)：`compute_loss`、`predict_action` 与 `oa_step_convention`。
