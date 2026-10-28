# TD-MPC2 单纯形潜变量与精英轨迹更新

## 1. 问题

TD-MPC2 单纯形潜变量与精英轨迹更新：实现下列接口并通过数值、边界和实验检查。

提取SimNorm与MPPI已选精英的加权矩更新；候选轨迹与价值作为输入，不学习动力学/Q网络或执行真实MPC。

## 2. 核心公式

$$
z_{g,:}=\operatorname{softmax}(x_{g,:}),\quad w_i=\operatorname{softmax}(\tau Q_i),\quad\mu=\sum_iw_i a_i,\quad\sigma=\operatorname{clip}(\sqrt{\sum_iw_i(a_i-\mu)^2},\sigma_{min},\sigma_{max}).
$$

## 3. 公式到代码

- `simnorm(values: Tensor, group_dim: int) -> Tensor`
- `elite_moments(actions: Tensor, values: Tensor, temperature: float, min_std: float, max_std: float) -> tuple`

SimNorm在每个小组内归一化，不能让整个latent共享一个softmax。轨迹权重沿精英E维，对整条时域共享。官方temperature乘价值；方差围绕加权均值，最后再夹标准差。公式用精确归一化，省略源码分母的1e-9数值保护。

## 4. 本题约定

simnorm输入任意rank>=1浮点张量，末维被group_dim整除且group_dim>=2。actions为(H,E,D)，values为(E,)，temperature>=0且为乘数；0<min_std<=max_std。返回(H,D)的mean/std。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

`experiment.py` 在固定输入上调用提交的全部接口，记录学习或采样曲线与末端指标。实验逻辑和阈值见 T3，输出同样用于可视化。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/td_mpc2 --impl problems/robotics/td_mpc2/starter.py
paperleet mutate robotics/td_mpc2
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828) — SimNorm and official _plan elite weighted mean/std update
- [Verified source](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py)
- [Verified source](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
