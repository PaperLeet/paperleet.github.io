# DiffMimic 示范回放与可微PD控制

## 1. 问题

DiffMimic 示范回放与可微PD控制：实现下列接口并通过数值、边界和实验检查。

实现论文状态误差触发的示范回放、PD力矩和时域状态匹配；实验是标量可微积分器，不是Brax人形机器人复现。

## 2. 核心公式

$$
\tilde s=\begin{cases}s&\|s-s^*\|^2<\epsilon\\s^*&\text{otherwise}\end{cases},\quad\tau=k_p(q^*-q)-k_d\dot q,\quad L=\frac1B\sum_{b,t}\|s_{bt}-s^*_{bt}\|^2.
$$

## 3. 公式到代码

- `replay_state(state: Tensor, reference: Tensor, threshold: float) -> Tensor`
- `pd_torque(position: Tensor, velocity: Tensor, target: Tensor, kp: float, kd: float) -> Tensor`
- `state_matching_loss(states: Tensor, reference: Tensor) -> Tensor`

回放在下一次策略/物理转移前替换偏离过大的状态，因此保留分支可传梯度、回放分支不将梯度传给旧状态。状态匹配loss应由真实轨迹误差计算，不能因为回放而将当前错误记录清零。PD目标角速度为零。

## 4. 本题约定

replay输入state/reference为(B,D)，threshold>=0；误差严格小于阈值才保留原状态，边界回放，reference固定detach。PD输入(B,J)且同形，kp/kd非负。loss输入(B,T,D)，先按时间及坐标求和再batch平均，目标detach。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

`experiment.py` 在固定输入上调用提交的全部接口，记录学习或采样曲线与末端指标。实验逻辑和阈值见 T3，输出同样用于可视化。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/diffmimic --impl problems/robotics/diffmimic/starter.py
paperleet mutate robotics/diffmimic
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [DiffMimic: Efficient Motion Mimicking with Differentiable Physics](https://arxiv.org/abs/2304.03274) — Equations 3.1, 3.3, 3.5 and Algorithm 1: PD torque, state matching and demonstration replay
- [Verified source](https://arxiv.org/html/2304.03274)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
