# π0 反向时间流与动作积分

## 1. 问题

π0 反向时间流与动作积分：实现下列接口并通过数值、边界和实验检查。

提取官方openpi的线性流路径和Euler采样方向；显式给定噪声与时间，用线性速度回归替代VLA动作专家。

## 2. 核心公式

$$
x_t=t\epsilon+(1-t)a,\quad u=\epsilon-a,\quad x_{t+\Delta t}=x_t+\Delta t\,v_t,\quad \Delta t\le0.
$$

## 3. 公式到代码

- `flow_interpolate(actions: Tensor, noise: Tensor, time: Tensor) -> tuple`
- `euler_step(state: Tensor, velocity: Tensor, dt: float) -> Tensor`

官方代码的时间方向是动作到噪声，速度标签为noise-actions。采样时负步长把噪声拉回动作；不能同时翻转标签和遗漏负步长。detach仅针对监督标签，不针对混合状态。

## 4. 本题约定

action/noise为(B,H,D)，time为(B,)且在[0,1]。返回混合状态和detach的速度标签；状态保留梯度。Euler输入同形，dt有限且<=0。t=1为噪声，采样走向t=0。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

`experiment.py` 在固定输入上调用提交的全部接口，记录学习或采样曲线与末端指标。实验逻辑和阈值见 T3，输出同样用于可视化。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/pi0 --impl problems/robotics/pi0/starter.py
paperleet mutate robotics/pi0
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) — Official openpi Pi0.compute_loss and sample_actions time orientation
- [Verified source](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
