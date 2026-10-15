# GAE 反向递推

## 1. 这篇论文解决什么

策略梯度需要知道一个动作比当前策略的平均选择好多少。GAE 把单步 TD 与 Monte Carlo 回报连接起来，让同一个参数控制估计的偏差和方差。[论文第 3 节，式 16–18](https://arxiv.org/html/1506.02438v6#S3)。

本模块目标：从 TD 残差实现 GAE 反向递推，正确处理 episode 边界。 只实现 advantage 与 value target；轨迹和价值估计由测试提供。 完成后记为这篇论文的一个核心模块。

## 2. 核心公式与题面

### 返回值要求

按下列结构返回，保留形状中的单例维度。标量 Tensor 的形状为 `()`，不是 Python `float`。

#### `compute_gae` · 需要实现

返回 `(advantages, returns)`，按此顺序组成两个浮点 Tensor 的元组。两者形状均为 `[T, N]`（`T` 个时间步、`N` 个环境），保留 `rewards` 的 dtype/device；其中 `returns = advantages + values`。即使 `T = N = 1` 也保留两个轴，不对时间或环境求平均。

### 公式与约定

实现 `compute_gae`，返回 advantage 和 value target。奖励 $r_t$、当前价值 $V_t$、下一状态价值 $V_{t+1}$ 都是 `[T,N]`，时间在前；两个结束标记也是 `[T,N]` 布尔张量。

$$
\delta_t=r_t+\gamma(1-\mathrm{term}_t)V_{t+1}-V_t,
\qquad
\hat A_t=\delta_t+\gamma\lambda(1-\mathrm{done}_t)\hat A_{t+1}.
$$

这里 $\mathrm{done}=\mathrm{term}\lor\mathrm{trunc}$，递推初值 $\hat A_T=0$，返回值为 $(\hat A,\hat A+V)$。$\gamma$ 决定未来奖励的折扣，$\lambda$ 决定远期 TD 残差的权重。默认分别为 0.99 和 0.95。前者改变优化目标的时间尺度，后者调节估计器，二者不能互相替代。$\lambda=0$ 时只保留当前残差；$\lambda=1$ 时中间价值项抵消，剩下带末状态 bootstrap 的完整回报。

## 3. 从公式到代码

```text
输入 r,V,next_V,term,trunc: [T,N]
计算 delta: [T,N]
carry: [N]，初始为零
沿 t=T-1,...,0 反向遍历：
    当前残差 + 折扣后的下一步 carry → A[t]
返回 A 和 A+V: 两个 [T,N]
```

不要把不同环境串成一条轨迹。每一列有自己的结束标记；一次向量运算可以同时推进所有环境，时间轴必须按依赖方向处理。输出 dtype 和 device 跟随奖励，不能为了创建初值而悄悄转换为默认精度。

## 4. 本题约定

| 项目 | 本题 | 常见实现的选择 |
|---|---|---|
| 终止 | 无后续价值，不 bootstrap | 通常与本题一致 |
| 时间截断 | 保留真实末状态的价值，但切断递推 | 某些 rollout 示例合并两个标记，需要另行补偿 |
| `next_values` | 每个时间步显式传入 | 也可由 `values[t+1]` 与最后价值拼接 |
| returns | 始终是 $\hat A+V$ | 不等于任意 $\lambda$ 下的纯 MC 回报 |

T3 用动态规划计算七状态随机游走的精确 $Q,V$，固定扰动价值函数后，在同一批 4000 条轨迹上比较首个动作的策略梯度。偏差针对精确梯度计算；图中的方差是单轨迹估计的方差，不能拿 advantage 的均值代替策略梯度来验证无偏性。

## 5. 交互可视化

[打开 GAE 时间轴](#viz-gae_timeline)：拖动 $\gamma,\lambda$，观察残差 $\delta_t$ 如何向前传播为 $\hat A_t$，并查看两种 episode 边界。

[打开偏差与方差曲线](#viz-bias_variance_vs_lambda)：移动 $\lambda$，对照同一批轨迹的偏差、方差以及 Monte Carlo 端点。

## 6. 怎么运行

```bash
paperleet init rl/gae
paperleet test rl/gae
paperleet test rl/gae --layers T0,T2
```

编辑 `workspace/rl/gae.py`。先验证 $\lambda=0$ 和单步边界，再测试 episode 隔离，最后查看行为实验报告。

## 7. 延伸

下一题 [PPO](../ppo_clip/README.md) 把 advantage 放进策略目标。理解 GAE 后，可继续研究 value function 拟合误差如何影响优化，以及截断 rollout 的 bootstrap 与真正终止的差别。
