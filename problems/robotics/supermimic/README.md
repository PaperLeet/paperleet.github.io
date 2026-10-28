# SuperMimic：可微想象回报与价值目标

## 1. 练习目标

实现 [SuperMimic](https://arxiv.org/abs/2610.28480) Section 3.3 的回报计算：可微 actor 目标与冻结的 critic 监督目标。对应 2026 年基于可微世界模型的方法，见[作者项目页](https://supermimic.github.io/)。

## 2. 接口与返回值

| 函数 | 输入 | 返回 |
|---|---|---|
| `imagined_return(rewards, termination, bootstrap, gamma)` | `(B,H)` 奖励及终止概率，`(B,)` 终点价值，折扣 | 零维标量 Tensor：应当**最大化**的 batch 平均回报 |
| `lambda_return_targets(rewards, termination, values, gamma, lam)` | `(B,H)` 奖励和终止概率，`(B,H+1)` 状态价值，折扣与 lambda | 已停止梯度的 `(B,H)` critic 目标 |

输入为有限 CPU float32/float64，dtype/device 一致，batch 和 horizon 非空。保持输入不变，返回 dtype/device 不变。`gamma` 有限且属于 `[0,1)`，`lam` 属于 `[0,1]`，终止概率属于 `[0,1]`。提供的 `_rollout` 仅验证输入，返回 `None` 或抛出 `ValueError`。

### 返回要求与时间约定

`rewards[:,t]` 表示到达奖励 `r_(t+1)`，`termination[:,t]` 表示 `d_(t+1)`；`values[:,t]` 表示 `V_t`，最后一列为 `V_H`。

actor 输出必须保留奖励、终止概率与终点价值三条梯度路径。调用方冻结世界模型和 critic 的**参数**，但保留它们关于状态、动作的导数。因此不能把终点价值整体 detach。critic 目标函数则在内部冻结三个张量输入，输出 `requires_grad=False`。

## 3. 核心公式

从 `c_0=1` 开始，更新存活概率 `c_(t+1)=c_t*(1-d_(t+1))`。actor 最大化：

$$
\hat J=\frac1B\sum_b\left[\sum_{t=0}^{H-1}\gamma^t c_{b,t}r_{b,t+1}+\gamma^H c_{b,H}V_{b,H}\right].
$$

先计入本次到达奖励，再用本次终止概率限制后续回报。保持软概率可微，不取整、不做硬阈值。

critic 目标以 `R_H=V_H` 初始化，再反向递推：

$$
R_t=r_{t+1}+\gamma(1-d_{t+1})\left[(1-\lambda)V_{t+1}+\lambda R_{t+1}\right].
$$

返回时间顺序 `t=0,...,H-1` 的目标，不包含初始化列 `R_H`。`lambda=0` 对应单步 bootstrap，`lambda=1` 的首个目标等于该样本完整的 bootstrap 回报。

## 4. 小例子

奖励 `[2,3,100]`，终止 `[0,1,0]`，`gamma=.5` 时，回报为 `2+.5*3=3.5`，与终点价值无关。终止转移仍获得奖励 `3`；其后的奖励和终点价值不再贡献。软概率保留相应比例的未来回报。

## 5. 确定性实验

四个一维初始状态共享策略 `action=gain*(goal-state)`。固定可微动力学执行八步，解析函数提供跟踪奖励、终止概率和终点价值。你的 actor 回报驱动梯度上升，学习共享反馈增益；lambda 目标通过 `lambda=1` 的完整回报恒等式独立核验。

这是回报机制的教学改编，采用固定解析世界模型，没有拟合论文中的神经网络，也不复现人形跟踪、经验采集或 sim-to-real 结果。

## 6. 运行实现

补全 `starter.py` 的两个函数后通过 PaperLeet 运行。T0 检查接口，T1 使用独立标量递推，T2 检查有限差分梯度及时间边界，T3 训练反馈策略并生成图表。

## 7. 来源

- [SuperMimic 论文](https://arxiv.org/abs/2610.28480)：Section 3.3，Eq. (2) actor 回报与 Eq. (3) critic 目标。
- [官方项目](https://supermimic.github.io/)：Qiayuan Liao 等，CoRL 2026。
