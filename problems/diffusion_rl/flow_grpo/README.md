# Flow-GRPO：随机流转移与组相对策略优化

[Flow-GRPO](https://arxiv.org/abs/2505.05470) 将 flow matching 生成过程变为可按组奖励训练的随机策略。本题实现反向时间的高斯转移，以及带截断和 KL 约束的策略更新。

## 1. 从 flow ODE 到高斯策略

采用论文从数据到噪声的时间约定 $x_t=(1-t)x_0+t\epsilon$，velocity 指向噪声。采样时移向更小时间，故 $\Delta t=s-t<0$。噪声强度为 $a\geq0$ 时，定义

$$
\sigma_t^2=a^2\frac{t}{1-t},\qquad
b_\theta=v_\theta+\frac{\sigma_t^2}{2t}\left(x_t+(1-t)v_\theta\right).
$$

反向时间 Euler–Maruyama 转移为

$$
\mu_\theta=x_t+\Delta t\,b_\theta,
\qquad q=\sigma_t\sqrt{-\Delta t},
\qquad x_s=\mu_\theta+qz,\quad z\sim\mathcal N(0,I).
$$

实现 `sde_transition(state, velocity, time, next_time, noise_level)`，返回 `(mean, std)`。状态与速度具有相同非空浮点形状 `(B, ...)`；均值保持该形状，标准差是相同 dtype/device 的标量张量。要求 `0 <= next_time < time < 1`，排除公式奇异的端点。噪声强度零返回 ODE 均值和零标准差。

观察到的状态需要 detach，梯度经过 velocity。这是固定 rollout 状态下的局部策略更新。反向时间下噪声尺度使用时间步绝对值的平方根，与[官方采样器](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py)一致。

## 2. 组优势与逐转移概率比

实现 `flow_grpo_loss(current, old, reference, actions, std, rewards, clip_epsilon, beta, eps=1e-8)`，输入为：

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `current`, `old`, `reference` | `(P,G,T,D)` | 当前、采样时和参考策略的转移均值 |
| `actions` | `(P,G,T,D)` | 已采样的下一状态 |
| `std` | `(T,)` | 每步共同使用的正高斯标准差 |
| `rewards` | `(P,G)` | 每个 prompt 下 G 条轨迹的终点奖励 |

每个 prompt 至少有两条轨迹。使用总体标准差，在**每个 prompt 内部**归一化奖励：

$$
A_{pg}=\frac{R_{pg}-\overline R_p}
{\sqrt{\frac1G\sum_j(R_{pj}-\overline R_p)^2}+\varepsilon_{\rm std}}.
$$

同一轨迹的终点优势用于所有转移。组内奖励全相同时优势为零。等方差高斯的精确联合 log-density 比为

$$
\log r_{pgt}=\sum_d\frac{(a_{pgtd}-\mu_{\rm old,pgtd})^2-(a_{pgtd}-\mu_{\theta,pgtd})^2}{2q_t^2}.
$$

新旧分布的方差相同，归一化常数相消。相对固定参考策略的精确 KL 为

$$
K_{pgt}=\sum_d\frac{(\mu_{\theta,pgtd}-\mu_{\rm ref,pgtd})^2}{2q_t^2}.
$$

返回标量损失

$$
L=\frac1{PGT}\sum_{p,g,t}\left[
-\min\left(r_{pgt}A_{pg},\operatorname{clip}(r_{pgt},1-\epsilon,1+\epsilon)A_{pg}\right)
+\beta K_{pgt}\right].
$$

梯度只流向 `current`。在损失内部冻结旧均值、参考均值、采样动作、奖励和标准差。

## 3. 精确约定

本题采用论文概率比定义中的**联合高斯密度**：先对潜变量坐标求和，再对 prompt、轨迹和时间取目标均值。官方采样器采用坐标 log-density 均值作为实现归一化，得到的是几何平均概率比；本题不采用该约定。精确 KL 同样对坐标求和。

要求有限的 `0 < clip_epsilon < 1`、`beta >= 0`、`eps > 0` 和严格为正的 `std`。非法形状、空张量、非浮点张量、dtype/device 不匹配、非有限输入都应抛出 `ValueError`。保持 CPU float32/float64，不修改输入。零优势和已触发上界截断的正优势应稳定处理，避免指数溢出；其余非零目标项位于浮点可表示范围。零噪声 ODE 不具有普通高斯密度，不能用于此损失。

## 4. 验证层次

T0 检查接口和非法时间表。T1 用独立标量循环检查转移统计、高斯密度、组内归一化、截断与 KL。T2 检查含 score 修正的速度 Jacobian、解析策略梯度、冻结边界、各 prompt 奖励偏移独立性、时间均值和正负优势两侧的截断。T3 通过当前策略的新随机轨迹进行 80 次真实在线更新。

## 5. 在线实验

三个标量 prompt 决定目标下一状态。一个可训练权重控制条件速度，每个 prompt 采样 24 次转移，终点奖励为目标误差平方的负值。实验通过你的 SDE 函数采样，再用你的损失更新策略，图表展示期望下一状态的目标误差。CPU 教学实验使用单转移轨迹，多时间步归约由 T1/T2 覆盖；无需 SD3 图像训练，也不复现论文的去噪步数削减基准。

## 6. 怎么运行

```bash
paperleet init diffusion_rl/flow_grpo
paperleet test diffusion_rl/flow_grpo
```

## 7. 来源

- Liu 等，[第 4.1 节式 (4)–(6)、第 4.2 节式 (9)–(13)](https://arxiv.org/html/2505.05470v1)。
- [官方反向时间采样器](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py)。

本题根据数学定义独立编写精简实现。
