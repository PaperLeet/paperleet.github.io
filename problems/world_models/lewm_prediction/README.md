# LeWM 联合预测与潜在展开

[LeWorldModel](https://arxiv.org/html/2603.19312v2) 同时学习编码器和动作条件预测器。训练时用 `(z_t,a_t)` 预测下一帧表示 `z_{t+1}`，**上下文与目标两条编码路径都保留梯度**，再通过已提供的 SIGReg 正则项防止表示坍塌。

实现两个函数：

| 函数 | 输入 | 输出 |
|---|---|---|
| `lewm_objective(embeddings, actions, predictor, regularizer, weight=.1)` | 表示 `(B,T,D)`、动作 `(B,T-1,A)`、两个可调用对象 | 三个标量 `(total, prediction, regularization)` |
| `latent_rollout(initial, actions, transition)` | 初始状态 `(B,D)`、动作 `(B,H,A)`、转移函数 | 预测状态 `(B,H,D)` |

## 联合训练目标

### 返回值要求

#### `lewm_objective` (需实现)

按顺序返回 `(total, prediction, regularization)`，恰好三个标量浮点 `Tensor`，形状 `()`，保持 embedding/回调的共同 dtype/device。对于 embeddings `[B,T,D]`，prediction 平均 B*(T-1)*D 个下一状态平方误差；regularization 是覆盖全部 T 帧的回调标量；total 为 prediction + weight*regularization。上下文与目标 embedding 两条路径及回调梯度都要保留，下一状态 target 也不能冻结。不要 detach 任一分量或修改输入。

#### `latent_rollout` (需实现)

返回一个浮点 `Tensor` `[B,H,D]`，保持 initial/transition 的 dtype/device；按时间顺序包含 H 个预测状态，不包含初始 `[B,D]` 状态。actions 为 `[B,H,A]`，A 为动作维度。保留每一步经递推传向 initial、actions 与可微 transition 参数的梯度。不 detach、不返回 list、不修改输入。

#### `_tensor` (已提供)

已提供校验函数：x 是秩为 ndim 的有限、非空 float32/float64 Tensor时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

### 公式与约定

调用一次 `predictor(embeddings[:, :-1], actions)`。提供的预测器满足时间因果性，输出 `(B,T-1,D)`；你不需要实现它的架构。把结果与 `embeddings[:, 1:]` 对齐，并按官方 `train.py` 的约定对批次、时间、特征的**全部元素取平均**：

$$L_{pred}=\frac{1}{B(T-1)D}\sum_{b,t,d}(\hat z_{b,t+1,d}-z_{b,t+1,d})^2.$$

对**全部 T 帧**调用一次 `regularizer(embeddings.transpose(0,1))`，其输入是 `(T,B,D)`，返回值已完成逐时间统计及平均。返回 `prediction + weight * regularization` 以及两个分项。不要把预测项系数写成 `1-weight`，不要 detach 未来目标、引入 EMA 目标，或把时间维合进批次维。`weight` 为有限非负数，也允许为零。

## 自回归展开

按动作顺序调用 `transition(current_state, actions[:, t])`，将本步预测作为下一步状态。返回 H 个预测，**不包含初始状态**。转移函数保持 `(B,D)` 形状，初始状态、动作和前序预测都需要保留梯度。

本接口明确采用**单状态上下文 N=1** 的教学版本。论文预测器可接收更长的历史窗口，其 Transformer、图像编码器与历史缓存不属于本题。训练使用观测表示作为上下文，展开使用递归预测的表示。

所有浮点张量使用相同 float32/float64 类型与设备，要求 `B,D,A,H>=1`、`T>=2`。不要修改输入。starter 已提供输入检查。

例如 `z_0=2`、动作 `[1,-2]`、转移 `transition(z,a)=.5*z+2*a`，应得到 `[3,-2.5]`。

## Toy：学习一维潜在动力学

**教学改编。** 标量编码器与两个参数的线性预测器在 24 条、每条含四次转移的序列上联合训练。实验调用你的目标函数，提供按照官方 `[0,3]` 积分节点实现的一维固定方向 SIGReg，再调用你的展开函数预测其中一条训练轨迹。图表显示预测 MSE、总损失与编码器缩放系数。

该实验不复现图像训练，也不评估未见数据泛化。T2 另外使用解析公式检查未来目标梯度、递推梯度和时间因果性；T3 实际训练你提交的函数，预览使用参考解。
