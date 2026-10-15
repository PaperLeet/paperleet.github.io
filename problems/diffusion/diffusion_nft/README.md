# DiffusionNFT 正负双分支流匹配

## 1. 题目目标

实现 [DiffusionNFT 式 (5)](https://arxiv.org/html/2509.16117v1#S3.SS2)。

## 2. 接口与返回值

### 返回值要求

#### `implicit_velocities`（需实现）

按顺序返回二元组 `(positive, negative)`，两项都是与 `current` 同形状 `[B,...]`、dtype/device 的 Tensor；B 为样本数，末尾为全部特征维度。保留两条分支对 current 的梯度，并在函数内部 detach old；不修改输入。

#### `nft_loss`（需实现）

返回与 `current` 同 dtype/device 的标量 Tensor，形状 `()`。每个样本先对全部非 batch 坐标的平方误差求和，再对 B 个样本取平均；不能把全部元素直接求平均。梯度只进入 current；old、target、rewards 均停止梯度，不修改输入。

#### 已提供的局部辅助函数

- `_check_velocity`：已提供：当velocity 与 beta 满足约定时返回 Python `None`，不返回布尔值或校验后的张量。无效输入抛出 `ValueError`。仅检查，不修改输入或其梯度设置；不存在返回张量的形状、dtype/device。

### 公式与接口说明

单个可训练 velocity `v_current` 围绕固定 rollout velocity `v_old` 隐式定义两个策略：

$$
v^+=(1-\beta)v_{old}+\beta v_{current},\qquad
v^-=(1+\beta)v_{old}-\beta v_{current}.
$$

给定前向过程的 velocity 目标 `v` 及 `[0,1]` 内的最优性概率 `r`，优化：

$$
\mathcal L=\operatorname{mean}_{b}\left[r_b\|v_b^+-v_b\|_2^2+(1-r_b)\|v_b^--v_b\|_2^2\right].
$$

隐式分支使用 **beta 本身**，不是倒数。负样本通过非负平方误差训练负分支，不能改成给 MSE 乘负奖励。

### 实现接口

| 函数 | 输入 | 输出 |
|---|---|---|
| `implicit_velocities(current, old, beta)` | 两个同形状 `(B,...)` velocity 与正有限 beta | 正 velocity、负 velocity，按此顺序 |
| `nft_loss(current, old, target, rewards, beta)` | 三个匹配的 velocity 与 `(B,)` 奖励 | 标量损失 |

对**所有非 batch 坐标求平方和**，然后对样本求平均。这明确了图像或 token 形状特征的归约约定。奖励已是归一化概率，不要在函数内裁剪、再归一化或跨样本广播。starter 提供验证，使用非空 CPU float32/float64 输入。

梯度仅进入 `current`。在 `implicit_velocities` 内 detach `old`；在 `nft_loss` 内 detach `target` 和 `rewards`，即使这些输入意外启用了梯度也应如此。保留输入值和 dtype/device。

前向加噪、rollout 数据与训练循环均由脚手架提供。toy 约定 `x_t=(1-t)*x0+t*noise`，velocity 目标为 `noise-x0`。在线生成、奖励模型及旧策略 EMA 更新不在本题范围内。

## Toy：正样本、负样本和中性样本

**来源分类：teaching adaptation。** 固定二维前向过程 minibatch 的五个奖励为 `0`、`0.25`、`0.5`、`0.75`、`1`。用你的损失优化其 velocity 输出，并与独立推导的二次目标最优点比较：

$$
v_{current}^*=v_{old}+\frac{2r-1}{\beta}(v-v_{old}).
$$

正负样本推动相反方向，中性样本停留在旧 velocity。图表展示损失收敛与不同奖励对应的轨迹。这是小规模输出空间实验，不声称完成在线策略改进或复现图像生成 benchmark。

T0 检查接口；T1 使用独立标量公式及端点奖励；T2 验证解析梯度、冻结上下文、中性锚定、反射对称、特征归约与闭式最优点；T3 调用提交函数执行训练并生成图表。预览使用参考实现。
