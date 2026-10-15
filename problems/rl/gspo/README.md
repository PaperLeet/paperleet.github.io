# GSPO 序列比率与截断

实现序列重要性比率和截断策略损失。

论文：[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071v1)。定位：Section 4.1, Eqs. (6–8): sequence objective and geometric importance ratio；Section 4.2, Eq. (11): equal token gradient weights。

## 精选核心

### 返回值要求

按下列结构返回，保留形状中的单例维度。标量 Tensor 的形状为 `()`，不是 Python `float`。

#### `sequence_ratio` · 需要实现

对 `[B, L]` 分数/mask（`B` 个回答、`L` 个 token 位置），返回浮点 Tensor `ratio`，形状 `[B]`。每项是有效 token 重要性比率的几何平均；空回答返回 `1`。保留输入浮点 dtype/device 及 `logp` 梯度，`old_logp` 固定；不对 batch 继续聚合。

#### `gspo_loss` · 需要实现

返回形状 `()` 的浮点标量 Tensor `loss`，将截断序列目标加负号并对 **全部 `B` 个回答** 平均，空回答以零参与平均。保留 `logp` 梯度，固定 `old_logp` 和 advantage；全空 mask 返回可反传的零 Tensor，不是 Python 数字。

### 公式与约定

只实现两个函数；组 advantage 计算、采样和分布式 RL 不属于本模块。

$$
s_i=\exp\left(\frac{\sum_t m_{it}(\log\pi_{it}-\log\pi^{old}_{it})}{\sum_t m_{it}}\right)
$$

- `sequence_ratio(logp: Tensor, old_logp: Tensor, mask: Tensor) -> Tensor`
- `gspo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor, clip_eps: float=0.2) -> Tensor`

## 接口与实验

`logp`、`old_logp`、二元 `mask` 形状为 `[B,L]`，`advantages` 为 `[B]`。先对有效 log-ratio 求和、除以有效长度，再取指数。每个回答只截断这个序列比率，与冻结 advantage 相乘取较小项，最后对 batch 求负均值。

旧策略与 advantage 不参与梯度。全 padding 回答比率为 1、损失贡献为 0，仍计入 batch 大小。必须在指数运算前屏蔽 padding。重复一个回答内部的 token 不改变结果。本题实现 GSPO，不是 GSPO-token。默认 `.2` 截断是便于观察的教学设置，不代表 Qwen 生产训练参数。

实验使用 token 比率 `[2, 0.5]`，它们的几何比率为 1，两个有效 log-probability 的梯度相同。正 advantage 的比率 1.5 和负 advantage 的比率 0.5 均被截断。产物保存学员实现计算的比率和梯度矩阵。

## Toy 来源

**PaperLeet 教学改编。** PaperLeet 自编的三回答解析实验，不是 Qwen 训练结果。 判题 harness 调用本次提交的函数，保存由其产生的指标与曲线。Lab 静态预览来自参考实现，Run/Submit 的产物来自你的代码。`experiment.py` 与 `tests/` 已提供实验和判题脚手架。

## 来源与验证

[作者说明（尚未验证作者训练代码发布）](https://qwenlm.github.io/blog/gspo/)。题目实现与合成数据由 PaperLeet 编写；论文链接固定公式依据的版本。

T0 验证接口，T1 对照参考值，T2 检查数学性质与梯度，T3 用你的实现运行 toy。在在线编辑器补全 starter 后 Run 或 Submit，无需部署完整大模型训练系统。
