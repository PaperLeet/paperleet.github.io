# DPO 偏好目标与隐式奖励

## 1. 这篇论文解决什么

DPO 把 KL 正则化奖励最大化改写为偏好对上的分类目标，训练时不需要单独拟合奖励模型。[论文第 4 节](https://arxiv.org/html/2305.18290v3#S4) 从最优策略推导这个等价关系。

本模块目标：从策略与参考模型的序列分数实现偏好损失和隐式奖励。 只实现 dpo_loss；序列 log-prob 聚合与小型训练实验已提供。 完成后记为这篇论文的一个核心模块。

## 2. 核心公式与题面

只需实现 `dpo_loss`。starter 已提供 `sequence_logprob`，处理 next-token 标签移位、回答 mask 和 log-prob 求和，直接产生目标函数所需的序列分数。

四个序列 log-prob 输入的形状都是 `[B]`，其中 `B` 是偏好对的数量。

### 返回格式

`dpo_loss` 必须按以下顺序返回一个包含 **3 个 Tensor 的元组**：

```python
return loss, chosen_rewards, rejected_rewards
```

| 返回值 | 形状 | 含义与梯度要求 |
|---|---|---|
| `loss` | `()`，零维 Tensor | 对 batch 取平均的 DPO loss，保留对策略输入的梯度 |
| `chosen_rewards` | `[B]` | `beta * (pi_chosen - ref_chosen)`，返回时 detach |
| `rejected_rewards` | `[B]` | `beta * (pi_rejected - ref_rejected)`，返回时 detach |

只写 `return loss` 不符合本题接口；测试会读取全部三个返回值。`loss` 不要调用 `.item()` 或 `.detach()`；两个奖励向量不要取平均，也不要用 `squeeze()` 压掉 batch 维，即使 `B = 1` 也应保留 `[1]`。计算 loss 时 reference 输入视为固定值，不应收到梯度；用于计算 loss 的策略分支仍需保留梯度。

### 目标函数

$$
h=(\log\pi_c-\log\pi^{ref}_c)-(\log\pi_r-\log\pi^{ref}_r),\qquad
L=-\mathbb E[(1-\varepsilon)\log\sigma(\beta h)+\varepsilon\log\sigma(-\beta h)].
$$

$c,r$ 分别代表 chosen 和 rejected，$\beta$ 默认为 0.1，标签平滑 $\varepsilon$ 默认为零，此时就是原论文的核心目标；非零平滑是保留的可选兼容参数。隐式奖励为 $\beta(\log\pi-\log\pi^{ref})$。这是相对参考策略的奖励，所以不能拿 chosen 的原始 log-prob 当作预测偏好分数。

## 3. 从公式到代码

```text
已提供 sequence_logprob → 四个 [B] 序列分数
冻结 reference；计算 chosen / rejected reward 和 margin
用稳定的 log-sigmoid 合并正反标签 → 标量 loss
```

整个计算应留在 log 空间。对于 $\beta h=-10000$，先求 sigmoid 会下溢为零，再取 log 会产生无穷；稳定函数直接计算同一个数学目标，不需要人为截断 margin，也不会抹掉困难样本的梯度。

## 4. 本题约定

| 项目 | 本题 | 相关方法或库 |
|---|---|---|
| token 聚合 | 回答 log-prob 总和 | SimPO 使用长度归一化，目标不同 |
| mask | 与右移后的标签对齐 | prompt 与 padding 均排除 |
| label smoothing | 正反 sigmoid loss 的凸组合 | TRL v0.23.1 的保守 DPO 说明 |
| reference 与奖励 | reference 无梯度，返回奖励 detach | 奖励用于诊断而非再次反传 |

[TRL v0.23.1 文档](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer) 将这一平滑联系到保守 DPO；它与 `robust` loss 的标签噪声校正不是同一公式。T3 使用六个候选、固定非均匀 reference 和两万对 Bradley–Terry 偏好，比较训练策略与 $\pi^*\propto\pi_{ref}e^{r^*/\beta}$，同时检查隐式奖励排序完全一致。

当参考模型更偏爱某个回答时，策略必须相对参考提高得更多，才能形成正的偏好边际。这个相对量解释了为什么固定参考分布会改变闭式最优策略，也解释了实验中选择非均匀参考分布的原因。

## 5. 交互可视化

[打开标签与 mask 网格](#viz-shift_mask_grid)：检查每个位置预测的下一个 token，查看被 prompt mask 排除的贡献。

[打开 margin 曲线](#viz-loss_vs_margin)：调节 $\beta,\varepsilon$，观察 loss 与梯度权重；训练奖励散点展示隐式奖励逐渐恢复真实奖励的排序。

## 6. 怎么运行

```bash
paperleet init rl/dpo
paperleet test rl/dpo
```

编辑 `workspace/rl/dpo.py` 中的 `dpo_loss`，保留已提供的 `sequence_logprob`。优先查看 reference-adjusted margin、beta 缩放、梯度与极端值测试。序列测试检查辅助代码的集成，现成行为实验训练一个小 categorical 策略。

## 7. 延伸

继续 [GRPO](../grpo/README.md)，比较离线偏好训练与在线组采样。也可以研究长度归一化或带噪偏好改变了哪些最优性假设。
