# iBOT：中心化教师与掩码 Patch 蒸馏

## 1. 问题

[iBOT](https://arxiv.org/abs/2111.07832) 将图像级自蒸馏与在线教师的遮挡 patch 预测结合。其 patch 目标是学习到的 prototype 上的概率分布，而非固定 tokenizer ID。本题从给定 logits 出发，只实现**同一全局视图、单 crop 的 patch 分支**。

实现 `teacher_patch_targets(teacher_logits, center, teacher_temp=.04)` 和 `ibot_patch_loss(student_logits, teacher_logits, mask, center, student_temp=.1, teacher_temp=.04)`。学生与教师 logits 为 `[B,N,K]`，center 为 `[K]`，Boolean mask `[B,N]` 选择被监督的学生 patch。两个函数分别返回停止梯度的 `[B,N,K]` 教师概率与标量损失。

## 2. 核心公式

先对教师 prototype 分数做中心化，再除温度进行 softmax，停止整个教师分支的梯度：

$$
q_{bnk}=\operatorname{sg}\!\left[\operatorname{softmax}_k\left(\frac{t_{bnk}-c_k}{\tau_t}\right)\right],\qquad
\log p_{bnk}=\operatorname{logsoftmax}_k\left(\frac{s_{bnk}}{\tau_s}\right).
$$

[官方 `iBOTLoss.forward`](https://github.com/bytedance/ibot/blob/main/main_ibot.py) 先计算**每张图内部的遮挡均值**，然后平均所有图像。记 $M_b=\sum_n m_{bn}$：

$$
\ell_{bn}=-\sum_k q_{bnk}\log p_{bnk},\qquad
\mathcal L=\frac1B\sum_b\frac{\sum_n m_{bn}\ell_{bn}}{\max(M_b,1)}.
$$

没有遮挡 patch 的图贡献零，但仍计入 `B`。选中学生分数的梯度是 $(p_{bnk}-q_{bnk})/(B M_b\tau_s)$。教师 logits 与 center 不求梯度。当 `p=q` 时损失等于教师熵，通常为正；软教师的最优交叉熵并不是零。

## 3. 公式到代码

1. 减去 `[K]` center，除以教师温度，沿最后一轴 softmax，detach 整个目标。
2. 学生 logits 除以学生温度，沿最后一轴计算稳定 log-softmax。
3. 乘教师概率，沿 prototype 轴求和并取负。
4. 应用 patch mask；每张图的和除以至少为一的遮挡数量，再对所有图求均值。

调用本文件的 helper，保留输入校验。不要将软教师换成 argmax 标签、打乱图像对应关系，或对所有遮挡 patch 做全局均值。CLS 分支的跨视图配对不适用于这个 patch 分支。

## 4. 本题约定

- 学生/教师为形状一致且非空的 `[B,N,K]`，center 为 `[K]`，均为有限、同 dtype/device 的 float32/float64。mask 为同设备 Boolean `[B,N]`。不修改输入，保持 dtype/device，支持非连续视图。
- 两个温度均为有限正 Python int/float，拒绝 bool 与 Tensor 标量。默认值对应官方名义温度；本题不实现温度调度。
- 教师目标连同 center 一起停止梯度。可见学生 patch 梯度为零；全空 mask 返回可微零。`K=1` 时目标概率为一，交叉熵为零。
- 调用期间 center 是固定输入。本题不实现 center EMA、教师 EMA、多 crop 均值、共享投影头结构或图像级 CLS 目标。`[K]` 是官方 `[1,1,K]` 广播 center 的简化接口。
- 各 patch 的学生/教师 logits 可分别加常数而不改变概率。教师与 center 同时加同一 prototype 向量也不改变目标。

测试覆盖独立标量概率与 CE、学生解析及有限差分梯度、教师/center 停梯度、不等及空 mask 行、一致置换、软目标熵、单类别词表与极端分数。

## 5. 交互图与 CPU 实验

Lab 直接优化一个 **`3×4×4` 学生 logit 表**，拟合确定性的固定教师 logits 与 center。三张图分别选中 `1,3,0` 个 patch。Adam 以 `.08` 进行 **120 次真实更新**；学生/教师温度取 `.7/.5`，便于观察软分布。教师与 center 故意加入 optimizer 检查梯度边界，正确实现应使它们保持不变。

图表展示提交 CE、教师熵下界、独立计算的图像均衡 KL，以及一个遮挡 patch 的教师/学生概率。参考 CPU 的 KL 从约 `.10581` 降到 `1.8e-7`；CE 接近 `.81916`，而非零。T3 要求 KL 小于 `1e-5` 且小于初值的 `.001`，与 oracle 差异小于 `1e-12`，教师、center 和可见学生 logits 漂移均为零。该有界合成蒸馏实验不训练编码器、不更新在线 tokenizer，也不复现 iBOT 基准结果。

## 6. 运行方式

补全两个函数后选择 **Run tests**。原始完整测试与真实优化循环支持 CPU/WASM，无需图像、GPU 或预训练权重。

```bash
paperleet test representation/ibot --impl problems/representation/ibot/solution.py
paperleet mutate representation/ibot
paperleet ci --problem representation/ibot
paperleet vizdata representation/ibot
```

## 7. 延伸阅读

对照 iBOT 第 3 节与官方 `iBOTLoss` 的两个分支。为什么 patch 匹配采用同一全局视图，而 class token 使用跨视图项？教师中心化与温度分别起什么作用？为什么教师权重与 center 必须在这个目标的梯度下降之外更新？

比较 [BEiT](https://arxiv.org/abs/2106.08254) 的硬 token 目标和 [DINO](https://arxiv.org/abs/2104.14294) 的图像级自蒸馏。推导图像均衡的遮挡均值何时等于全局遮挡 token 均值、何时不同。
