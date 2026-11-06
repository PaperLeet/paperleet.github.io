# MAE：掩码像素重建

## 1. 问题

[Masked Autoencoders](https://arxiv.org/abs/2111.06377) 通过预测输入中被移除的图像内容学习表示。原方法采用不对称编码器与解码器，并遮挡较大比例的 patch。本题提取其中的像素重建目标，输入已完成 patch 划分。

实现 `normalize_patch_targets(target, eps=1e-6)` 和 `masked_reconstruction_loss(prediction, target, mask, norm_pix_loss=False, eps=1e-6)`。预测与目标均为 `[B,N,P]`，依次表示图像、patch 和展平像素坐标。`mask[b,n]=True` 表示被移除、需要监督重建的 patch。两个函数分别返回同形状的归一化目标和标量损失。

## 2. 核心公式

可选目标归一化在每个 patch 内独立进行。依据[官方 `forward_loss`](https://github.com/facebookresearch/mae/blob/main/models_mae.py) 的 `target.var(dim=-1)`，本题采用除以 `P-1` 的**样本方差**：

$$
\mu_{bn}=\frac1P\sum_p x_{bnp},\quad
v_{bn}=\frac1{P-1}\sum_p(x_{bnp}-\mu_{bn})^2,\quad
\hat x_{bnp}=\frac{x_{bnp}-\mu_{bn}}{\sqrt{v_{bn}+\epsilon}}.
$$

原始目标取 $t=x$，归一化目标取 $t=\hat x$。令 $M=\sum_{bn}m_{bn}$：

$$
e_{bn}=\frac1P\sum_p(\mathrm{prediction}_{bnp}-\operatorname{sg}(t_{bnp}))^2,\qquad
\mathcal L=\frac{\sum_{bn}m_{bn}e_{bn}}{\max(M,1)}.
$$

损失内的 `sg` 将目标当作固定数据；独立调用归一化函数仍保留梯度。这一显式 detach 与空 mask 返回零属于本题接口约定；官方 MAE 输入为原始图像数据，训练 mask 非空。

例：`[-1,1]` 的样本方差是 `2`，归一化除以 `sqrt(2+eps)`。两张图分别遮挡一个、两个 patch 时，三个 patch 的权重相同。

## 3. 公式到代码

1. 沿最后的像素轴减均值，保留维度以广播。
2. 将中心化像素的平方和除以 `P-1`，在平方根内部加入 epsilon。
3. 损失内先 detach 目标，仅在 `norm_pix_loss=True` 时调用本文件的目标归一化函数。
4. 先对每个 patch 内像素误差求均值，再乘 mask 求和，除以至少为一的遮挡 patch 总数。

保留提供的输入校验。不能导入参考实现，也不要 detach 预测或自动归一化预测。

## 4. 本题约定

- 输入是非空 `[B,N,P]`，值有限、同 dtype/device 的 float32/float64 tensor；支持非连续视图，不修改输入。mask 为同设备的 Boolean `[B,N]`。
- 原始像素损失允许 `P=1`；归一化要求 `P>=2`。常量 patch 的归一化结果为零。
- `eps` 是有限正 Python int/float，拒绝 bool 与 Tensor 标量；`norm_pix_loss` 必须是 Python bool。
- 归约对**所有遮挡 patch** 求均值。可见 patch 的预测梯度恰为零。空 mask 返回与预测保持计算图连接、梯度为零的标量零。
- 每个 patch 的独立平移不改变归一化结果。非零缩放 `s` 同时将 epsilon 缩放为 `s²` 倍时，结果乘以 `sign(s)`；epsilon 必须保持正值。

测试包含独立标量 oracle、预测解析梯度、归一化 VJP 有限差分、不等遮挡数量、平移、重复 batch、空/全 mask 与单像素边界。

## 5. 交互图与 CPU 实验

Lab 训练一个包含偏置的**共享 3→4 线性解码器**，模型共 16 个可训练参数。24 个确定性三角函数 context 向量组成六张图、每图四个 patch，其中 14 个被监督；目标由固定线性映射生成。Adam 使用你的原始像素损失，以 `.06` 学习率进行 **120 次真实更新**，每十步记录。目标也放进 optimizer，以验证正确停止梯度后它们完全不移动。

图表比较提交损失与独立 Python 像素 MSE，并展示一个 patch 的预测。参考 CPU 运行中 MSE 从约 `.671` 降至 `6.6e-6`。T3 要求最终 MSE 小于 `1e-4` 且小于初值的 `.001`，损失与 oracle 差异小于 `1e-12`，目标漂移为零。该实验只训练合成解码器，不复现图像遮挡流程、ViT 或 MAE 基准结果。

## 6. 运行方式

补全两个函数后选择 **Run tests**。完整测试及训练实验支持 CPU/WASM，无需外部数据或预训练网络。

```bash
paperleet test representation/mae --impl problems/representation/mae/solution.py
paperleet mutate representation/mae
paperleet ci --problem representation/mae
paperleet vizdata representation/mae
```

## 7. 延伸阅读

对照 MAE 第 3 节、官方 `forward_loss`、`patchify` 与随机遮挡流程。为什么只重建移除位置？目标归一化如何改变亮度与对比度敏感性？为什么此处样本方差不同于 Barlow Twins 的 BatchNorm 总体方差？

进一步比较以离散 tokenizer ID 为目标的 [BEiT](https://arxiv.org/abs/2106.08254)，以及以在线教师软概率为目标的 [iBOT](https://arxiv.org/abs/2111.07832)。三道题分别锁定不同的预测目标和归约契约。
