# 对称对比学习

[CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) 使用配对图文学习共享表示空间。图 3 给出对比训练目标，§2.5 说明温度初始化与 scale 截断。作者发布的模型代码见 [openai/CLIP](https://github.com/openai/CLIP)。

## 核心任务

### 返回值要求

#### `contrastive_logits` (需实现)

返回一个浮点 `Tensor`，形状为 `[B,B]`：第 i 行对应图像 i，第 j 列对应文本 j，B 为配对 batch 大小。保留特征的 dtype/device、两侧编码器梯度及未超过上限时的标量 `logit_scale` 梯度（超过上限时为零）。返回的是分数矩阵，不是概率、tuple 或已归约的 loss；不修改输入。

#### `clip_loss` (需实现)

返回一个标量浮点 `Tensor`，形状 `()`，dtype/device 与特征一致。它是图像到文本、文本到图像两个 batch 平均交叉熵的平均值。保留两侧特征及未截断温度的梯度，不 detach、不转成 Python float，不修改输入。

### 公式与约定

实现 `contrastive_logits` 与 `clip_loss`。输入为编码器已经生成的 `[B,D]` 图像与文本特征。逐行归一化，构造两两相似度，然后同时训练两个检索方向。第 `i` 个图像匹配第 `i` 个文本，batch 内其他项是负例。

$$
\bar i=i/\|i\|_2,\quad \bar t=t/\|t\|_2,\quad
S=\min(e^s,100)\,\bar i\bar t^\top,
$$
$$
L=\tfrac12\big[\operatorname{CE}(S,\operatorname{arange}(B))+
\operatorname{CE}(S^\top,\operatorname{arange}(B))\big].
$$

`s` 是**逆温度的对数**标量，不是温度本身。本题在前向中落实论文 scale 最大为 100 的约束，先在 log 空间截断，再取指数。这样极大的 `s` 也不会造成溢出。上限以下保留温度梯度，上限以上梯度为零。

| 项目 | 约定 |
|---|---|
| 特征 | 相同 batch 和宽度的浮点 `[B,D]`，编码器已运行 |
| 归一化 | 逐行 L2，epsilon 为 `1e-12`，零向量仍为零 |
| 正样本 | 对角线配对，batch 非空 |
| `contrastive_logits` | `[B,B]`，行对应图像、列对应文本 |
| `clip_loss` | 两个方向各自的平均交叉熵再取平均 |
| 梯度 | 两个特征张量，以及未触顶的 `logit_scale` |

官方发布仓库主要用于预训练模型推理，本题明确规定论文的训练损失与截断行为。无需实现完整图像、文本编码器。

## 小例子

两个匹配的正交单位向量，`s=log(2)` 时 logits 为 `2I`，两个方向的 loss 都为 `log(1+exp(-2))`，约 0.1269。一般非对称相似度矩阵的两个方向 loss 不相等，因此只用完全相同的图文特征测试会漏掉单向损失错误。

## Toy 实验

提供的实验训练八组随机 8 维向量和可学习温度，共 80 步。报告包括 loss、余弦相似度矩阵和两个方向的检索准确率。训练调用提交的 loss，检索则从训练后的归一化向量独立计算。

**来源：** PaperLeet 根据图 3 设计的教学改编，用可训练向量替代大型图文编码器，不复现 CLIP 预训练或零样本基准。无需下载数据或权重。T3 要求双向准确率达到 1，最终 loss 小于 .05。

## 验证

T0 检查单个配对与零特征稳定性；T1 手算正交示例；T2 检查独立推导的双向损失、特征缩放不变性、配对排列、温度行为及全部梯度；T3 训练配对检索。错误变体覆盖漏归一化、漏一个方向、误用 log scale 与遗漏上限。
