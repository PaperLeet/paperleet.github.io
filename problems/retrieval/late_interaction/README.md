# ColBERT：逐 token MaxSim 与成对排序

## 1. 论文解决什么问题

把所有 token 压成一个均值向量，可能丢失“分别匹配两个概念”的证据。ColBERT 保留查询和文档的多个向量，分别编码后再交互打分。本题供给投影后的 token 向量，只实现余弦 MaxSim 和成对排序损失；不下载编码器，不训练 BERT，无需 GPU。

## 2. 核心公式

先逐向量归一化，再为每个有效查询位置找文档内的最佳匹配，最后累加：

$$\widehat{x}=\frac{x}{\max(\|x\|_2,\epsilon)},\qquad S_{b,n}=\sum_{i:\,m^q_{b,i}}\max_{j:\,m^d_{n,j}}\widehat{q}_{b,i}^{\mathsf T}\widehat{d}_{n,j},\quad\epsilon=10^{-12}.$$

对正负文档分数，最小化正样本的 softmax 负对数概率：

$$\mathcal L=\frac1B\sum_b\log(1+\exp(S_b^- - S_b^+)).$$

归一化对应[原文 §3.2](https://arxiv.org/abs/2004.12832)，余弦打分与成对训练对应 §3.3、式 (3)。

## 3. 从公式到代码

`maxsim_scores(query, documents, query_mask=None, document_mask=None)` 计算所有查询与所有文档的组合，而非逐行配对。两个查询轴向量，对含两个轴向量的文档打 2 分，对只有一个混合向量的文档打 $\sqrt2$ 分。逐 token 归一化，在**文档 token 维**取最大，在**查询 token 维**求和。

`pairwise_loss(positive, negative)` 对 `[B]` 分数对返回平均损失。正样本应比负样本高；不要直接对很大的分差求指数。形状与 mask 校验函数 `_checked_masks`、`_check_pairs` 已提供，两个核心函数需要你实现。

## 4. 本题约定与论文边界

- 输入是有限 CPU float32/float64 `query[B,Q,D]`、`documents[N,L,D]`，dtype/device 相同。返回 `[B,N]`，保持精度与设备，支持非连续张量；查询、文档两侧都必须保留梯度，不得修改输入。
- Boolean mask 分别为 `[B,Q]`、`[N,L]`，**True 表示保留**。省略时保留全部位置。padding 不应成为最佳匹配，即使真实匹配分数全为负。
- 原版 ColBERT 的 `[MASK]` **查询增强位置参与打分**。这里的 `query_mask=False` 只代表不存在的批次填充位置，不能据此删除所有 BERT `[MASK]` 位置。本题没有分词器或特殊标记过滤，由调用者供给编码位置与 mask。
- B/N/Q 可以为零，物理文档长度 L 与特征维度 D 必须为正。空查询或全遮挡查询对有效文档贡献零；全遮挡文档始终得 `-inf`，包括空查询情况。这是明确的教学扩展约定。排除无效文档后的梯度必须有限、被遮挡位置梯度为零。
- 精确相同的最大值按 PyTorch `max(dim)` 的第一个有效位置路由梯度；这不是论文规定的唯一次梯度。正比例缩放不变性只在缩放前后范数均超过 epsilon 时成立。
- 成对损失输入为有限、非空、同 dtype/device 的 `[B]` 分数，输出标量；正负两侧均有梯度。已提供校验函数拒绝错误形状、mask 和不兼容精度。
- 仅覆盖余弦版本，不实现平方 L2 变体、BERT、投影训练、查询增强生成、标点过滤、ANN 索引或 MS MARCO 评测。

## 5. 交互实验

使用四个双 token 合成查询、八个固定文档。每个相关文档保留两个轴向量，其负样本只有混合向量；两者归一化后的均值方向完全相同，用来展示池化会丢失哪些信息，不代表一般数据上的检索性能结论。

“我的运行结果”调用你提交的打分函数和损失，对查询向量执行完整 **120 步 Adam 更新**。实验展示损失与正负分差曲线，使用独立 Python 标量余弦公式检查结果，并对四个给定的合成标签计算 Recall@1。初始结果选择混合负样本，目标是优化后召回全部四个相关文档。这不是微调 BERT 或复现论文指标；参考图与实际 T3 产物分开展示。

## 6. 怎么运行

```bash
paperleet test retrieval/late_interaction --impl problems/retrieval/late_interaction/solution.py
```

浏览器模式在 CPU/WASM 执行完整原测试：T0 检查契约，T1 手算对照，T2 检查独立数值、有限差分梯度、mask、平局与稳定排序损失，T3 使用你的函数进行训练和检索。

## 7. 延伸

在“检索与推荐”集合中比较 CLIP 的单向量对比匹配、ColBERT 的多向量晚交互、TIGER 的生成式语义 ID。它们是不同的检索机制，本平台只复现选定的教学核心。
