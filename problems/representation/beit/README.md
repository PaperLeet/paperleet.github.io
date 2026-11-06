# BEiT：掩码离散视觉 Token 预测

## 1. 问题

[BEiT](https://arxiv.org/abs/2106.08254) 将掩码图像建模转为离散视觉 token 分类。图像 tokenizer 提供目标 ID；遮挡图像编码器预测移除 patch 对应的 ID。本题从给定 ID 与 patch logits 出发，提取核心预测目标。

实现 `masked_token_cross_entropy(logits, token_ids, mask)`。输入依次是有限 `[B,N,K]` 词表分数、int64 `[B,N]` 目标与 Boolean `[B,N]` mask。`True` 选中一个需要监督的遮挡位置。返回标量平均损失。

## 2. 核心公式

对选中的 patch $(b,n)$，分数 $z_{bnk}$ 预测类别 $y_{bn}$：

$$
p_{bnk}=\frac{\exp z_{bnk}}{\sum_j\exp z_{bnj}},\qquad
\ell_{bn}=\log\sum_k\exp z_{bnk}-z_{bn,y_{bn}}.
$$

令 $M=\sum_{bn}m_{bn}$，损失为

$$
\mathcal L=\begin{cases}\frac1M\sum_{bn:m_{bn}=1}\ell_{bn},&M>0,\\0,&M=0.\end{cases}
$$

使用稳定的 log-sum-exp 或 log-softmax。遮挡位置的 logit 梯度为 $(p_{bnk}-\mathbf1[k=y_{bn}])/M$；可见位置梯度为零。均匀 logits 的损失为 `log(K)`，单类别词表损失为零。

[官方训练代码](https://github.com/microsoft/unilm/blob/master/beit/engine_for_pretraining.py) 在 `no_grad` 下获取 ID，选出遮挡位置标签，再对遮挡 logits 计算均值 `CrossEntropyLoss`。本题额外定义空 mask 的可微零结果，并允许可见目标槽位使用 sentinel。

## 3. 公式到代码

1. 保留输入校验；只有遮挡位置的 ID 必须是合法词表索引。
2. 选择遮挡位置的 logits 行和对应 ID。选择为空时返回与输入 logits 保持连接的标量零。
3. 计算每个标签对应的分数，以及稳定的词表 log partition。
4. 对全部选中 token 的负对数概率求均值。

不要把概率当作 logits 再算交叉熵，不要除以 batch 大小，也不要平均每张图的均值。选中分数与归一化项均须保留梯度。

## 4. 本题约定

- `B,N,K>=1`；logits 为值有限的 float32/float64。ID 为 int64，mask 为 Boolean，设备和 `[B,N]` 轴与 logits 对齐。保持 dtype/device，不修改输入，支持非连续视图。
- 遮挡 ID 位于 `[0,K)`；未选位置 ID 可为任意 sentinel，不能参与 gather。
- mask 标记**需要预测**的位置。可见分数不得影响损失或梯度。
- 空 mask 返回可微标量零，`K=1` 合法。离散 ID 与 mask 不求梯度。
- 每个 patch 的词表分数加独立常数不改变损失；一致的类别重编号、重复数据也不改变均值。

独立测试覆盖标量 log-sum-exp、解析 softmax 梯度及有限差分、不等遮挡数量、sentinel、分数平移、词表反转、空/全 mask 和量级 1000 的 logits。

## 5. 交互图与 CPU 实验

Lab 训练一个有 16 个参数的**共享四分类线性预测头**。12 个合成 one-hot context 向量组成三张图，其中六个位置被遮挡、覆盖四个类别，遮挡数量分别为 `1,3,2`。Adam 使用你的函数，以 `.08` 学习率进行 **100 次真实更新**。独立诊断每十步用 Python 标量重新计算 CE 和遮挡位置准确率。

参考 CPU 的 CE 从 `log(4)≈1.3863` 降到约 `.01292`，准确率达到 `1.0`。T3 要求最终 CE 小于 `.03` 且小于初值的 `.03`、准确率为 `1.0`、提交损失与 oracle 差异小于 `1e-12`。图表展示实际优化与训练集准确率。one-hot context 使该实验成为有界且可学习的目标演示；它不训练 tokenizer、图像编码器或 BEiT 完整预训练系统。

## 6. 运行方式

补全函数后选择 **Run tests**。全部数值、梯度与训练检查支持 CPU/WASM，无需外部模型或数据。

```bash
paperleet test representation/beit --impl problems/representation/beit/solution.py
paperleet mutate representation/beit
paperleet ci --problem representation/beit
paperleet vizdata representation/beit
```

## 7. 延伸阅读

对照 BEiT 第 2 节与官方 `engine_for_pretraining.py`。视觉 tokenizer 目标与原始像素有何差异？为什么标签来自未遮挡图像？当 tokenizer 将不同外观 patch 归为同一类别时，学习目标怎样改变？

比较重建像素的 [MAE](https://arxiv.org/abs/2111.06377)，以及使用中心化教师软目标的 [iBOT](https://arxiv.org/abs/2111.07832)。解释 iBOT patch 分支提供的归约为何不同于本题的全局 token 均值。
