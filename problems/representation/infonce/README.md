# InfoNCE：从候选中识别正样本

## 1. 问题

给定每个上下文对 `K` 个候选的打分，实现 `info_nce_loss(logits, positive_indices)`，使模型提高指定正样本的概率。输入已经是 **logits**，每行正样本的位置可以不同。

[Contrastive Predictive Coding（CPC）](https://arxiv.org/abs/1807.03748) 使用上下文预测未来表征，并通过正负候选构造训练信号。本题实现第 2.3 节式 (4) 的 InfoNCE 数值核心。没有人工类别标签；正例位置来自预测任务中已知的样本对应关系。

## 2. 核心公式

令 `s[b,j]` 为 logit、`p[b]` 为正样本列索引。每行包含一个正例和 `K−1` 个负例：

$$
\ell_b=-\log\frac{\exp(s_{b,p_b})}{\sum_{j=0}^{K-1}\exp(s_{b,j})},\qquad
L=\frac1B\sum_{b=0}^{B-1}\ell_b.
$$

正例必须同时出现在分子和分母中。用稳定形式计算：

$$
\ell_b=\operatorname{logsumexp}_j(s_{b,j})-s_{b,p_b},\qquad
\frac{\partial L}{\partial s_{b,j}}=\frac{\operatorname{softmax}(s_b)_j-\mathbf1[j=p_b]}B.
$$

例：一行候选的指数分数为 `[2,3,5]`，正例列为 `1`，则损失为 `log(10/3)`。所有候选同分时损失为 `log(K)`；`K=1` 时损失及梯度都是零。

## 3. 公式到代码

- 沿候选轴 `dim=1` 计算每行的 `logsumexp`，结果为 `[B]`。
- 根据 `positive_indices` 逐行取出正例 logit，结果为 `[B]`。
- 两者相减，再对全部上下文取均值，返回零维 Tensor。

不要先对原始大 logits 求指数。每行同时加一个常数不应改变损失；不同上下文之间不竞争同一个 softmax 分母。输入已经是分数，因此无需额外 L2 归一化、温度缩放或停止梯度。

## 4. 本题约定

`logits` 为有限 float32/float64 `[B,K]`，`B>=1,K>=1`；`positive_indices` 为同 device 的 int64 `[B]`，每个索引满足 `0<=p<K`。返回值保留浮点 dtype、device 与所有 logits 的梯度。支持非连续输入，不修改输入。提供的 `_check_inputs` 负责拒绝空轴、形状/dtype 不匹配、越界索引及 NaN/Inf。

梯度必须包含分母对全部候选的贡献。测试同时验证独立标量计算、闭式数值、解析梯度、有限差分、上下文及候选重排、逐行平移、大分数稳定性和真实优化行为。

## 5. 交互图与 CPU 实验

题目页展示真实优化轨迹：6 个上下文各有 4 个候选，直接训练一张 `6×4` 分数表，共 24 个参数。固定三角函数构造初始分数，正例列依次为 `[1,3,0,2,1,0]`；使用学习率 `1.5` 的 SGD，执行 80 次全 batch 更新，每 10 步记录损失、平均正例概率和严格 top-1 正确率（平局不算正确）。

这是原创教学实验，用于观察 InfoNCE 如何改变候选概率。它不包含 CPC 的编码器或自回归网络，图中指标是该分数表上的训练指标。T3 调用你实现的损失，要求最终六组全部识别正确、平均正例概率高于 `.9`、损失低于 `.1`，且记录值与独立标量公式一致。

## 6. 运行方式

网页点击 **Run tests**，在 CPU/WASM 上运行全部测试。也可以在仓库根目录运行：

```bash
paperleet test representation/infonce --impl problems/representation/infonce/solution.py
paperleet mutate representation/infonce
paperleet vizdata representation/infonce
```

练习时将 `--impl` 替换为你补全的 starter 路径。无需 GPU、数据下载或预训练权重。

## 7. 延伸与来源

继续练习 [SimCLR](#/representation/simclr)：它负责从两组视图构造 `2B` 个候选，排除锚点自身，再应用同一类正例分类目标。[MoCo](#/representation/moco) 则用动量编码器和队列提供候选。这些方法决定分数和候选从哪里来；本题先把损失本身写对。

论文：[CPC 原文](https://arxiv.org/html/1807.03748v2)，第 2.3 节式 (4)、(5)，作者 Aaron van den Oord、Yazhe Li、Oriol Vinyals，2018。原文采用正分数 `f`，本题输入 `s=log(f)`。论文中的互信息下界依赖其正负样本采样假设，不能把任意给定分数表的损失直接当作互信息估计。题面、验证、测试、实现和合成实验均为 PaperLeet 原创教学内容。
