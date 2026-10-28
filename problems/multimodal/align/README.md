# ALIGN 双向温度对比目标

## 1. 问题

实现 `align_loss(images: Tensor, texts: Tensor, temperature: Tensor) -> Tensor`。

归一化双编码器特征、可微温度与双向 batch softmax；以合成图文配对演示训练，不包含含噪网络数据采集。

## 2. 核心公式

$$
L=-\frac{1}{2B}\sum_i\left[\log\operatorname{softmax}_{j}(S_{ij})_i+\log\operatorname{softmax}_{j}(S_{ji})_i\right],\quad S_{ij}=\frac{\hat x_i^T\hat y_j}{\tau}.
$$

## 3. 公式到代码

先将每个样本投到单位球，再构造整个 batch 的交叉相似度。行 softmax 表示图找文，列 softmax 表示文找图；两者分母不同，不能仅把一个方向乘二。正样本是对角线。温度在除法内，其梯度由两向检索概率共同决定；只在校验时读取标量，计算时保留张量。

## 4. 本题约定

images、texts 为非空 [B,D] 浮点张量；temperature 为正的标量张量。同索引为正对，零向量归一化为零，范数下限 1e-12；两个检索方向各取 batch 均值，再取平均。所有三个输入保留梯度。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/align --impl problems/multimodal/align/solution.py
paperleet mutate multimodal/align
paperleet vizdata multimodal/align
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 3: normalized embeddings and symmetric image-text contrastive loss

- [https://arxiv.org/abs/2102.05918](https://arxiv.org/abs/2102.05918)
