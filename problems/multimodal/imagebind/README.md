# ImageBind 图像锚点跨模态目标

## 1. 问题

实现 `image_anchor_loss(images: Tensor, modalities: Tensor, temperature: Tensor) -> Tensor`。

多模态共用图像锚点的配对图和归约；以合成模态演示无需非图像配对的对齐，不复现六种编码器。

## 2. 核心公式

$$
L=\frac1M\sum_{m=1}^M\frac{L_{I\to m}+L_{m\to I}}2,\quad S^{(m)}_{ij}=\hat I_i^T\hat X^{(m)}_j/\tau.
$$

## 3. 公式到代码

图像是配对图的公共中心，每个模态只与图像做对比。对一个模态构造 [B,B] 相似度，计算双向损失，然后对 M 个模态取平均。这样无需文本—音频等直接配对，也能通过同一图像身份组织表示。实验固定图像锚点，仅训练两个合成模态，并检查二者间的检索。

## 4. 本题约定

images=[B,D]，modalities=[M,B,D]，M/B/D 均正；每个模态都与相同图像 batch 对齐，但不同模态之间不直接加入损失。归一化后，各图像—模态对做双向 batch softmax，最终平均 M 个模态。temperature 为正标量。此教学核允许图像梯度；冻结图像编码器应由调用者处理。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/imagebind --impl problems/multimodal/imagebind/solution.py
paperleet mutate multimodal/imagebind
paperleet vizdata multimodal/imagebind
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 3 and Equation 1: binding modalities using image-paired contrastive learning

- [https://arxiv.org/abs/2305.05665](https://arxiv.org/abs/2305.05665)
