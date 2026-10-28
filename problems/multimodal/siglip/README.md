# SigLIP：图文成对 Sigmoid 对齐

## 1. 要解决的问题

一个 batch 含有 $B$ 对图像与文本特征。图像 $i$ 与文本 $i$ 匹配，其余配对不匹配。实现 [SigLIP](https://arxiv.org/abs/2303.15343) 的训练核心，让匹配对得分升高，让不匹配对得分降低。

本题提供验证和训练脚手架。你负责特征归一化、可学习的相似度标定，以及覆盖全部配对的稳定 sigmoid 损失。

## 2. 接口与返回值

```python
pairwise_logits(images, texts, log_scale, bias) -> Tensor
siglip_loss(logits) -> Tensor
```

`images`、`texts` 均为非空浮点张量 `(B,D)`。`log_scale`、`bias` 是同 dtype/device 的标量张量，不能转为 Python 数值或 detach。`pairwise_logits` 返回 `(B,B)`；`siglip_loss` 返回形状为 `()` 的标量张量。

令每行归一化特征为

$$
u_i=\frac{x_i}{\max(\lVert x_i\rVert_2,10^{-12})},\qquad
v_j=\frac{y_j}{\max(\lVert y_j\rVert_2,10^{-12})}.
$$

用可学习的 log scale $s$ 与加性偏置 $b$ 计算

$$
\ell_{ij}=\exp(s)u_i^\top v_j+b.
$$

采用 Algorithm 1 和官方代码的加性偏置约定。零向量归一化后仍为零。所有输入必须有限，且 `exp(log_scale)` 必须有限。

## 3. 损失与归约约定

对角线标签为 $z_{ii}=+1$，其他位置为 $z_{ij}=-1$。实现

$$
L=\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B}\operatorname{softplus}(-z_{ij}\ell_{ij}).
$$

全部 $B^2$ 个配对参与损失，最终除以 **$B$**。不要额外除以 $B$，也不要引入行或列 softmax。使用稳定的 softplus/log-sigmoid 形式，保证绝对值为 1000 的 logits 也有有限损失与梯度。

当 `logits=zeros(4,4)` 时，损失应为 $4\log 2$。当 $B=1$ 时，仍计算唯一正样本对的 logistic 损失。

## 4. 梯度与测试

保留两个特征塔、`log_scale` 和 `bias` 的梯度，不修改输入。测试包含标量余弦与 logistic 基准、梯度解析式、四类输入的有限差分、配对置换与特征缩放不变性。

T3 真实执行 120 次 CPU 梯度更新：优化四对合成特征向量及标定参数，观察检索准确率和正负配对分离。这是教学实验，不使用图像编码器、文本编码器或预训练数据。

## 5. 常见错误

- 对 $B^2$ 个配对取平均，损失和梯度缩小了 $B$ 倍。
- 仅保留正样本，遗漏负配对的监督。
- 对整个 batch 归一化，或使用未归一化的点积。
- 翻转偏置符号，或切断文本塔梯度。

## 6. 怎么运行

```bash
paperleet init multimodal/siglip
paperleet run multimodal/siglip
```

## 7. 来源

[论文](https://arxiv.org/abs/2303.15343)，Section 3.2、Algorithm 1；[官方训练代码](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py)。本题以单机完整配对矩阵实现数学目标，未实现跨设备分块通信。
