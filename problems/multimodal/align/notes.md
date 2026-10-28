# ALIGN 双向温度对比目标 — 题解

先将每个样本投到单位球，再构造整个 batch 的交叉相似度。行 softmax 表示图找文，列 softmax 表示文找图；两者分母不同，不能仅把一个方向乘二。正样本是对角线。温度在除法内，其梯度由两向检索概率共同决定；只在校验时读取标量，计算时保留张量。

$$
L=-\frac{1}{2B}\sum_i\left[\log\operatorname{softmax}_{j}(S_{ij})_i+\log\operatorname{softmax}_{j}(S_{ji})_i\right],\quad S_{ij}=\frac{\hat x_i^T\hat y_j}{\tau}.
$$

参考实现按上述布局和归约逐步计算：

```python
def align_loss(images: Tensor, texts: Tensor, temperature: Tensor) -> Tensor:
    _floating(images, texts, temperature)
    if images.ndim != 2 or images.numel() == 0 or texts.shape != images.shape or temperature.ndim != 0 or temperature.item() <= 0:
        raise ValueError("expected matching nonempty (B,D) features and positive scalar temperature")
    x = F.normalize(images, dim=-1, eps=1e-12)
    y = F.normalize(texts, dim=-1, eps=1e-12)
    scores = x @ y.T / temperature
    return -(F.log_softmax(scores, dim=1).diag().mean() + F.log_softmax(scores, dim=0).diag().mean()) / 2
```

错误实现检测覆盖这些常见问题：

- Drops text-to-image contrast
- Omits image normalization

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

归一化双编码器特征、可微温度与双向 batch softmax；以合成图文配对演示训练，不包含含噪网络数据采集。

- [https://arxiv.org/abs/2102.05918](https://arxiv.org/abs/2102.05918)
