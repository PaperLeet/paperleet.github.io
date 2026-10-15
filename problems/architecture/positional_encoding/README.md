# 正弦位置编码

Transformer 没有循环或卷积来提供 token 的顺序信息。[Attention Is All You Need §3.5](https://arxiv.org/html/1706.03762v7#S3.SS5) 在 token embedding 上加上固定的正弦/余弦位置特征。本题实现位置编码；词嵌入、注意力和训练在其他模块完成。

## 你要实现的核心

### 返回值要求

#### `sinusoidal_encoding` — 需要实现

返回一个浮点 Tensor `[length,d_model]`，不添加批维度。第 i 行编码绝对位置 start+i，列为特征通道，偶数列为正弦、奇数列为余弦。遵循指定的 float32/float64 dtype 和 device；奇数宽度保留最后一个正弦通道，length=0 返回 `[0,d_model]`。固定编码的 requires_grad=False；返回 Tensor，不返回列表或两张表。

### 公式与约定

实现 `sinusoidal_encoding(length, d_model, start=0, *, dtype=torch.float32, device=None)`：

$$
PE_{p,2i}=\sin\left(p/10000^{2i/d_{\mathrm{model}}}\right),\qquad
PE_{p,2i+1}=\cos\left(p/10000^{2i/d_{\mathrm{model}}}\right).
$$

同一对 sine/cosine 共享频率。指数的分母是完整模型维度，不是通道对数。

| 接口 | 约定 |
|---|---|
| 输出 | `[length, d_model]`，位置从 `start` 到 `start + length - 1` |
| 维度 | 整数 `length >= 0`、`d_model > 0`、`start >= 0`；不满足时抛出 `ValueError` |
| 精度 | 显式指定 `torch.float32` 或 `torch.float64`；其他 dtype 抛出 `ValueError` |
| 设备 | 在指定设备上构造；测试使用 CPU |
| 奇数宽度 | 保留最后一个单独的 sine 通道，频率仍使用完整维度计算 |
| 空序列 | 返回 `[0, d_model]` 空张量 |
| 梯度 | 编码无可训练参数；加到 embedding 后仍保留 embedding 的梯度 |

奇数宽度、起始偏移和显式 dtype 是 PaperLeet 的接口扩展。原始模型使用偶数宽度 512。论文 §3.4 对 token embedding 乘以维度的平方根；不要把这个缩放应用到位置编码。本题不需要 dropout。

位置零、宽度五的结果是 `[0, 1, 0, 1, 0]`。对 `[B,L,D]` 的 embedding `x`，使用 `x + sinusoidal_encoding(L,D,dtype=x.dtype,device=x.device)`，即可在各 batch 共享同一组位置。

## Toy 实验

固定的相对位移对应每一对 `[sin, cos]` 的固定旋转：

$$
\sin(\theta+\delta)=\sin\theta\cos\delta+\cos\theta\sin\delta,\quad
\cos(\theta+\delta)=\cos\theta\cos\delta-\sin\theta\sin\delta.
$$

实验生成 32 个位置、八维特征，将位置整体移动七步，再将你的编码与独立计算的旋转结果比较。同时测量不同位置向量之间的最小距离。实验页展示实际曲线、旋转曲线和最大误差。

**来源：** 根据论文 §3.5 讨论的相对位移性质设计的 PaperLeet 教学实验。论文没有报告这个 32 位置实验。T3 调用你提交的实现，参考实现预览单独标注。

## 验证

T0 检查形状、精度、设备及空输出；T1 用独立标量公式核验大偏移与奇数宽度；T2 检查 sine/cosine 对的单位范数、分块位置、embedding 梯度及无效参数；T3 运行相对位移实验。典型错误包括频率指数少乘二、两个通道都用 sine、忽略偏移及丢失指定精度。
