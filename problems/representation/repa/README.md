# REPA 表征对齐损失

实现 [REPA](https://arxiv.org/abs/2410.06940)（ICLR 2025 Oral）将噪声扩散特征对齐到干净语义表征的损失。冻结的图像编码器提供目标，可训练投影层把扩散模型的中间 patch 映射到相同维度。

## 实现要求

### 返回值要求

#### `repa_alignment` (需实现)

返回一个标量浮点 `Tensor`，形状 `()`，保持特征 dtype/device。对 `[B,N,D]` 特征先归约 D 个通道，再在 B 个样本、N 个 patch 上平均负余弦相似度。保留 `projected` 梯度、冻结 `teacher`，不修改输入；范数不小于 eps 的 patch 完全一致时结果为 -1，不是零。

#### `repa_objective` (需实现)

返回一个标量浮点 `Tensor`，形状 `()`：所有元素上的平均速度 MSE，加上 `weight` 倍标量对齐损失。保持共同输入 dtype/device，保留 `prediction` 与 `projected` 梯度，冻结 `velocity_target` 和 `teacher`。只返回总损失，不返回分量 tuple，不修改输入。

### 公式与约定

输入 `p`、`z` 的形状都是 `[B,N,D]`：

$$
\bar p_{bn}=\frac{p_{bn}}{\max(\lVert p_{bn}\rVert_2,\epsilon)},\quad
\bar z_{bn}=\frac{\operatorname{sg}(z_{bn})}{\max(\lVert z_{bn}\rVert_2,\epsilon)},\qquad
L_{align}=-\frac1{BN}\sum_{b,n}\bar p_{bn}^{\mathsf T}\bar z_{bn}.
$$

实现 `repa_alignment(projected, teacher, eps=1e-8)` 和 `repa_objective(prediction, velocity_target, projected, teacher, weight=0.5, eps=1e-8)`：

$$L=\operatorname{mean}((v_{pred}-\operatorname{sg}(v_{target}))^2)+\text{weight}\,L_{align}.$$

完全同向时对齐损失是 **−1**。先沿最后一维分别归一化每个 patch，再对 batch 和 patch 取均值。两个范数分别截到 `eps`；全零教师 patch 的贡献是零。即使目标传入时设置了 `requires_grad=True`，教师和速度目标也必须停止梯度，学生特征和速度预测则保留梯度。

## 接口约定

特征形状相同且非空，为 `[B,N,D]`。两个速度张量形状相同且非空，不要求与特征形状相同。输入为有限的 CPU float32/float64，可为非连续视图；保留 dtype、device 和输入值。脚手架提供有限正数 `eps`、有限非负 `weight` 及形状校验。不实现投影层、预训练编码器或完整扩散模型。

例如 `p=[[[1,0],[0,2]]]`、`z=[[[3,0],[0,-1]]]`，一个 patch 同向、一个反向，损失为 `0`。

## Toy 实验

**教学改编。** 使用固定的合成干净特征作为教师，给输入表征加少量噪声，用你的总损失优化一个 3×3 投影矩阵 80 步。图中展示实际平均余弦和提交实现返回的损失；余弦指标由独立标量计算得到。这是目标函数的小实验，不是原文的图像生成评测。

预览调用参考实现，T3 调用你的提交。其他测试覆盖标量公式、解析梯度、目标冻结及逐 patch 缩放不变性。[作者实现](https://github.com/sihyun-yu/REPA/blob/main/loss.py)。
