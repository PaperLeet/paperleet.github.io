# SwAV：平衡分配与交换预测

## 1. 题目

实现 `sinkhorn_assignments(scores, epsilon=.05, iterations=3)` 和 `swapped_prediction_loss(scores1, scores2, assignments1, assignments2, temperature=.1)`。先将原型得分转成平衡的软分配，再用一个视图的得分预测另一个视图的分配。

来源为 [SwAV](https://arxiv.org/abs/2006.09882) 第 3 节。本题取单进程、两个视图的目标；输入已是原型得分，不要求图像编码器、分布式同步、队列或原型参数归一化。

## 2. 核心公式

得分形状为 `[B,K]`。先建立 `K×B` 矩阵并归一化总质量：

$$
Q=\exp(S^\top/\epsilon),\qquad Q\leftarrow Q/\sum_{ki}Q_{ki}.
$$

严格执行 `iterations` 轮，先归一化原型行，再归一化样本列：

$$
Q_{ki}\leftarrow\frac{Q_{ki}}{\sum_jQ_{kj}}\frac1K,
\qquad Q_{ki}\leftarrow\frac{Q_{ki}}{\sum_lQ_{li}}\frac1B.
$$

返回停止梯度的 `A=(BQ).T`。预测概率为 `P_v=softmax(S_v/temperature)`，损失为

$$
L=-\frac1{2B}\sum_{i,k}\left[\mathrm{sg}(A_{2,ik})\log P_{1,ik}
+\mathrm{sg}(A_{1,ik})\log P_{2,ik}\right].
$$

分配的 epsilon 与预测温度是两个独立参数。

## 3. 从公式到实现

请在对数域用 `logsumexp` 执行 Sinkhorn，避免指数溢出。按顺序减去对应轴的对数和以及 `log(K)` 或 `log(B)`，最后加 `log(B)` 再取指数。返回结果必须符合指定轮数的迭代，而非某个未指定迭代次数的完全收敛解。

预测部分使用稳定的 `log_softmax`，交叉配对两个视图，并平均两项交叉熵。两个函数都需阻断分配的梯度，包括调用者直接传入损失函数的分配。

## 4. 约定与边界

输入为有限、非空的 `[B,K]` float32/float64 张量。损失函数四个张量的形状、dtype、device 一致，可不连续，且不得原地修改。分配非负，每行和为一，允许 `1e-5` 的误差。损失为标量，梯度只流入得分。

`epsilon` 和 `temperature` 是有限正 Python 数字，`iterations` 是正 Python 整数；不接受 bool 或 Tensor 标量控制参数。校验辅助函数已提供。

有限轮迭代后，每个样本的分配行和为一；各原型列和随收敛趋近 `B/K`，有限轮数不保证精确相等。`B` 可不等于 `K`，`K=1` 合法。测试覆盖有限轮数的独立标量结果、两种边际、排列与全局平移不变性、极端得分、解析得分梯度和分配停止梯度。

## 5. 交互图与 CPU 小实验

两组固定三角函数生成的 `6×3` 证据含共同原型偏置。以 epsilon `.2`、40 轮 Sinkhorn 产生分配，再训练两个独立 `6×3` 得分表：80 步 Adam，学习率 `.04`，预测温度 `.3`。共 36 个可学习参数，初始得分取另一视图证据的相反数。每八步记录损失与分配熵，并绘制各原型固定的分配质量。

实验实际调用你实现的两个函数，并以独立标量运算检查分配与每步损失。通过条件：最终交叉熵减去分配熵下界小于 `.002`，损失下降大于 `.1`，标量误差小于 `1e-11`，原型不平衡小于 `1e-8`。CPU float64，种子 42。这个有界实验展示平衡目标与交换预测，不评估图像表征能力。

## 6. 运行方式

```bash
paperleet test representation/swav --impl problems/representation/swav/solution.py
paperleet mutate representation/swav
```

将 `--impl` 指向你完成的 starter。浏览器运行完整 CPU/WASM 测试，无需图像数据、GPU 或预训练权重。

## 7. 延伸阅读

比较分配 epsilon 与预测温度的作用。观察一轮 Sinkhorn：最后的列归一化确保每个样本得到概率分布，但会再次扰动原型边际。交换视图决定哪个得分预测哪个目标；停止分配梯度则避免梯度经聚类过程回流。

来源：[论文](https://arxiv.org/abs/2006.09882)、[官方训练循环与分布式 Sinkhorn](https://github.com/facebookresearch/swav/blob/main/main_swav.py)。本题将算法限定为单进程、双视图；测试和向量实验独立编写。
