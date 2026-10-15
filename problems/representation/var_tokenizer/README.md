# VAR 多尺度残差量化

实现 [VAR](https://arxiv.org/html/2404.02905v2) 使用的冻结潜空间 tokenizer：每个尺度量化前面尺度尚未解释的特征。算法 1–2 给出编码与重建过程，[官方代码](https://github.com/FoundationVision/VAR/blob/main/models/quant.py) 确定了 area 与 bicubic 插值约定。

## 需要实现

### 返回值要求

#### `multiscale_quantize` (需实现)

返回 Python tuple `(token_maps, reconstructions)`，两项都是按尺度排列、长度为 K=`len(scales)` 的 Python list。`token_maps[k]` 是 int64 `Tensor` `[B,n_k,n_k]`；`reconstructions[k]` 是已 detach 的浮点 `Tensor` `[B,C,H,H]`，表示截至第 k 个尺度的累计重建，而非单个尺度的贡献。B 为 batch 大小，C 为通道数，H 为最终尺度。保持特征 dtype/device，各重建快照必须独立存储；不修改输入、不反传。

#### `decode_multiscale` (需实现)

返回一个已 detach 的浮点 `Tensor` `[B,C,H,H]`，B 为 token map 的 batch 大小，C 为码本宽度，H 为最大 map 尺度。它是所有尺度提升后相加的结果，与编码器最终累计重建一致，不是 list 或尺度平均值。保持码本 dtype/device，不修改输入、不要求梯度。

#### `lift_codes` (已提供)

已提供 helper：由 int64 tokens `[B,n,n]`、码本 `[V,C]` 和通道投影 `[C,C]` 返回浮点 `Tensor` `[B,C,full_size,full_size]`。保持码本/投影 dtype/device。helper 自身不截断码本与投影的梯度；冻结的编码/解码调用方在 no_grad 下执行。不修改输入。

#### `resize_residual` (已提供)

已提供 helper：由 residual `[B,C,H,H]` 返回浮点 `Tensor` `[B,C,size,size]`，保持 dtype/device 和梯度路径。`size==H` 时直接返回原 Tensor（共享对象），不是副本或 `None`；否则返回缩放后的结果。不原地修改 residual。

#### `_validate_codes` (已提供)

已提供 helper：返回 Python `list[int]`，按输入顺序列出 K 个方形 token map 的尺度，不是 Tensor 或 `None`。校验非空 int64 maps `[B,n_k,n_k]`、从 1 开始严格递增的尺度、合法 code 索引、码本 `[V,C]` 与匹配投影 `[K,C,C]`；被校验条件失败时抛出 `ValueError`。不修改输入，输出无梯度。

### 公式与约定

`multiscale_quantize(features, codebook, scales, projections) -> (token_maps, reconstructions)`

初始化残差 `e=features` 和最大分辨率重建 `f_hat=0`。依次处理由小到大的尺度 `n_k`：

1. 调用已提供的 `resize_residual(e,n_k)`，逐空间位置在共享码本中寻找平方欧氏距离最小的向量。距离完全相同时选择最小索引。
2. 调用 `lift_codes(indices,codebook,projections[k],H)`，得到最大分辨率贡献 `h_k`。
3. 更新 `e=e−h_k`、`f_hat=f_hat+h_k`，保存整数 token 图和 `f_hat` 的独立快照。

返回 `K` 个 int64 图 `[B,n_k,n_k]`，以及 `K` 个累计浮点重建 `[B,C,H,H]`。最后一个尺度处理剩余的全分辨率残差，不能在每个尺度都重新量化原始特征。

`decode_multiscale(token_maps, codebook, projections) -> Tensor`

将每个 token 图放大、投影到最大分辨率后相加，返回 `[B,C,H,H]`。同一组 token 与投影的解码结果，必须等于编码时最后一个累计重建。

## 已提供 helper 与约定

`resize_residual` 使用 area 插值；`lift_codes` 完成码本查表、`align_corners=False` 的 bicubic 放大、通道矩阵乘法。二者已提供。本教学题用给定 `[K,C,C]` 矩阵替代原系统已学习的空间 `Phi` 层；码本选择不使用余弦归一化。

`features` 是有限、非空、正方形 `[B,C,H,H]`；共享码本为 `[V,C]`，`V,C>0`。浮点输入使用 CPU float32/float64，dtype/device 一致，允许非连续张量。尺度是从 `1` 到 `H` 严格递增的正整数，允许 `K=1,H=1`。解码 token 图遵守同样尺度约定，使用有效范围内的 int64 索引。不得修改输入，各尺度快照不得共享同一可变存储。

这里 tokenizer 已冻结，模板保留 `no_grad`。无需实现直通梯度、码本更新、commitment loss、编码器或图像解码器。

## 小实验

教学改编使用一张 8×8 潜特征图，包含粗粒度区域与小幅交替细节，配以九个标量码和恒等投影。你的编码、解码函数生成重建误差曲线与一行潜特征对照，检查细节恢复及编码/解码一致性。实验不假定任意码本、投影的每一个中间尺度都一定降低误差，也不声称复现 ImageNet 训练。
