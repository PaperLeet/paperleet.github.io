# VAR 下一尺度预测

[VAR](https://arxiv.org/html/2404.02905v2) 每一步并行预测一个完整分辨率的 token map。本题实现冻结 tokenizer 的训练输入构造和注意力 mask。[官方代码](https://github.com/FoundationVision/VAR/blob/main/models/quant.py) 用**更低尺度** token 的累计重建，作为当前尺度的输入。

## 需要实现

### 返回值要求

#### `scale_causal_mask` — 需要实现

返回一个位于指定 device 的 `torch.bool` Tensor `[L,L]`，其中 `L=sum(n*n for n in scales)`。行为查询 token，列为键 token，按尺度递增、尺度内光栅顺序拼接。同一尺度及更早尺度的键为 True，包括尺度内所有配对。它是不带梯度的可见性矩阵，不是浮点加性掩码。

#### `next_scale_inputs` — 需要实现

返回一个浮点 Tensor `[B,L_next,C]`：B 为批大小，C 为 codebook 通道宽度，`L_next=sum(n*n for n in scales[1:])`。按尺度顺序拼接目标尺度 2..K 的特征，各尺度内部按行优先空间顺序排列；不包含单独提供的首尺度起始嵌入。保持 codebook 的 dtype/device；仅有一个尺度时返回 `[B,0,C]`。冻结 tokenizer 的结果必须 requires_grad=False，即使 codebook/projections 需要梯度也不例外，不修改输入。

#### `lift_codes` — 已提供

返回一个浮点 Tensor `[B,C,F,F]`；B 为批大小，C 为投影后的通道宽度，F=full_size。它是提升到完整空间分辨率并经通道投影的 codebook 特征，采用通道在前布局。保持 codebook/projection 的共同 dtype/device，不修改输入。此 helper 自身不切断 codebook/projection 梯度；next_scale_inputs 在 no-grad 上下文调用它。

#### `_validate_codes` — 已提供 helper

校验尺度序列和 Tensor 布局后，返回 Python `list[int]`，按输入顺序列出每个 token map 的正方形边长，不是 Tensor。受检输入不合法时抛出 ValueError，不修改输入。

### 公式与约定

`next_scale_inputs(token_maps, codebook, projections) -> Tensor`

输入是 `K` 个整数目标图 `[B,n_k,n_k]`。已提供 `lift_codes`，完成查码本、bicubic 放大、通道投影。记最大尺度 `H=n_K`，预测第 `k>0` 个尺度的输入为

$$F_{<k}=\sum_{j<k}\operatorname{lift\_codes}(r_j,Z,P_j,H),\qquad x_k=\operatorname{area}_{n_k}(F_{<k}).$$

将每个 `x_k` 按空间行优先顺序展开为 `[B,n_k²,C]`，再拼接 `k=1,…,K−1`，返回 `[B,Σ_{k>0}n_k²,C]`。第一尺度另有 class/start embedding，因此本函数不为它返回一行；`K=1` 时返回 `[B,0,C]`。最后一个目标图不会进入任何输入。

`scale_causal_mask(scales, device=None) -> Tensor`

返回布尔矩阵 `[L,L]`，其中 `L=Σn_k²`，`True` 表示**可见**。尺度 `i` 的 query 可读尺度 `j` 的 key，当且仅当 `j≤i`。同尺度内部所有位置互相可见，包括逆 raster 方向。[官方 mask](https://github.com/FoundationVision/VAR/blob/main/models/var.py) 采用这一规则。此处不会泄漏同尺度答案，因为这些输入位置装的是更低尺度重建，而不是自己的目标 token。

## 约定

尺度从 `1` 开始，均为正整数并严格递增；token 图均为正方形，共享 `B>0`。token dtype 为 int64，范围 `[0,V)`；共享码本为 `[V,C]`，投影为 `[K,C,C]`。浮点输入有限，使用 CPU float32/float64，dtype/device 一致，允许非连续张量。不得修改输入，应保持 dtype/device。这里 tokenizer 已冻结，模板中的 `no_grad` 应保留。

`lift_codes` 已给出：先用 `align_corners=False` 的 bicubic 放大到最大分辨率，再逐像素计算 `P_k @ feature`。原系统的 `Phi` 使用已学习的空间卷积；本题用固定通道矩阵缩小实现范围。位置编码、类别条件、码本训练和完整 Transformer 均不属于本题接口。

## 小实验

教学改编采用 1×1、2×2、4×4 目标图和一个固定注意力层，输入与 mask 均调用你的实现。改变某一目标图后，同尺度和更粗尺度的预测都应保持不变；更低尺度的信息应能影响更细尺度。另一张图比较 VAR 与 raster causal attention 的可见 key 数量。这是信息流实验，不是图像生成基准复现。
