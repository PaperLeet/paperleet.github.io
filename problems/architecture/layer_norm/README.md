# LayerNorm 与残差连接

实现 [Attention Is All You Need §3.1](https://arxiv.org/abs/1706.03762) 中的归一化和残差连接。归一化方法来源于 [Layer Normalization](https://arxiv.org/abs/1607.06450)；在这里，它作为 Transformer 论文中的一个模块。

## 需要实现的核心

### 返回值要求

#### `layer_norm` — 需要实现

返回一个与 x 同形状 `[...,D]` 的浮点 Tensor；D 为末轴特征宽度，保留所有前导批/token 维度。返回的是仿射变换后的归一化特征，不是均值/方差统计量。保持 x 的 dtype/device 及 x、weight、bias 的梯度，不修改输入。

#### `post_norm_residual` — 需要实现

返回一个浮点 Tensor `[...,D]`，与 x、sublayer_output 同形状，保留所有前导维度，D 为特征宽度。其值为残差相加后的归一化结果。保持共同 dtype/device 及两条残差分支、weight、bias 的梯度，不原地更新分支。

### 公式与约定

使用张量归约实现 `layer_norm` 和 `post_norm_residual`。子层输出和实验脚手架已经提供。本题采用原始 Transformer 的 **post-LN** 顺序，关闭 dropout：

$$
\mu=\frac{1}{D}\sum_{i=1}^{D}x_i,\qquad
\sigma^2=\frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2,
$$
$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,
\qquad y=\operatorname{LN}(x+s).
$$

`s` 是已经算好的子层输出。统计量仅沿**最后的特征维**计算，每个 token、每个样本彼此独立。方差除以 `D`，因此若使用 `torch.var`，需要设置 `correction=0`。`epsilon` 放在平方根内部。

| 输入或输出 | 约定 |
|---|---|
| `x`、`s` | 相同形状 `[..., D]`，`D >= 1` |
| `weight`、`bias` | 形状 `[D]` 的可学习仿射参数，所有 token 共用 |
| `eps` | 正标量，默认 `1e-5` |
| 输出 | 形状、dtype、device 与 `x` 相同，不修改输入 |
| 输入范围 | 有限 float32/float64 张量，仿射参数与输入 dtype/device 一致 |
| 自动微分 | 保留到 `x`、`s`、`weight`、`bias` 的梯度 |

请通过张量运算自己计算统计量，不直接调用 `torch.nn.LayerNorm` 或 `torch.nn.functional.layer_norm`；测试会独立使用库算子作为对照。

## 示例与边界情况

`[1, 3, 5]` 的均值为 `3`，总体方差为 `8/3`。仿射参数取单位缩放、零偏移时，输出为 `[-2, 0, 2] / sqrt(8/3 + eps)`。一个 token 的特征全部相同，或 `D=1` 时，输出恰好等于仿射偏移 `bias`。

给同一个 token 的全部特征加上相同常数，不改变归一化结果。固定非零 epsilon 时，缩放不变性只是近似成立；若正比例缩放输入并把 epsilon 乘以缩放系数的平方，则在实数算术下严格成立。其他 token 或 batch 样本的变化不能影响当前 token。

## Toy 实验

实验提供 8 个有不同偏移与尺度的六维 token，加入固定子层输出后调用你的 `post_norm_residual`。图中展示归一化前后的逐 token 均值和总体标准差，独立的 Python 标量计算提供期望结果。单位仿射参数下，输出方差为 `variance / (variance + eps)`，因此不一定严格等于 1。扰动其他所有 token 则检验统计量的独立性。

**来源说明：**这是根据论文 §3.1 编写的 PaperLeet 教学改编，不是论文原始实验。实验关闭 dropout，不训练模型。T3 保存实际提交代码产生的测量值，参考预览单独标识。

## 验证与范围

T0 检查形状、dtype 和输入未被修改；T1 检查带非平凡仿射参数的手算结果；T2 检查常数/单特征输入、极小方差、token 独立性、平移/缩放恒等式、独立梯度对照与残差顺序；T3 运行上述实验。

原始论文先加残差，再进行 LayerNorm。Pre-LN 将归一化放在子层之前，接口和计算顺序不同。RMSNorm 不减均值，是另一种归一化算子，不是原始 Transformer 的 LayerNorm。本题覆盖子层输出已知时的 Add & Norm，不要求完整实现 encoder 或 decoder。
