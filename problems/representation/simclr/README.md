# SimCLR：NT-Xent 与同视图负样本

实现 `nt_xent_loss(view1, view2, temperature=.5)`。本题聚焦 [SimCLR](https://proceedings.mlr.press/v119/chen20j.html) 第 2.1 节式 1 与 Algorithm 1 的损失；输入已经是投影后的向量。

## 输入与配对

两组有限浮点输入均为 `[B,D]`，`B>=1, D>=1`，共享 float32/float64 dtype 与 device。保留输入 dtype、device 与两组输入的梯度，不改写输入，支持非连续视图。返回零维标量。提供的 `_check_inputs` 负责验证：温度必须是有限、正的 Python int/float，不能是 bool 或 Tensor；空轴、形状/dtype 不匹配和 NaN/Inf 均拒绝。

本练习**先拼全部 view1，再拼全部 view2**。编号 `i` 的正样本为 `i+B`（`i<B`），否则为 `i-B`。论文 Algorithm 1 使用交错编号；不能照搬相邻编号到本题。

## 目标与手算

逐行归一化采用本题明确的零向量约定：

$$
z_i=\frac{v_i}{\max(\lVert v_i\rVert_2,10^{-12})},\qquad s_{ij}=z_i^\top z_j/\tau.
$$

对每个锚点，排除自身，保留正样本及其余所有视图：

$$
\ell_i=\operatorname{logsumexp}_{j\ne i}(s_{ij})-s_{i,p(i)},\qquad
L=\frac{1}{2B}\sum_{i=0}^{2B-1}\ell_i.
$$

每行有 1 个正样本和 `2B-2` 个负样本，负样本包含同视图的其他实例。正样本**仍在分母中**。两方向一起平均；不额外乘 2。使用稳定的 logsumexp 或 cross entropy，避免直接计算指数后溢出。

例：`view1=view2=[[1,0],[0,1]]` 时，每行三个合法候选的余弦分数为 `1,0,0`，所以损失为 `log(1+2*exp(-1/temperature))`。若 `B=1`，只有正样本作为候选，损失与两组梯度均为 0。若所有非零行完全相同，损失为 `log(2B-1)`。

## 和 CLIP 比较

CLIP 题的每个图像只与文本候选比较；本题需要在全部 `2B` 个向量中比较。把已有的 `[B,B]` 跨视图对比损失直接复用，会遗漏同视图负样本。这里的温度是给定常数，未学习 logit scale。

测试包含独立 Python 标量分母、两组输入的有限差分、成对置换、交换视图、共同正交变换、逐行正缩放、零/近零向量和小温度稳定性。8 个错误实现分别检查归一化、自身掩码、候选集合、配对编号、方向、平均尺度、温度和梯度断开。

## CPU 小实验

已提供两张独立初始化的 `6×8` 向量表，共 96 个可训练数值。使用温度 `.2`、学习率 `.07` 的 Adam，实际执行 120 次全 batch 更新。每 10 步记录损失、正样本/最难负样本余弦和训练配对检索正确率；最后展示每个锚点的分离情况。

训练正确率严格要求正样本的分数高于所有其他合法候选；自身被排除，平局不算正确。记录值由实际更新计算，并用独立标量目标核对。这是 **PaperLeet 合成教学改编**：直接学习向量表，不涉及图像增强、编码器、投影头、真实数据或下游评估，不能据此声称复现论文性能。

无需 GPU、账号、数据下载或预训练权重，可在 CPU/WASM 浏览器判题；超出浏览器支持范围的 PyTorch 用法保留原生服务端判题。

## 来源与尺度

[论文 PDF](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf) 的 Algorithm 1 明确对 `2B` 个锚点取均值，本题遵循该尺度。[作者 objective.py](https://github.com/google-research/simclr/blob/master/objective.py) 将两个方向各自归约后相加；默认均值归约下是本题损失的两倍，比较日志或学习率时需先统一尺度。这里的 epsilon、输入契约、测试和小实验都是本题自行定义的实现约定。
