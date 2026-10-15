# LightGCN：归一化二部图与多层协同传播

用户的历史可以构成二部图：用户连接自己观察过的物品，多个用户也会通过共同物品间接相连。本题实现 [LightGCN 原论文](https://arxiv.org/abs/2002.02126)的图构造和传播核心，并用一个小型合成实验观察训练过程。

## 1. 对称归一化二部图

`normalized_bipartite_graph(interactions, dtype=None)` 接收 Boolean `[U,I]` 历史，返回 `[N,N]` 浮点矩阵，`N=U+I`。节点顺序为全部用户，再接全部物品；编号从 0 开始，没有 padding。默认 float32，也接受显式 float64，并保留输入 device。

$$
A=\begin{bmatrix}0&R\\R^\top&0\end{bmatrix},\qquad
\widehat A=D^{-1/2}AD^{-1/2},\qquad D_{vv}=\sum_w A_{vw}.
$$

每条已观察边的权重是两端度数乘积的平方根的倒数。必须包含两个方向，不加自环。孤立节点的整行和整列保持 0；可以只为避免除零把度数下限设为 1。没有边、用户轴为空、物品轴为空、两轴均为空都合法。不要改写输入历史。

## 2. 线性传播与层聚合

### 返回值要求

#### `normalized_bipartite_graph` (需实现)

返回一个浮点 `Tensor` `[N,N]`；交互表为 `[U,I]` 时 N=U+I，两个轴都先用户、后物品。dtype=None 时使用 float32，否则使用指定的 float32/float64，device 与交互表一致。返回对称归一化邻接矩阵而非 embeddings：无自环，孤立行/列为零，支持空轴。不修改输入，Boolean 历史无梯度。

#### `lightgcn_embeddings` (需实现)

返回一个浮点 `Tensor` `[N,D]`，保持初始 embedding 的 dtype/device，为第 0 层到 K=`layers` 层共 K+1 项的均匀平均。深度为零时返回初始值，N=0 合法。保留初始 embeddings 与可微图矩阵的全部传播路径梯度，不只保留最后一层。不修改输入、不 detach 输出。

#### `_check_graph` (已提供)

已提供校验函数：interactions 是 Boolean `[U,I]`，dtype 明确为 float32 或 float64（调用方先把 None 转为 float32）时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

#### `_check_propagation` (已提供)

已提供校验函数：initial `[N,D]`（D>0）与 graph `[N,N]` 是有限且匹配的 float32/float64 Tensor，layers 是非负、非 Boolean 的 Python int时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

### 公式与约定

`lightgcn_embeddings(initial, normalized_graph, layers=2)` 接收 `[N,D]` 初始 embedding 和 `[N,N]` 图，`D>0`，两者共享 float32/float64 dtype 和 device。`layers` 是非负 Python 整数，Boolean 不算整数。

$$
E^{(0)}=E_0,\qquad E^{(k+1)}=\widehat A E^{(k)},\qquad
E=\frac{1}{K+1}\sum_{k=0}^K E^{(k)}.
$$

输出保留 `[N,D]` 形状、dtype、device。`K=0` 返回初始层的值；`N=0` 仍合法。每一步只有图乘法，不引入特征投影、激活函数或隐式自环。原文允许不同层权重，本题采用论文实验使用的均匀权重。

包含第 0 层，因此孤立节点的最终向量等于 `E0/(K+1)`，而不是零或原向量。保持初始 embedding 和图的自动微分路径，支持非连续输入且不改写输入。传播函数的检查只要求图为同 dtype、同 device 的有限方阵；测试也会用一般线性算子验证梯度。本实验中的图固定，**只学习 E0**。

## 3. 两跳的协同含义

一个用户的信号先传播到物品，再传给另一名观察过同一物品的用户。共享物品越热门，其贡献越小。测试用手算系数检查两跳传播，也用独立 Python 标量循环、有限差分、线性叠加和节点重编号验证实现，避免仅匹配参考实现的一条代码路径。

## 4. 实际训练实验

四名合成用户分别观察 `{0,1}`、`{1,2}`、`{3,4}` 和空集，物品 5 未被观察。前两名用户共享物品 1，用户 3 和物品 5 是孤立节点。初始表有 `10×3=30` 个可训练数值，使用两次传播并平均第 0、1、2 层。

已提供的循环枚举全部 24 个正样本／未观察物品三元组，完整执行 160 次 Adam 更新：

$$
L=\frac{1}{24}\sum_{(u,i,j)}\operatorname{softplus}\left(-e_u^\top(e_i-e_j)\right)
+0.001\lVert E_0\rVert_F^2.
$$

这里 `e` 是聚合后的向量，正则只作用于完整初始表 E0。原文式 15 的数据项求和，本题取均值；与原文求和尺度对照时，`l2=lambda_paper/24`。训练循环和稳定的 BPR 目标已提供，无需重新实现 BPR。

图表显示真实损失轨迹和三名活跃用户的物品分数。成对正确率及 Recall@2 都描述训练集；空历史用户不参与 Recall。按相反偏好的方向计数，任何只使用一套全局物品分数的预测器，严格成对正确率至多为 `17/24`；这是计数上界，不是实测模型基线。

这是 **PaperLeet 教学改编**：小型稠密图、固定合成历史、全 batch 训练、单一初始表先验；不复现大规模稀疏图系统、真实数据集或论文的 held-out Recall/NDCG。CPU/WASM 可直接在浏览器判题，无需 GPU、账号、权重或外部 API；不支持的自定义 PyTorch 用法保留原生判题路径。

## 5. 来源与延伸

[原文](https://arxiv.org/pdf/2002.02126)第 3.1.1–3.1.3 节、式 3–8 定义传播与层组合；第 3.2.3 节、式 13–14 分析两跳影响；第 3.3 节、式 15 给出训练目标。[作者代码](https://github.com/kuandeng/LightGCN)可用于了解真实稀疏图实现。先做 BPR 了解成对训练，再比较 SASRec 的顺序建模和 TIGER 的生成式召回。
