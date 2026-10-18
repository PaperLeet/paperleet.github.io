# GraphSAGE：采样邻域均值与归纳编码

## 1. 题目

实现 Hamilton、Ying、Leskovec 在 NIPS 2017 论文 [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) 中提出的均值聚合与同步编码。对应 Algorithm 1 第 4–7 行，以及 Section 3.3 的 mean aggregator。模型学习共享的特征与邻域变换，因此同一组权重可以编码训练时未见过的节点。

### 返回要求

`masked_neighbor_mean(features, neighbors, valid)` 返回 `[N,D]` 邻居均值。`graphsage_encode(features, neighbors, masks, weights, eps=1e-12)` 返回 `[N,D_last]` 表示；零跳时返回输入。保留 dtype、device、所有输入值，以及特征和每层权重的梯度。参数验证函数已经提供：保留调用，补全两个 TODO 函数体即可。

## 2. 核心公式

对节点 $v$，第 $k$ 跳的有效采样槽位为 $S_k(v)$，输入来自上一跳：

$$
m_v^k=\frac{1}{|S_k(v)|}\sum_{u\in S_k(v)}h_u^{k-1},\quad
z_v^k=\operatorname{ReLU}\left([h_v^{k-1}\Vert m_v^k]W_k\right),\quad
h_v^k=\frac{z_v^k}{\max(\|z_v^k\|_2,\varepsilon)}.
$$

采用行向量约定，权重形状为 `[2*Din,Dout]`。拼接顺序为**自身特征在前，邻居均值在后**。每一跳都先 ReLU，再沿输出特征轴逐节点做 L2 归一化。第 $k$ 跳的所有节点同步读取第 $k-1$ 跳的状态；逐节点原地更新会让结果依赖节点顺序。

## 3. 从公式到代码

每一跳提供独立的 `[N,S]` 邻居索引和同形布尔 mask。**先处理无效索引，再 gather**：被 mask 的槽位可以是 `99`、`-99` 等任意值。只累加有效槽位，除以有效数量，数量下限设为 1。重复 ID 按出现次数计入，因为这里表示采样槽位而非集合。空邻域的上下文为零，但自身分支仍然参与编码。

例如特征为 `[[1,2],[-3,4],[5,-2]]`，采样分别为 `[1,1,pad]`、`[0,2,pad]`、`[pad,pad,pad]`，均值应为 `[[-3,4],[3,0],[0,0]]`。不要自动添加自环或去重；只有显式采到自身 ID 时，自身才进入邻居分支。

## 4. 实现约定

| 输入 | 约定 |
| --- | --- |
| 特征 | 有限 float32/float64 `[N,D]`；`N>=0`、`D>0` |
| 采样 | list/tuple，每跳一个 int32/int64 `[N,S_k]`；有效 ID 在 `[0,N)` |
| 掩码 | 每跳一个同形布尔矩阵；无效槽位的索引完全忽略 |
| 权重 | 每跳一个有限 `[2*Din,Dout]` 矩阵；`Dout>0`，dtype/device 与特征一致 |
| 深度 | 采样、mask、权重长度相同；零跳是恒等映射 |
| Epsilon | 有限正标量，不接受 bool；分母为 `max(norm,eps)` |

支持非连续张量、零节点、零采样槽位与整行 padding。每次聚合和归一化都必须保持自动微分路径，且不能修改输入。

**范围说明：** 本题采用 Algorithm 1 的拼接结构、均值算子和逐跳 ReLU。论文 Equation (2) 是另一种将自身一起取均值的 GCN 变体。作者 TensorFlow 代码还包含可配置的投影、拼接和末层激活选择，本题不要求匹配整套实现。固定采样表、padding、空邻域与 epsilon 稳定化是明确的教学扩展。

## 5. 实验

训练两层共享编码器和线性分类器，共 80 个参数，执行 100 次真实 Adam 更新。所有被监督的查询节点特征都为零，标签平衡，需要读取锚点邻域才能区分；只看自身特征的分类器准确率最多为 50%。

随后冻结权重，在另一张图上评估：锚点特征、节点排列不同，节点数从训练图的 14 个变为 18 个。曲线展示实际训练损失、查询节点准确率，以及冻结模型在两张图上的准确率。提交后的实验调用你的编码器，参考预览调用提供的参考实现。

这是小型合成教学实验，不使用节点 ID 查表参数、原始基准数据、预训练模型、GPU 或 PyG；准确率不代表论文基准成绩。

## 6. 怎么运行

在网页编辑器中补全实现并运行全部测试，或使用：

```bash
paperleet test representation/graphsage_mean --impl problems/representation/graphsage_mean/solution.py
paperleet mutate representation/graphsage_mean
```

## 7. 延伸

T0 检查输入约定与空输入；T1 是手算样例；T2 检查独立标量基准、梯度、采样/节点重排与逐跳归一化；T3 在两张不同的合成图上训练和评估你的编码器。

为什么邻居顺序不影响结果，而采样重复次数会影响？把邻居均值 detach 后，前一层的梯度会如何改变？如果为每个训练节点单独学习一个向量，为什么不能直接完成同样的归纳迁移？
