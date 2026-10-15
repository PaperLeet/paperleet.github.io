# BPR：隐式反馈与个性化成对排序

同一件物品，有的用户喜欢，有的用户尚未接触。BPR 从已观察与未观察物品之间构造相对偏好，再学习用户专属排序。本题选择 [BPR 论文](https://proceedings.mlr.press/r7/rendle09a.html)第 3.2、4.1、4.3.1 节的三元组与矩阵分解核心。

## 1. 用户级负样本候选

`eligible_negatives(interactions, user_ids)` 接收 Boolean `[U,I]` 历史和 int64 `[B]` 用户 ID，返回 Boolean `[B,I]`：

$$M_{b,j}=\neg S_{u_b,j}.$$

`True` 表示这名用户**未观察**该物品，可以作为比较项；它不是已知负反馈标签。排除该用户历史中的全部物品，不能只排除当前正样本，也不能排除其他用户喜欢的物品。保持用户顺序和重复行。用户、物品编号从 0 开始，**没有 padding 编号**。空 batch、空物品表、全观察／完全未观察用户都需要支持。

## 2. BPR-MF 目标

`bpr_mf_loss(user_factors, item_factors, triples, l2=0.)` 接收浮点 `[U,D]` 用户表、`[I,D]` 物品表和 int64 `[B,3]` 三元组。每行是 `(用户, 正样本, 未观察物品)`；三元组已构造，不必在 loss 中重新采样。两张表共享 float32/float64 dtype、device，`D > 0`。先取出对应行，再计算：

$$
x_{uij}=w_u^\top(h_i-h_j),
$$

$$
L=\frac{1}{\max(1,B)}\sum_{(u,i,j)}\operatorname{softplus}(-x_{uij})
+\lambda\left(\sum_{u,d}W_{u,d}^2+\sum_{i,d}H_{i,d}^2\right).
$$

返回同 dtype/device 的标量。使用稳定的 log-sigmoid 或 softplus，不要先求 sigmoid 再直接取 log。损失基于**正负分数之差**，与 SASRec 的两个独立二元项不同。向量内积沿特征求和；对三元组取平均。保持正、负物品及用户的梯度，重复 ID 必须累积梯度。

## 3. 本题的先验约定

`l2` 是有限非负标量。惩罚两张**完整因子表**的平方和，包括这批三元组没有选中的行，每行只计一次，不额外乘 `1/2`。不能用 batch 中抽取向量的范数替代，否则重复 ID 和未抽中行都会改变目标。

`l2=0` 时应跳过先验计算，避免未选中大值因子的平方溢出后产生 `0*Inf=NaN`。数据项仍需保持用户和物品的完整梯度表。

原文 BPR-Opt 将 log-likelihood 求和；本题把数据项归一化为均值，并将先验系数作为对应尺度下的 `l2`。对非空 batch，和原文求和约定对照时 `l2 = lambda_paper / B`。空三元组的数据项定义为可微零，仍保留全表先验，两张表都应有正确的梯度图。保留输入值和非连续布局支持，使用 starter 已提供的检查。

## 4. 如何验证

T0 检查形状、精度、索引、空轴和输入保护；T1 使用手算例子；T2 用独立 Python 标量概率、有限差分、重复索引、未选中行梯度、batch 复制、公共平移及极端 margin 检查目标。支持全部默认测试后再看训练结果。

## 5. 个性化训练实验

三个合成用户分别喜欢物品 `{0,1}`、`{2,3}`、`{4,5}`。实验用你的候选 mask 构造 24 个训练三元组，用你的 loss 优化 27 个用户／物品因子，完整执行 120 次 Adam 更新。图中显示实际损失轨迹及各用户对六件物品的分数。

对任何两个不同兴趣组的物品，两个用户给出相反偏好。因此共享一套物品分数的预测器在这些三元组上的严格成对正确率至多为 `1/2`；这来自方向计数，不是跑出的模型基线。用户条件化的因子应突破这个界限，学会全部训练偏好。

这是 **PaperLeet 教学改编**：全 batch、单一全表先验、固定合成历史，**没有复现 LearnBPR 的 bootstrap SGD、按角色的多个正则系数或论文数据集**。Recall@2 和正确率都描述训练集，不是泛化成绩或 held-out 推荐指标。

## 6. 在浏览器中运行

补全两个函数并运行全部测试。CPU/WASM 无需 GPU、账号、模型权重、数据下载或外部 API；不支持的自定义 PyTorch API 仍保留服务端路径。

## 7. 来源与延伸

[论文原文](https://auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf)、[出版记录](https://proceedings.mlr.press/r7/rendle09a.html)、[arXiv 存档](https://arxiv.org/abs/1205.2618)。继续比较 SASRec 的逐位置二元目标、ColBERT 的 pairwise softmax，以及 TIGER 的语义 ID 生成检索。
