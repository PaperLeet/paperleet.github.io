# SASRec：因果历史与逐位置负采样目标

预测用户下一次点击时，训练输入常包含一整段历史，输出是向后错开一位的物品。当前位置只能读取自己和过去，否则会直接看到待预测的未来信息。

本题选取 [SASRec](https://arxiv.org/abs/1808.09781v1) 的 III-B 因果约束与 III-E 训练目标。只实现两个函数，注意力编码器和优化循环已提供。

## 1. 历史可见性

`causal_item_mask(item_ids)` 接收 `[B,T]` 的非负 int64 物品 ID，0 表示 padding。返回 `[B,T,T]` 布尔张量：

$$M_{b,i,j}=[s_{b,i}\ne0]\,[s_{b,j}\ne0]\,[j\le i].$$

`True` 表示 query i 可以读取 key j。当前位置自己的物品已被观察，因此对角线应保留；预测目标是下一物品。Padding query 和 key 都不可参与。可以有左、右或内部 padding，全部 padding 的行全为 False。

例如 `[0,4,2]` 的三行可见性为：

```text
0 0 0
0 1 0
0 1 1
```

不要把它误写成排除自己的严格下三角，也不要只隐藏未来而留下 padding key。

## 2. 逐位置二元目标

### 返回值要求

#### `causal_item_mask` (需实现)

返回一个 Boolean `Tensor` `[B,T,T]`，device 与 item ID 一致，三个轴依次为 batch、query 位置 i、key 位置 j。True 表示两个 ID 都非 padding 且 j<=i，包含对角线。支持空轴。返回可见性，而非 attention 权重或加性浮点 mask；无梯度、不修改输入。

#### `sampled_next_item_loss` (需实现)

返回一个标量浮点 `Tensor`，形状 `()`，保持 logits 的 dtype/device。每个有效目标位置先相加两个二元损失项，再仅按非零 target ID 的位置数取平均，不额外除以二。空目标或全 padding 时返回同时连接两个 logits 输入的可微零值，梯度为零。保留有效 logits 两侧梯度，不修改输入。

#### `_check_ids` (已提供)

已提供校验函数：item_ids 是非负 int64 Tensor `[B,T]`，允许空维度时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

#### `_check_loss` (已提供)

已提供校验函数：非负 int64 target ID 与浮点 logits 的 `[B,T]` 形状、device 匹配，且有效目标位置的 logits 有限；忽略位置可包含 NaN/Inf时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

### 公式与约定

`sampled_next_item_loss(positive, negative, target_ids)` 的三个输入均为 `[B,T]`；前两者是相同 float32/float64 精度的**未归一化 logits**，目标 ID 为 int64。正负分数已由脚手架取出，无需生成负样本或再做 softmax。

令 $V=\{(b,t):o_{b,t}\ne0\}$：

$$L=\frac{1}{\max(1,|V|)}\sum_{(b,t)\in V}\left[\operatorname{softplus}(-r^+_{b,t})+\operatorname{softplus}(r^-_{b,t})\right].$$

这两个二元项分别把正分数推高、负分数压低，不是只优化分差的 BPR／成对 softmax。分母是有效目标数，不能包括 padding，也不能额外除以 2。论文给出求和目标；这里与作者代码一样按有效位置取平均，使用数值稳定的 logits 形式，避免原始 sigmoid/log 加小常数的极值饱和。

`target_ids` 决定损失 mask，不能用 history 的位置数代替。有效位置必须为有限 logits；被忽略位置可以含 NaN/Inf，必须在进入损失运算前屏蔽。没有有效目标或空维度时返回可反向传播的零，正负输入梯度均为零。所有输入保留 dtype/device 且不能原地修改。`_check_ids`、`_check_loss` 已提供，保留 starter 中的校验调用即可。

## 3. 从公式到实现

- 将物品 ID 是否为零广播到 query 和 key 两个维度，再与 `j <= i` 合并。
- 训练目标中先屏蔽无效 logits，计算两个稳定二元项，只累加有效位置。
- 正、负分数都必须保留梯度。不要对整个目标或正分数调用 `detach()`。

T0 检查接口、精度、空输入与输入不被修改；T1 是可手算示例；T2 使用 Python 标量 oracle、中心差分、mask 逐元素检查、极大 logits 和未来信息隔离；T3 调用你的实现执行完整小型训练。

## 4. 实验与结果的含义

提供两条长度为 4 的历史 `[0,1,3,4]`、`[0,2,3,4]` 和一条全空历史。两条历史以相同物品 4 结尾，却分别要求预测 5 和 6。模型包含固定的物品向量、单头恒等 K/V、可学习的 Q 投影与读出、残差 LayerNorm 和小幅固定位置偏移。你的 mask 控制注意力，损失控制实际训练梯度。160 次 Adam 更新优化 72 个参数。

图表展示逐次训练损失，以及第一条历史每个 query 对四个 key 的注意力权重。损失曲线的负样本随步数轮换，所以不保证每一步单调。只看当前物品和位置的任何预测器，在这些训练位置上的准确率最多为 5/6；实验要求利用更早历史超过这个严格上界。该上界通过按当前物品与位置分组取多数目标计算，并非另一模型的测量结果。指标另检查标量目标误差、修改未来物品后的过去状态变化和全空行注意力。

这是 **PaperLeet 教学改编**：轮换负样本只排除当前正目标，可能出现在历史中，与作者的历史外随机负采样不同；没有 FFN、dropout、完整可学习 embedding／投影栈或真实用户数据。准确率只在这些训练位置上计算，不是 held-out Recall/NDCG，更不能视为论文基准复现。完整 SASRec 还包括更多可学习层、训练数据和评估协议。

## 5. 前置学习

可先完成 Attention 与 LoRA 的矩阵／梯度练习，再比较本题的历史可见性和训练目标。

## 6. 在网页运行

在右侧 Python 编辑器完成两个函数并运行全部测试。可直接使用浏览器 CPU/WASM，无需 GPU、账号、模型下载或外部 API。不支持的自定义 Python/PyTorch 用法仍可转服务端判题。

## 7. 来源与后续

- [论文 v1：III-B、III-E](https://arxiv.org/abs/1808.09781v1)
- [作者训练目标与 padding 处理](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/model.py)
- [作者因果与 query/key mask](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/modules.py)

之后可学习 TIGER 的物品语义 ID 与生成式召回，比较它和本题直接对物品打分的路径。
