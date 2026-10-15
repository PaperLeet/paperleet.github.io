# Causal-JEPA 对象轨迹遮蔽

[Causal-JEPA](https://arxiv.org/html/2602.11389v2) 沿历史时间窗口遮蔽选中的对象轨迹，保留首帧作为身份锚点。预测器根据剩余上下文，恢复被遮蔽的历史并预测所有对象的未来状态。

本题实现仅含对象 slot 的输入组装与学习目标。冻结的对象表示、投影矩阵和嵌入参数均已给定，不需要训练图像编码器或 Transformer。

| 函数 | 输入 | 输出 |
|---|---|---|
| `object_mask_inputs(history, masked_slots, anchor_projection, time_embeddings, mask_token)` | `(B,H,S,D)`、bool `(B,S)`、`(D,D)`、`(T,D)`、`(D,)` | 预测器输入 `(B,T,S,D)` 与 bool 目标掩码 `(B,T,S)` |
| `masked_prediction_loss(predictions, targets, history_length, masked_slots)` | 两个 `(B,T,S,D)`、整数 H、bool `(B,S)` | `(total, history_loss, future_loss)` |

要求 `T>H>=1`。`masked_slots=True` 表示该对象在首帧之后的历史被隐藏。允许每个样本有不同选择，是官方代码共享对象选择的确定性教学扩展。

## 构造输入

### 返回值要求

#### `object_mask_inputs` (需实现)

按顺序返回 `(inputs, target_mask)`，恰好两个 Tensor：浮点 inputs `[B,T,S,D]` 与 Boolean target_mask `[B,T,S]`。B 为 batch 大小，T 为历史加未来总帧数，S 为对象 slot 数，D 为潜在维度；历史 H 帧满足 T>H>=1。True 标记预测目标：选中对象的非首帧历史及所有未来 slot，首帧永不作为目标。保持 history dtype/device，并保留可见历史、身份锚点与参数梯度；mask 无梯度。不修改输入。

#### `masked_prediction_loss` (需实现)

按顺序返回 `(total, history_loss, future_loss)`，三项都是标量浮点 `Tensor`，形状 `()`，保持 prediction dtype/device。历史和未来分别在各自选中的所有元素上平均 MSE；total 是两项之和，不是平均。未选中非首帧历史时，history_loss 是可微零值。仅保留监督位置的 prediction 梯度，函数内部冻结 targets。不修改输入。

#### `_float_tensor` (已提供)

已提供校验函数：x 是秩为 ndim 的有限、非空 float32/float64 Tensor时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

#### `_mask` (已提供)

已提供校验函数：masked_slots 是形状恰为 `[batch,slots]`、位于指定 device 的 Boolean Tensor时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

### 公式与约定

时间零的**所有对象都可见**。时间 `1...H-1` 只遮蔽选中对象；时间 `H...T-1` 遮蔽所有对象。

可见位置 `(b,t,s)` 的输入为 `history[b,t,s] + time_embeddings[t]`。被遮蔽位置使用身份查询：

$$q_{b,t,s}=history_{b,0,s}W^T+time_t+mask\_token.$$

必须使用首帧，不能读取当前被隐藏的真实状态。官方 `id_projector` 含偏置；本接口将共享偏置合并进 `mask_token`，查询公式在代数上等价。时间嵌入在对象之间共享，不添加对象序号的位置编码。

即使查询值恰好与真实值相同，也必须依据结构返回 bool 目标掩码，而非比较张量数值是否不等。不要修改输入；输入组装应保留可见历史、锚点和参数的梯度。图像编码器的冻结由调用方处理。

## 联合遮蔽目标

在损失内部对 `targets` 停止梯度。按照论文公式 (5) 的遮蔽位置指示器，并分别归约历史和未来两项，计算：

1. 时间 `1...H-1` 中选中对象的平方误差，对选中 token 的所有特征元素取平均。
2. 时间 `H...T-1` 中所有对象的平方误差，对全部未来元素取平均。

返回两者之和及两个分项。不要再把两项取平均，也不要使用共同分母。没有历史目标时，该分项为可微零。首帧和可见对象历史不参与历史损失。本接口只处理对象，不包含动作或本体感知辅助 token。

发布的 CLEVRER 训练循环还对选中对象的可见首帧计算历史损失。本题遵循论文的遮蔽位置指示器，排除该首帧，因此不是该训练循环的逐项复刻。

所有浮点张量为类型设备一致的有限 float32/float64，starter 已提供检查。例如 `H=3,T=5`、两个对象中选中一个，选中对象的目标掩码为 `[0,1,1,1,1]`，另一对象为 `[0,0,0,1,1]`。

## Toy：恢复交互对象的轨迹

**教学改编。** 16 条合成双对象序列暴露一个对象的运动，隐藏另一个对象在首帧之后的轨迹。你的输入组装函数提供数据给线性探针，你的损失函数训练它恢复历史并预测未来。图表显示两项损失及恢复轨迹。独立检查将隐藏历史加上 100，验证预测器输入完全不变。

实验检验可观测信息与目标设计，不复现 CLEVRER/Push-T、不训练视觉模型，也不证明因果可识别性。T3 调用提交的实现，预览使用参考解。
