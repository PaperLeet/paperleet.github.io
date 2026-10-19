# I-JEPA 掩码潜在目标与 EMA

[I-JEPA](https://arxiv.org/abs/2301.08243) 根据上下文表示预测被遮挡区域的目标表示。本题实现预测过程周围的目标准备、预测损失与动量更新。目标特征和掩码已给定，编码器和预测器不在实现范围内。

## 实现内容

### 返回值要求

#### `masked_targets` (需实现)

返回已 detach 的浮点 `Tensor` `[K*C*B,M,D]`，首维依次按目标块、context、batch 样本排列。K 为 mask 数量，C 为 `num_context`，B 为 batch 大小，M 为每个 mask 的 token 数，D 为特征维度。保持特征 dtype/device，不向 teacher 特征反传，不修改输入。

#### `masked_prediction_loss` (需实现)

返回一个标量 `Tensor`，形状 `()`，保持 prediction/target 的共同 dtype/device；它是 beta=1 的 smooth-L1 在所有元素上的平均值。保留 prediction 梯度，在函数内冻结 target；不返回逐 token loss 或 Python float，不修改输入。

#### `ema_update` (需实现)

返回一个新的、已 detach 的 `Tensor`，形状、浮点 dtype 和 device 与 `target`、`online` 一致，内容为更新后的目标参数。两个输入都不接收梯度，也不原地修改。这个函数返回新值，不返回 `None`。

#### `_features` (已提供)

已提供校验函数：x 是有限、非空的 float32/float64 Tensor，且指定 ndim 时秩匹配时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

#### `_pair` (已提供)

已提供校验函数：两个有限、非空浮点 Tensor 的形状、dtype、device 一致时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

#### `_check_targets` (已提供)

已提供校验函数：features `[B,N,D]`、等长 int64 masks `[B,M]`、正整数 context 数和有限正 epsilon 符合目标接口约定时返回 Python `None`；被校验条件不满足时抛出 `ValueError`。不修改输入，也不返回变换后的 Tensor。

### 公式与约定

`masked_targets(features, masks, num_context=1, eps=1e-5)` 接收 `[B,N,D]` 教师特征，以及 `K` 个目标掩码，每个是整数 `[B,M]`。完成：

1. 对教师特征 detach，再对每个 token 的 **D 个特征**独立做 layer norm，不使用仿射参数；方差除以 D，分母为 `sqrt(variance + eps)`。
2. 按给定顺序，收集每个 batch 样本的目标 token。
3. 将每个目标块的完整 B 样本 batch 重复 `C=num_context` 次，再拼接目标块。结果为 `[K*C*B,M,D]`，顺序是**目标块、上下文、batch 样本**。

两个目标块、两个上下文、两张图时，首维顺序为 `(块0,上下文0,图0)`、`(块0,上下文0,图1)`、`(块0,上下文1,图0)`、`(块0,上下文1,图1)`，然后按相同顺序排列块1。若逐张图连续重复，会与预测器输出错位。

`masked_prediction_loss(prediction, target)` 返回 `beta=1` 的 smooth L1 均值：

$$
\ell(r)=\begin{cases}\tfrac12r^2,&|r|<1\\ |r|-\tfrac12,&|r|\geq1,\end{cases}\qquad
L=\operatorname{mean}\ell(prediction-\operatorname{sg}(target)).
$$

这里遵循**作者发布代码** `src/train.py` 的训练约定；论文文字描述的是平均 L2 预测距离。本题明确区分两者，判题要求 smooth L1。即使传入的目标张量需要梯度，也应在函数内部停止目标梯度；预测值保留梯度。残差 `[-2,0.5]` 的损失为 `(1.5+0.125)/2=0.8125`。

`ema_update(target, online, momentum)` 返回

$$\theta_{target}^{new}=m\operatorname{sg}(\theta_{target})+(1-m)\operatorname{sg}(\theta_{online}).$$

返回新的无梯度张量，不修改两个输入。这是作者原地参数更新的函数式版本。实际训练中，调用方在 optimizer 更新后，逐个编码器参数执行它；预测器不参与目标编码器的 EMA。

## 接口约定

输入为有限的 CPU float32/float64 张量，配对张量的 dtype/device 一致，不修改输入。特征为非空 `[B,N,D]`；掩码为同设备 int64，M 非零且各块相同，索引合法。上下文数是正整数，epsilon 有限且为正。损失和 EMA 的张量对形状相同且非空。momentum 有限且在 `[0,1]` 内。支持非连续输入。校验代码已提供。

## 小实验

**教学改编。** 两张合成的六 token 图像提供潜在特征，两个目标掩码和两个上下文检查顺序。一个小预测表通过你的损失学习所选目标；另一个固定 online 参数的 EMA 示例展示目标参数的趋近过程。用标量归一化、收集公式和 EMA 闭式结果独立校验输出。实验不训练图像编码器，也不复现 ImageNet 预训练。预览使用参考实现，T3 调用你提交的全部三个函数。

[作者训练代码](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/train.py)。
