# SigLIP 2 掩码 Patch 自蒸馏

## 1. 问题

实现 `masked_patch_distillation(student: Tensor, teacher_prob: Tensor, mask: Tensor, temperature: float) -> Tensor`。

实现 SigLIP 2 辅助掩码预测中的概率蒸馏核；teacher 概率已给定，不实现 EMA 编码器、增强、投影头或完整多目标训练。

## 2. 核心公式

$$
L=-\frac{\sum_{b,p}m_{bp}\sum_k\operatorname{stopgrad}(q_{bpk})\log\operatorname{softmax}(z_{bp}/\tau)_k}{\sum_{b,p}m_{bp}}.
$$

## 3. 公式到代码

teacher_prob 已是目标分布，不应再次 softmax。先停止 teacher 梯度，再计算每个 patch 的类别交叉熵，最后按 mask 求和并除以选中 patch 数量。未选中位置的学生梯度必须为零；选中位置的导数是 (softmax(student/tau)-teacher)/(M*tau)。这一目标抽取了论文辅助训练的一部分。

## 4. 本题约定

student 和 teacher_prob 为 [B,P,K]；teacher_prob 非负且最后一维和为 1，是已完成 teacher 投影和归一化的概率。mask 是 [B,P] bool，至少一个 True；temperature 为正 Python float。只对掩码 patch 的交叉熵取平均；teacher 停梯度，student 保留梯度。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/siglip_2 --impl problems/multimodal/siglip_2/solution.py
paperleet mutate multimodal/siglip_2
paperleet vizdata multimodal/siglip_2
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 2.3: masked prediction uses stopped teacher targets at masked patch locations

- [https://arxiv.org/abs/2502.14786](https://arxiv.org/abs/2502.14786)
