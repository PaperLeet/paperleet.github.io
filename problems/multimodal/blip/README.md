# BLIP 动量软目标混合

## 1. 问题

实现 `momentum_target_loss(logits: Tensor, teacher_logits: Tensor, positives: Tensor, alpha: float) -> Tensor`。

实现 BLIP 一个检索方向的动量软标签蒸馏；支持非方形负样本队列，不实现队列更新、动量编码器或 CapFilt 数据生成。

## 2. 核心公式

$$
q_{ij}=\alpha\,\operatorname{stopgrad}(\operatorname{softmax}(s^m_i)_j)+(1-\alpha)\mathbf 1[j=p_i],\quad L=-\frac1B\sum_{i,j}q_{ij}\log\operatorname{softmax}(s_i)_j.
$$

## 3. 公式到代码

先在包含队列负例的列维度归一化 teacher，再混合软目标和明确指定的正样本 one-hot。每行目标的和仍为一；因此学生导数为 (softmax(logits)-targets)/B。alpha=0 退化为硬标签，alpha=1 仅蒸馏。动量教师的更新属于外部训练环节，不能让本损失向教师回传。

## 4. 本题约定

logits 和 teacher_logits 为 [B,N]，N 可含队列负例；positives 为 [B] int64 正样本列。alpha 在 [0,1]，目标为 alpha*teacher_softmax+(1-alpha)*one_hot。teacher 停梯度；对 B 行平均而非 B*N。logits 已完成温度缩放。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/blip --impl problems/multimodal/blip/solution.py
paperleet mutate multimodal/blip
paperleet vizdata multimodal/blip
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 3.1: image-text contrastive learning with momentum encoders and soft targets

- [https://arxiv.org/abs/2201.12086](https://arxiv.org/abs/2201.12086)
