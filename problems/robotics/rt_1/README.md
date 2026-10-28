# RT-1 TokenLearner 空间汇聚

## 1. 问题

RT-1 TokenLearner 空间汇聚：实现下列接口并通过数值、边界和实验检查。

提取视觉token压缩器；MLP注意力logits作为输入，不训练EfficientNet、Transformer或真实机器人。

## 2. 核心公式

$$
Z_{bkc}=\sum_l\operatorname{softmax}_l(S_{blk})X_{blc}.
$$

## 3. 公式到代码

- `token_learner_pool(features: Tensor, logits: Tensor) -> Tensor`

将logits转成(B,K,L)，沿最后维归一化后乘原始features。每个学习token是一组空间凸组合，不能沿K归一化，也不应对被汇聚features再次做LayerNorm。

## 4. 本题约定

features为(B,L,C)，logits为(B,L,K)；只沿空间L做softmax。输出(B,K,C)，两输入均保留梯度。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

`experiment.py` 在固定输入上调用提交的全部接口，记录学习或采样曲线与末端指标。实验逻辑和阈值见 T3，输出同样用于可视化。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/rt_1 --impl problems/robotics/rt_1/starter.py
paperleet mutate robotics/rt_1
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817) — Official tokenizers/token_learner.py: spatial softmax and einsum over original features
- [Verified source](https://github.com/google-research/robotics_transformer/blob/master/tokenizers/token_learner.py)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
