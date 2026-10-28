# π0.5 混合动作头防泄漏注意力

## 1. 问题

π0.5 混合动作头防泄漏注意力：实现下列接口并通过数值、边界和实验检查。

实现论文附录E/Figure18的训练注意力连接；prefix、离散FAST标签和连续动作expert的内容由调用者给定，省略模型预训练。

## 2. 核心公式

$$
M_{qk}=\begin{cases}k<P&q<P\\k<P\ \lor\ (P\le k\le q)&P\le q<P+F\\k<P\ \lor\ k\ge P+F&q\ge P+F\end{cases},\quad M\leftarrow M\land v_q\land v_k.
$$

## 3. 公式到代码

- `hybrid_attention_mask(valid: Tensor, prefix_length: int, fast_length: int) -> Tensor`

prefix仅看prefix；FAST看prefix与当前之前FAST；expert看prefix与所有expert。FAST包含干净动作标签，若允许expert读取，会在流匹配训练中泄漏答案。query与key两侧都需屏蔽padding。

## 4. 本题约定

valid为(B,L)布尔，prefix_length>=1，fast_length>=0且至少保留一个expert token。输出(B,L,L)布尔，行是query，列是key。FAST为已右移的teacher-forcing输入，可见自身；expert双向但不可见FAST。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

`experiment.py` 在固定输入上调用提交的全部接口，记录学习或采样曲线与末端指标。实验逻辑和阈值见 T3，输出同样用于可视化。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/pi0_5 --impl problems/robotics/pi0_5/starter.py
paperleet mutate robotics/pi0_5
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [π₀.₅: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054) — Appendix E, Figure 18: attention mask for discrete and continuous action training
- [Verified source](https://arxiv.org/html/2504.16054v1)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
