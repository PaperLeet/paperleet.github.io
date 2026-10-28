# RT-2 动作词表约束解码

## 1. 问题

RT-2 动作词表约束解码：实现下列接口并通过数值、边界和实验检查。

实现机器人任务与视觉语言任务共享logits时的输出约束；动作token集合由调用者提供，不运行PaLI-X、PaLM-E或语义推理。

## 2. 核心公式

$$
p_{bv}=\frac{\mathbf1[v\in A_b]e^{z_{bv}}}{\sum_{u\in A_b}e^{z_{bu}}},\quad A_b=A\ \text{(robot)},\ A_b=V\ \text{(VLM)}.
$$

## 3. 公式到代码

- `action_probabilities(logits: Tensor, action_tokens: Tensor, is_robot: Tensor) -> Tensor`

先建立逐行合法集合，再将非法logit替换为负无穷并做稳定softmax。先softmax再置零会破坏归一化。对机器人行的非法logits，输出及梯度都为零。

## 4. 本题约定

logits为(B,V)，action_tokens为(V,)布尔且至少一项true，is_robot为(B,)布尔；机器人行仅允许动作token，其他行允许全部词表。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

`experiment.py` 在固定输入上调用提交的全部接口，记录学习或采样曲线与末端指标。实验逻辑和阈值见 T3，输出同样用于可视化。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/rt_2 --impl problems/robotics/rt_2/starter.py
paperleet mutate robotics/rt_2
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818) — Section 3.2 Output Constraint
- [Verified source](https://arxiv.org/html/2307.15818)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
