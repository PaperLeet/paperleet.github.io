# 38. SDPO 反馈选择与自蒸馏

按 Table 2 选择自身成功解、同题成功解或环境反馈，再执行自蒸馏。

## 论文与范围

[Reinforcement Learning via Self-Distillation](https://arxiv.org/html/2601.20802v2) · Section 2, Algorithm 1, and teacher prompt Table 2 · [Code](https://github.com/lasgroup/SDPO)

实现反馈路由和带有效 token 掩码的 reverse KL；微型程序和 categorical 反馈教师已提供。

## 接口

### 返回值要求

按下列结构返回，保留形状中的单例维度。标量 Tensor 的形状为 `()`，不是 Python `float`。

#### `select_feedback` · 需要实现

按顺序返回 `(solution_indices, feedback_available)`，两者形状均为 `[N]`，设备与 `question_ids` 一致（`N` 次尝试）：

- `solution_indices`：`torch.long` 行索引。成功尝试优先选自身，否则选择相同问题 ID 的首个成功尝试；不存在则为 `-1`。
- `feedback_available`：布尔 Tensor，有可用解答或该尝试有丰富反馈时为真。

两者都是无梯度的路由结果。丰富反馈可使 availability 为真，但对应的解答索引仍为 `-1`。

#### `feedback_reverse_kl` · 需要实现

返回与 student 同 dtype/device 的浮点标量 Tensor `loss`，形状 `()`。对 `[N, L, V]` logits 先沿词表 `V` 求反向 KL 和，再对 `token_mask & feedback_available[:, None]` 为真的 token 平均。保留 student 梯度，固定 teacher logits；无符合条件的反馈 token 时返回可反传的零，student 梯度为零。

### 公式与约定

```python
select_feedback(question_ids, success, rich_feedback)
feedback_reverse_kl(student_logits, teacher_logits, token_mask, feedback_available)
```

`select_feedback` 接收 int64 `question_ids[B]`、bool `success[B]`、bool `rich_feedback[B]`，B>0，返回 `(solution_index[B],available[B])`，依次采用以下优先级：

1. 原尝试成功：成功解索引为**自身**，available=True。
2. 原尝试失败：选择**同题**成功尝试，available=True。多个成功候选时选最小索引。
3. 没有成功解但有丰富环境反馈：索引 -1，available=True。
4. 两者都没有：索引 -1，available=False。

成功解优先于 rich feedback，遵循论文 Table 2；仅“多个同题成功解取最小索引”是本题确定性约定。`feedback_reverse_kl` 接收学生/教师有限 logits `[B,T,V]`、bool token mask `[B,T]` 与反馈可用标记 `[B]`。B,V 为正，T 可为零；CPU float32/float64，dtype/device 相同。返回所有可用 token 的平均 reverse KL；无有效 token 返回可微的 0。不要改写输入，仅学生可求梯度。

## 定义

在学生原有回答前缀上，教师额外读取反馈 f；损失是 `sum_v p_student*(log p_student-log p_teacher)`。有效位置同时满足 token_mask 和 feedback_available。成功解只进入教师上下文，不能替换原始学生轨迹；不同问题之间不能借用回答。即使上下文包含另一个成功解，教师仍评分原学生回答的前缀。

## 例子

questions=[7,8,7]，success=[False,True,True]，无 rich feedback：solution_index=[2,1,2]，available=[True,True,True]。单个失败尝试有 rich feedback 时返回 (-1,True)，否则返回 (-1,False)。

## 小实验

对加一、乘二、平方程序的六次尝试运行给定单测：两次成功尝试用自身成功解，两次失败尝试用同题成功解，一次失败尝试用丰富反馈，最后一次无反馈尝试应保持不变。已提供的 categorical teacher 输出 logits。这是教学缩小版，不是论文完整 coding benchmark。

T0 检查接口；T1 使用独立数值例子；T2 检查梯度与不变量；T3 在小实验中实际调用提交实现。
