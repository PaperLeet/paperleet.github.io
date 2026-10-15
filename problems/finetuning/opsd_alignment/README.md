# 36. OPSD 学生轨迹的双上下文对齐

对齐同一段学生回答在不同长度上下文下的 next-token logits。

## 论文与范围

[Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/html/2601.18734v3) · Section 3.2 and Algorithm 1 · [Code](https://github.com/siyan-zhao/OPSD)

实现因果位置选择、变长掩码和教师梯度隔离；logits 与 toy 训练循环已提供。

## 接口


### 返回值要求

#### `align_rollout_logits` — 需要实现

依次返回 Python 元组 `(student, teacher, mask)`。student/teacher 为浮点 Tensor `[B,T,V]`；B 为批大小，V 为词表大小，T=max(response_lengths)，分别保留来源 logits 的 dtype/device。mask 为同设备的布尔 Tensor `[B,T]`，有效回答位置为 True；填充位置 logits 为精确零。回答全部为空时返回 `[B,0,V]`、`[B,0,V]`、`[B,0]`。仅保留被选中学生位置的梯度，教师值 detach，不修改输入。

### 公式与约定
```python
align_rollout_logits(student_logits, teacher_logits, student_prompt_lengths, teacher_prompt_lengths, response_lengths)
```

`student_logits[B,Ls,V]` 与 `teacher_logits[B,Lt,V]` 是右侧补齐后的模型输出。两个上下文后面附加**完全相同的学生 rollout**；教师额外看到参考解。`student_prompt_lengths[B]`、`teacher_prompt_lengths[B]` 为正的 int64，`response_lengths[B]` 为非负 int64。所有输入在 CPU，batch 非空，dtype 为 float32/float64；每行序列长度足够容纳 prompt 和 response，填充 logits 有限。输出 `(student[B,T,V], teacher[B,T,V], mask[B,T])`，其中 `T=max(response_lengths)`；空回答位置置零。教师必须 detach，学生保留梯度。

## 定义

第 j 个回答 token 的预测来自 `prompt_length - 1 + j`，不是该 token 自身的位置。两个分支分别使用自己的 prompt 长度。

## 例子

学生 prompt 长 2，教师 prompt 长 5，回答长 3：分别选取 `[1,2,3]` 与 `[4,5,6]`；mask 为 `[True,True,True]`。

## 小实验

以三条不同长度的 categorical 轨迹训练学生 logits，观察真实回答位置的正确 token 概率。教学实验提供教师目标，考查位置对齐；没有调用 LLM，也不代表论文数学推理成绩。

T0 检查接口；T1 使用独立数值例子；T2 检查梯度与不变量；T3 在小实验中实际调用提交实现。
