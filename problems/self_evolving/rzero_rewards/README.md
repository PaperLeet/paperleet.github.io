# 107. R-Zero 出题奖励

## 1. 问题

让出题者偏好 Solver 尚未稳定掌握的问题，同时惩罚重复模板。输入已完成答案归一化、题目聚类和格式检查。

实现 `uncertainty_reward`, `challenger_rewards`。

## 2. 核心公式

$$
r_i=v_i\max(0,1-2|p_i-1/2|-\lambda |C_i|/B).
$$

## 3. 公式到代码

先算三角形不确定性奖励，再按整个批次计算聚类占比，最后截断并应用格式门控。例如 p=0.5，四题中两题同类，奖励为 0.5；若格式失败则为 0。

## 4. 本题约定

一致率不是正确率；所有人给出同一个错误答案也会得到 p=1。格式无效的行仍属于本题规定的聚类分母。观测奖励必须脱离计算图。

输入为有限 CPU 张量，不得原地修改。分数使用 float32/float64，ID 使用 long，掩码使用 bool。以下 API 明确形状、平票和梯度约定。

~~~python
def uncertainty_reward(consistency: Tensor) -> Tensor:
    """consistency: float [B] in [0,1]. Return detached same-dtype [B]: 1-2*abs(p-.5)."""
~~~

~~~python
def challenger_rewards(consistency: Tensor, clusters: Tensor, valid: Tensor, strength: float = 1.) -> Tensor:
    """Float consistency [B], long cluster IDs [B], bool valid [B]; B>=1, strength>=0.
    Count clusters across ALL B proposals, including invalid ones. Return detached
    float [B]: valid * max(0, uncertainty - strength*cluster_size/B). No mutation.
    Clustering/format parsing is supplied upstream; cluster IDs need not be contiguous."""
~~~

## 5. 交互图与实验

在讲解页逐步播放论文流程，观察当前更新的模型、冻结的评价者、反馈和筛选。调整控件会重新计算教学分数。Lab 从均匀 logits 开始，用你的函数提供奖励或筛选样本，执行 60 次学习率 0.5 的真实类别策略梯度更新。使用固定反馈记录和 CPU float64；目标动作概率须大于 0.72，且提升超过 0.3。本实验验证局部机制，不代表完整 LLM/VLM 自演化或论文基准成绩。

## 6. 运行方式

补全 starter 后运行测试。T0 检查接口；T1 使用独立标量与计数推导；T2 检查边界、不变量和梯度归属；T3 将你的实现接入教学训练循环。

~~~bash
paperleet test self_evolving/rzero_rewards --impl problems/self_evolving/rzero_rewards/solution.py
paperleet mutate self_evolving/rzero_rewards
~~~

## 7. 延伸

为什么最难的题不一定最有训练价值？

来源: [R-Zero: Self-Evolving Reasoning LLM from Zero Data](https://arxiv.org/abs/2508.05004), Section 2.2: uncertainty, repetition and format gating.
