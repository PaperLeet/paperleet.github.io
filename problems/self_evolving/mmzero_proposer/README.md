# 112. MM-Zero 视觉课程奖励

## 1. 问题

为图像生成课程计算 Proposer 奖励：图片必须渲染成功，简单题提供可解性信号，困难题提供一致率信号，多样性调整防止单一模板占满训练池。

实现 `diversity_adjustment`, `proposal_rewards`。

## 2. 核心公式

$$
r=\frac1N\sum_j e_j[\min(s_j,0.5)+\min(c_j,1-c_j)]+\sum_k a_k.
$$

## 3. 公式到代码

对全部渲染尝试取平均，失败样本贡献零并保留在分母。两次尝试只有一次成功，成功项 easy=1、c=0.5，则基础奖励为 0.5。格式失败覆盖全部项，最终返回 -1。

## 4. 本题约定

简单题得分上限为 0.5；困难度项也至多 0.5。聚类多样性按 caption/easy/hard 权重 0.45/0.20/0.35 计算。M 是上游有效提案数，不一定等于当前张量行数。其余两项调整由调用者提供。

输入为有限 CPU 张量，不得原地修改。分数使用 float32/float64，ID 使用 long，掩码使用 bool。以下 API 明确形状、平票和梯度约定。

~~~python
def diversity_adjustment(shares: Tensor, valid_count: int, strength: float = .5) -> Tensor:
    """Float [B,3] caption/easy/hard cluster shares, valid_count>=1, strength>=0.
    Return detached [B]: -clip(dot(shares-1/M,[.45,.20,.35])*M*strength,
    -strength,+strength). M=valid_count is supplied, not necessarily B.
    Clustering and the valid population count are provided upstream."""
~~~

~~~python
def proposal_rewards(executed: Tensor, easy_accuracy: Tensor, consistency: Tensor, adjustments: Tensor, valid: Tensor) -> Tensor:
    """bool executed [B,N], float easy_accuracy/consistency [B,N] in [0,1],
    float adjustments [B,3] = easy-hard penalty, type penalty, diversity adjustment,
    bool format valid [B]. B,N>=1. Return detached same-dtype [B].
    Mean over ALL N render attempts of executed*(min(easy,.5)+min(c,1-c)),
    then add all three adjustments. Failed renders contribute zero but remain
    in the denominator. Invalid format overrides the ENTIRE result with -1.
    Adjustment construction (except diversity_adjustment) is supplied upstream."""
~~~

## 5. 交互图与实验

在讲解页逐步播放论文流程，观察当前更新的模型、冻结的评价者、反馈和筛选。调整控件会重新计算教学分数。Lab 从均匀 logits 开始，用你的函数提供奖励或筛选样本，执行 60 次学习率 0.5 的真实类别策略梯度更新。使用固定反馈记录和 CPU float64；目标动作概率须大于 0.72，且提升超过 0.3。本实验验证局部机制，不代表完整 LLM/VLM 自演化或论文基准成绩。

## 6. 运行方式

补全 starter 后运行测试。T0 检查接口；T1 使用独立标量与计数推导；T2 检查边界、不变量和梯度归属；T3 将你的实现接入教学训练循环。

~~~bash
paperleet test self_evolving/mmzero_proposer --impl problems/self_evolving/mmzero_proposer/solution.py
paperleet mutate self_evolving/mmzero_proposer
~~~

## 7. 延伸

为什么只在成功渲染的样本上取平均会掩盖 Coder 的失败？

来源: [MM-Zero: Self-Evolving Multi-Model Vision Language Models From Zero Data](https://arxiv.org/abs/2603.09206), Section 2.4, Eqs. 4–10: execution gating, score balance and diversity.
