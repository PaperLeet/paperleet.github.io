# 109. G-Zero 提示概率差

## 1. 问题

对同一段无提示回答，比较原问题与问题加提示两种上下文下的 token 概率；再加入提示长度和题目重复惩罚。

实现 `hint_delta`, `proposer_rewards`。

## 2. 核心公式

$$
\delta=\frac{\sum_t m_t(\log p_t-\log p_t^h)}{\max(1,\sum_t m_t)},\quad r=\delta-0.03\max(L-200,0)/100-s.
$$

## 3. 公式到代码

两次评分必须对齐到同一回答 token；提示本身不进入平均值。有效 token 的概率差为 [2,1] 时 δ=1.5。空行返回 0。无效格式用 -1 代替 δ 与长度项，但仍减去聚类占比。

## 4. 本题约定

δ 是提示造成的概率变化代理量，不是真实质量增益，也不是完整 KL 散度。不要比较两段不同回答的概率。长度是字符数；奖励不截断到非负。Hint-δ 可微，最终奖励固定。

输入为有限 CPU 张量，不得原地修改。分数使用 float32/float64，ID 使用 long，掩码使用 bool。以下 API 明确形状、平票和梯度约定。

~~~python
def hint_delta(base_logp: Tensor, hinted_logp: Tensor, mask: Tensor) -> Tensor:
    """Float log-probabilities [B,T], bool mask [B,T]; same UNASSISTED answer tokens
    scored under two aligned contexts. Return float [B], mean(base-hinted) over
    valid tokens. All-masked rows return 0. Finite inputs, no mutation.
    Differentiable in both log-prob tensors; masked positions have zero gradient."""
~~~

~~~python
def proposer_rewards(delta: Tensor, hint_lengths: Tensor, cluster_share: Tensor, valid: Tensor) -> Tensor:
    """Float delta/share [B], integer hint character lengths [B], bool valid [B].
    Valid reward: delta - .03*max(length-200,0)/100 - cluster_share.
    Invalid reward: -1 - cluster_share, ignoring delta and length. Return detached
    same-dtype [B]; negative rewards are retained. Clustering is supplied upstream."""
~~~

## 5. 交互图与实验

在讲解页逐步播放论文流程，观察当前更新的模型、冻结的评价者、反馈和筛选。调整控件会重新计算教学分数。Lab 从均匀 logits 开始，用你的函数提供奖励或筛选样本，执行 60 次学习率 0.5 的真实类别策略梯度更新。使用固定反馈记录和 CPU float64；目标动作概率须大于 0.72，且提升超过 0.3。本实验验证局部机制，不代表完整 LLM/VLM 自演化或论文基准成绩。

## 6. 运行方式

补全 starter 后运行测试。T0 检查接口；T1 使用独立标量与计数推导；T2 检查边界、不变量和梯度归属；T3 将你的实现接入教学训练循环。

~~~bash
paperleet test self_evolving/gzero_hint --impl problems/self_evolving/gzero_hint/solution.py
paperleet mutate self_evolving/gzero_hint
~~~

## 7. 延伸

如果提示直接泄露答案，较大 δ 为什么可能误导训练？

来源: [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://arxiv.org/abs/2605.09959), Section 3.1, Eq. 3; Section 3.2, Eq. 5.
