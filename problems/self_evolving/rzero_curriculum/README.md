# 108. R-Zero 伪标签与课程

## 1. 问题

将历史回答投票转换为固定伪标签，只保留信息量适中的题，再为新回答构造训练奖励。

实现 `consensus`, `curriculum_mask`, `pseudo_rewards`。

## 2. 核心公式

$$
\tilde y_i=\arg\max_c n_{ic},\quad p_i=\max_c n_{ic}/K,\quad m_i=\mathbf1[\ell\le p_i\le u],\quad r_{ij}=\mathbf1[a_{ij}=\tilde y_i].
$$

## 3. 公式到代码

对每道题沿回答轴计数。投票 [1,1,0,0] 平票时按本题约定取 0，一致率 0.5。用旧标签 1 评价新回答 [2,2,1] 时奖励为 [0,0,1]。

## 4. 本题约定

平票取最小 ID 是确定性教学约定。边界包含 0.25 和 0.75。新回答不能重新投票替换旧标签，否则训练目标会随当前回答漂移。答案等价判断由上游提供。

输入为有限 CPU 张量，不得原地修改。分数使用 float32/float64，ID 使用 long，掩码使用 bool。以下 API 明确形状、平票和梯度约定。

~~~python
def consensus(answers: Tensor, num_answers: int) -> tuple[Tensor, Tensor]:
    """answers: long [B,K], B,K>=1, IDs in [0,num_answers). Answer equivalence is supplied.
    Return (labels: long [B], consistency: float64 [B]) on the input device.
    Ties choose the smallest answer ID. Count ALL K answers. No mutation."""
~~~

~~~python
def curriculum_mask(consistency: Tensor, low: float = .25, high: float = .75) -> Tensor:
    """Float [B], 0<=low<=high<=1. Return bool [B], retaining BOTH boundary values."""
~~~

~~~python
def pseudo_rewards(answers: Tensor, labels: Tensor) -> Tensor:
    """New long answers [B,K], frozen pseudo-labels [B]. Return float64 [B,K].
    Match against the supplied OLD labels, never recompute a vote from new answers.
    These are agreement rewards, not independent correctness verification."""
~~~

## 5. 交互图与实验

在讲解页逐步播放论文流程，观察当前更新的模型、冻结的评价者、反馈和筛选。调整控件会重新计算教学分数。Lab 从均匀 logits 开始，用你的函数提供奖励或筛选样本，执行 60 次学习率 0.5 的真实类别策略梯度更新。使用固定反馈记录和 CPU float64；目标动作概率须大于 0.72，且提升超过 0.3。本实验验证局部机制，不代表完整 LLM/VLM 自演化或论文基准成绩。

## 6. 运行方式

补全 starter 后运行测试。T0 检查接口；T1 使用独立标量与计数推导；T2 检查边界、不变量和梯度归属；T3 将你的实现接入教学训练循环。

~~~bash
paperleet test self_evolving/rzero_curriculum --impl problems/self_evolving/rzero_curriculum/solution.py
paperleet mutate self_evolving/rzero_curriculum
~~~

## 7. 延伸

怎样发现一个一致率很高但答案错误的伪标签？

来源: [R-Zero: Self-Evolving Reasoning LLM from Zero Data](https://arxiv.org/abs/2508.05004), Sections 2.3–2.4: majority-vote labels, informative band and solver rewards.
