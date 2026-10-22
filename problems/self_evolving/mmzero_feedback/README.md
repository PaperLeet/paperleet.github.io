# 113. MM-Zero 编码与解题反馈

## 1. 问题

分别计算 Coder 的执行与视觉反馈，以及 Solver 对银标签的一致性奖励。两个角色的奖励有不同的分量和端点。

实现 `coder_rewards`, `solver_rewards`。

## 2. 核心公式

$$
r_C=\mathbf1_e[1+s+\min(c,1-c)]-0.1\mathbf1_{\neg e}-0.05\mathbf1_{syntax},\quad r_S=\alpha\mathbf1[a=\tilde y]+(1-\alpha)v.
$$

## 3. 公式到代码

Coder 成功且 easy=0.8、c=0.5 时得 2.3；渲染与语法均失败得 -0.15。Solver 采用 α=0.9：答案同意但格式不合格仍得 0.9；不同意但格式合格得 0.1。

## 4. 本题约定

Coder 不使用 Proposer 的 easy 上限。失败渲染没有视觉分数，两类错误可叠加。论文 Eq.11 附近的 difficulty 二值范围注释与 Eq.7 不一致；本题采用 Eq.7 的连续 min(c,1-c)。Solver 格式项是加法而非门控。

输入为有限 CPU 张量，不得原地修改。分数使用 float32/float64，ID 使用 long，掩码使用 bool。以下 API 明确形状、平票和梯度约定。

~~~python
def coder_rewards(executed: Tensor, easy_accuracy: Tensor, consistency: Tensor, syntax_error: Tensor) -> Tensor:
    """bool executed/syntax_error [B], float easy_accuracy/consistency [B] in [0,1].
    Successful score=1+easy+min(c,1-c); failed score=-.1. Additionally subtract
    .05 for syntax_error (both error penalties may apply). Unrendered visual
    scores are ignored. Return detached same-dtype [B]. Uses Eq.7's continuous
    difficulty score; Eq.11's later binary range annotation is inconsistent."""
~~~

~~~python
def solver_rewards(answers: Tensor, format_valid: Tensor, num_answers: int, alpha: float = .9) -> Tensor:
    """Long answer IDs [B,K], bool format_valid [B,K], B,K>=1, 0<=alpha<=1.
    Answer IDs are normalized upstream and lie in [0,num_answers). Majority
    includes ALL K answers; tie chooses smallest ID. Return float64 [B,K]:
    alpha*agreement + (1-alpha)*format_valid. Format is an ADDITIVE component,
    not a gate. Consensus is a silver label, not an independent gold answer."""
~~~

## 5. 交互图与实验

在讲解页逐步播放论文流程，观察当前更新的模型、冻结的评价者、反馈和筛选。调整控件会重新计算教学分数。Lab 从均匀 logits 开始，用你的函数提供奖励或筛选样本，执行 60 次学习率 0.5 的真实类别策略梯度更新。使用固定反馈记录和 CPU float64；目标动作概率须大于 0.72，且提升超过 0.3。本实验验证局部机制，不代表完整 LLM/VLM 自演化或论文基准成绩。

## 6. 运行方式

补全 starter 后运行测试。T0 检查接口；T1 使用独立标量与计数推导；T2 检查边界、不变量和梯度归属；T3 将你的实现接入教学训练循环。

~~~bash
paperleet test self_evolving/mmzero_feedback --impl problems/self_evolving/mmzero_feedback/solution.py
paperleet mutate self_evolving/mmzero_feedback
~~~

## 7. 延伸

三个角色共用一套奖励时，哪种奖励投机会首先出现？

来源: [MM-Zero: Self-Evolving Multi-Model Vision Language Models From Zero Data](https://arxiv.org/abs/2603.09206), Sections 2.5–2.6, Eqs. 11–14: execution feedback and silver-answer reward.
