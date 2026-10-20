# 110. G-Zero 偏好样本筛选

## 1. 问题

从已经评分的候选池中构造问题级偏好对，让 Generator 在没有提示的输入上学习有提示回答。

实现 `preference_mask`, `build_preferences`。

## 2. 核心公式

$$
m_i=\mathbf1[\delta_i\le\operatorname{median}(\delta)]\land q_i,\quad (x,y_w,y_l)=(question,assisted,unassisted).
$$

## 3. 公式到代码

先在完整候选池上求线性插值中位数，再应用质量掩码，保持原始顺序。δ=[-3,-2,10,11] 且前两项质量失败时，结果为空；不能重算中位数把后两项补进来。

## 4. 本题约定

包含中位数平票，所以保留数可能多于一半。允许负 δ。不要把 hint 放进 prompt，不要交换 chosen/rejected。质量评分本身已由上游完成。

输入为有限 CPU 张量，不得原地修改。分数使用 float32/float64，ID 使用 long，掩码使用 bool。以下 API 明确形状、平票和梯度约定。

~~~python
def preference_mask(delta: Tensor, quality: Tensor) -> Tensor:
    """Finite float delta [B], bool quality [B], B>=1. Compute the LINEAR median
    of the full pool first; retain delta<=median AND quality. Include ties and
    negative deltas; a singleton is retained iff quality. Return bool [B].
    Quality checks are upstream. The median is NOT recomputed after filtering."""
~~~

~~~python
def build_preferences(records: list[dict], delta: Tensor, quality: Tensor) -> list[dict]:
    """B records have question, hint, unassisted, assisted string fields.
    Return NEW dictionaries in original order for preference_mask-selected rows:
    exactly prompt=question, chosen=assisted, rejected=unassisted. Never put the
    hint into prompt, mutate inputs or include delta/hint in output dictionaries."""
~~~

## 5. 交互图与实验

在讲解页逐步播放论文流程，观察当前更新的模型、冻结的评价者、反馈和筛选。调整控件会重新计算教学分数。Lab 从均匀 logits 开始，用你的函数提供奖励或筛选样本，执行 60 次学习率 0.5 的真实类别策略梯度更新。使用固定反馈记录和 CPU float64；目标动作概率须大于 0.72，且提升超过 0.3。本实验验证局部机制，不代表完整 LLM/VLM 自演化或论文基准成绩。

## 6. 运行方式

补全 starter 后运行测试。T0 检查接口；T1 使用独立标量与计数推导；T2 检查边界、不变量和梯度归属；T3 将你的实现接入教学训练循环。

~~~bash
paperleet test self_evolving/gzero_pairs --impl problems/self_evolving/gzero_pairs/solution.py
paperleet mutate self_evolving/gzero_pairs
~~~

## 7. 延伸

为什么出题阶段追求高 δ，而训练数据阶段保留较低 δ？

来源: [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://arxiv.org/abs/2605.09959), Section 3.3: lower-half delta filter and question-only DPO; official phase2.py.
