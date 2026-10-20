# 111. VisPlay 图像分组奖励

## 1. 问题

对每张无标注图片分别生成一组问题，并在图片内部计算重复惩罚。不同图片可以使用相同的局部聚类 ID。

实现 `image_group_rewards`。

## 2. 核心公式

$$
r_i=v_i\max(0,1-|2p_i-1|-\lambda |C_i^{(I)}|/G_I).
$$

## 3. 公式到代码

先用 image_ids 构造组，再在组内比较 clusters。某张图有四题，其中两题重复且 p=0.5，奖励为 0.5。加入另一张图片的题不能改变已有组的奖励。

## 4. 本题约定

分母是当前图片的题目数，不是全局批次大小。局部聚类 ID 跨图片重用不代表重复。VisPlay 仍需要外部无标注图片；这里只实现观测反馈的奖励聚合。

输入为有限 CPU 张量，不得原地修改。分数使用 float32/float64，ID 使用 long，掩码使用 bool。以下 API 明确形状、平票和梯度约定。

~~~python
def image_group_rewards(consistency: Tensor, image_ids: Tensor, clusters: Tensor, valid: Tensor, strength: float = 1.) -> Tensor:
    """Float consistency [B] in [0,1]; long image/cluster IDs [B]; bool valid [B].
    Cluster IDs are LOCAL to each image. For each row count all proposals for its
    image (including invalid ones), and the same-image, same-cluster proposals.
    Return detached same-dtype [B]: valid*relu(1-abs(2*p-1)-strength*cluster/group).
    B>=1; IDs need not be contiguous; strength>=0. No mutation."""
~~~

## 5. 交互图与实验

在讲解页逐步播放论文流程，观察当前更新的模型、冻结的评价者、反馈和筛选。调整控件会重新计算教学分数。Lab 从均匀 logits 开始，用你的函数提供奖励或筛选样本，执行 60 次学习率 0.5 的真实类别策略梯度更新。使用固定反馈记录和 CPU float64；目标动作概率须大于 0.72，且提升超过 0.3。本实验验证局部机制，不代表完整 LLM/VLM 自演化或论文基准成绩。

## 6. 运行方式

补全 starter 后运行测试。T0 检查接口；T1 使用独立标量与计数推导；T2 检查边界、不变量和梯度归属；T3 将你的实现接入教学训练循环。

~~~bash
paperleet test self_evolving/visplay_rewards --impl problems/self_evolving/visplay_rewards/solution.py
paperleet mutate self_evolving/visplay_rewards
~~~

## 7. 延伸

为什么跨图片做全局去重会错误地惩罚合法问题？

来源: [VisPlay: Self-Evolving Vision-Language Models from Images](https://arxiv.org/abs/2511.15661), Section 2.3, Eqs. 5–8: per-image diversity and uncertainty.
