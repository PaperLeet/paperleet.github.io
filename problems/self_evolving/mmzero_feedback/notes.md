# 实现讲解

Coder 成功且 easy=0.8、c=0.5 时得 2.3；渲染与语法均失败得 -0.15。Solver 采用 α=0.9：答案同意但格式不合格仍得 0.9；不同意但格式合格得 0.1。

Coder 不使用 Proposer 的 easy 上限。失败渲染没有视觉分数，两类错误可叠加。论文 Eq.11 附近的 difficulty 二值范围注释与 Eq.7 不一致；本题采用 Eq.7 的连续 min(c,1-c)。Solver 格式项是加法而非门控。

参考行为以 starter 的显式契约为准，包括确定性平票处理和上游观测量。教学策略实验不证明自生成标签正确；应把独立标准答案评估与合成标签一致率分开记录。

https://arxiv.org/abs/2603.09206

[官方 Coder 奖励实现](https://github.com/zli12321/MM-Zero/blob/main/MM-zero_final/reward_function/codegen_reward.py) 同样使用连续的 `min(c, 1-c)`。本题抽取论文奖励和两项明确的错误惩罚；仓库中的渲染类型惩罚不在本题 API 内。
