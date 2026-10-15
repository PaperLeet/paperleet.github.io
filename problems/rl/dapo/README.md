# DAPO 动态筛选与 Token 目标

筛选有效组，实现非对称的按 token 归一化截断目标。

论文：[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476v1)。定位：Section 3, Eqs. (8–11): token normalization, asymmetric clipping and dynamic sampling；Sections 3.1–3.3; Section 4.1 clip_low=.2 and clip_high=.28。

## 精选核心

### 返回值要求

按下列结构返回，保留形状中的单例维度。标量 Tensor 的形状为 `()`，不是 Python `float`。

#### `dynamic_group_mask` · 需要实现

对 `[P, G]` 奖励（`P` 个问题、每组 `G` 个回答），返回奖励同设备上的布尔 Tensor `keep`，形状 `[P]`。组内存在不同结果时为 `True`；全对或全错的组为 `False`。每个问题返回一个布尔值，不是逐回答或逐 token 的 mask。

#### `dapo_loss` · 需要实现

返回一个浮点标量 Tensor `loss`，形状 `()`：将 `[N, L]` 输入的截断目标按保留的有效 token 求和、加负号，再除以全部有效 token 数。保留当前 `logp` 的梯度，固定 `old_logp`/`advantages`；padding 梯度为零。空 mask 返回可反传的零 Tensor，不返回逐回答 loss 或诊断字典。

### 公式与约定

两个有界核心函数；不复现超采样调度、超长奖励整形和分布式训练。

$$
L=-\frac{\sum_{i,t}m_{it}\min(r_{it}A_i,\mathrm{clip}(r_{it},1-\epsilon_l,1+\epsilon_h)A_i)}{\sum_{i,t}m_{it}}
$$

- `dynamic_group_mask(rewards: Tensor) -> Tensor`
- `dapo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor, clip_low: float=0.2, clip_high: float=0.28) -> Tensor`

## 接口与实验

`dynamic_group_mask` 输入二元结果奖励 `[prompts,group]`，组大小至少为 2。输出 `[prompts]` 布尔向量，仅保留同时有成功与失败的组。支持 `0/1` 和 `-1/1` 编码。筛选是动态采样的核心原语，生产采样器还需继续补满 batch。

`dapo_loss` 输入 `[B,L]` 当前/旧 log-probability、二元 mask 和 `[B]` 已归一化 advantage。调用者已筛选这些回答。冻结旧分数与 advantage，独立使用上下界，按**全 batch 有效 token 总数**归一化，不能先求回答内均值。默认上下宽度为 `.2/.28`，不添加参考策略 KL。空 mask 返回可微零，指数前屏蔽极端 padding。

Toy 在全错、混合、全对三个组中只保留混合组，然后扫描 .5 至 1.5 的比率。正 advantage 的 1.25 比率在上界宽度 .28 时仍应有梯度；测试另检查不同回答长度的权重。

## Toy 来源

**PaperLeet 教学改编。** PaperLeet 二元奖励组与比率扫描，不是 AIME 复现。 判题 harness 调用本次提交的函数，保存由其产生的指标与曲线。Lab 静态预览来自参考实现，Run/Submit 的产物来自你的代码。`experiment.py` 与 `tests/` 已提供实验和判题脚手架。

## 来源与验证

[原论文链接的官方框架（重定向至 verl-project/verl）](https://github.com/volcengine/verl)。题目实现与合成数据由 PaperLeet 编写；论文链接固定公式依据的版本。

T0 验证接口，T1 对照参考值，T2 检查数学性质与梯度，T3 用你的实现运行 toy。在在线编辑器补全 starter 后 Run 或 Submit，无需部署完整大模型训练系统。
