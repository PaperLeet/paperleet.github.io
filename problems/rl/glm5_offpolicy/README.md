# GLM-5：异步 RL 的双边重要性采样

来源：[GLM-5 技术报告 v2](https://arxiv.org/abs/2602.15763v2)，§4.1.2 “Optimizing Asynchronous Training Stability”，公式 (3–5)。异步 rollout 可能由不同版本的策略生成。报告用当前概率与 rollout 概率的比值纠偏，并丢弃落在双边区间之外的 token。

## 需要实现

### 返回值要求

按下列结构返回，保留形状中的单例维度。标量 Tensor 的形状为 `()`，不是 Python `float`。

#### `masked_is_weights` · 需要实现

返回 **已 detach** 的浮点 Tensor `weights`，形状 `[B, L]`（`B` 个回答、`L` 个 token），dtype/device 与 `logp` 一致。保留位置存放严格落在 `(1-clip_low, 1+clip_high)` 内的 `exp(logp-rollout_logp)`；padding、边界及丢弃位置为零。返回完整 token 数组，不聚合、不修改输入。

#### `offpolicy_loss` · 需要实现

返回浮点标量 Tensor `loss`，形状 `()`，分母是 **原始有效 token 数**，包含因比率规则被丢弃的 token。梯度仅经过直接相乘的当前 `logp`，权重、rollout 分数和 advantage 固定。空 mask 返回仍连接计算图的零，支持 `.backward()`；不能返回 `.item()` 或 detach 的标量。

### 公式与约定

实现 `masked_is_weights` 与 `offpolicy_loss`。当前 log-prob、已记录的 rollout log-prob、二值有效 token mask 的形状均为 `[B,L]`。优势为 `[B]`，在每个回答的 token 上广播。

$$r_{it}=\exp(\log\pi_{it}-\log\pi_{rollout,it}), \quad
w_{it}=\operatorname{sg}\!\left[r_{it}\mathbf1(1-\epsilon_l<r_{it}<1+\epsilon_h)\right],$$
$$L=-\frac{\sum_{it}m_{it}w_{it}\operatorname{sg}(A_i)\log\pi_{it}}{\max(1,\sum_{it}m_{it})}.$$

`sg` 表示停止梯度。padding 和**两个端点**的权重为零，不能把越界权重截成端点。分母统计**所有原始有效 token**，包括被比率规则丢弃的 token。空 mask 返回可反向传播的零。

权重、rollout 分数与优势全部冻结；只有直接乘上的当前 log-prob 接收策略梯度。运算前排除任意 padding 值，包括 NaN；对极大且应被丢弃的比率也必须安全返回零。按计算出的比率及其原 dtype 比较边界；冻结的指数运算内部可以溢出，但要用 `torch.where` 选择零，不能把无穷乘零。保持 float32/float64 dtype，不修改输入。形状不符或区间参数非法时抛出 `ValueError`，要求 `0 < clip_low < 1` 且 `clip_high` 有限、正值。未被 mask 的 log-prob 和优势均有限，mask 为 0/1 或布尔值。

## 论文与实现约定

论文公式 (5) 使用**严格开区间**，当前官方 Slime 的 `importance_weights(mode="mis")` 则包含端点。本题明确采用论文的严格边界，不声称与该 helper 在端点完全一致。

论文展示的目标没有显式标出 stop-gradient。本题采用冻结的重要性权重，与 Slime 的 REINFORCE 路径将 detached 当前概率传给纠偏 callback 的做法一致。上述分母和空 mask 行为是明确的题目约定；默认宽度 `.2/.2` 是教学参数，不代表原文生产训练超参数。

官方 Slime 固定到 commit `05e3b017fed9e02de418c2fbac9d75602a534be5`：[重要性权重](https://github.com/THUDM/slime/blob/05e3b017fed9e02de418c2fbac9d75602a534be5/slime/utils/ppo_utils.py#L25)、[策略损失](https://github.com/THUDM/slime/blob/05e3b017fed9e02de418c2fbac9d75602a534be5/slime/backends/megatron_utils/loss.py#L923)中的 `icepop_function` 以及 `policy_loss_function` 内的 detached `train_log_probs_for_tis`。helper 的名字不意味着 GLM 的完整训练方法就是 ICE-POP。

## Toy 与验证

`experiment.py` 已提供比率扫描和确定性的双动作 bandit。扫描展示你的权重、截断权重对照和真实梯度；bandit 每五次更新才刷新 rollout 策略，用滞后的数据训练。这是 **PaperLeet 教学改编**，不是作者提供的 toy，也不是代码智能体成绩的复现。T3 的曲线来自本次提交，实验页的静态参考预览来自参考实现。

T0 检查空 mask 和接口；T1 使用独立标量循环 oracle；T2 检查严格端点、梯度冻结、padding、数值极端与 token 平均；T3 运行已提供的 toy。无需模型下载、付费 API、异步集群、tokenizer gateway 或代码沙箱系统。
