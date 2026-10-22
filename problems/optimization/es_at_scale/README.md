# 104. ES at Scale：无反向传播的奖励加权更新

## 问题

实现 `reward_zscores` 和 `es_step`。调用方评估受到随机扰动的模型并提供奖励；你需要让权重沿高奖励对应的方向移动。[ES at Scale](https://arxiv.org/abs/2509.24372) 将这一思路用于大模型全参数后训练，本题提取其中的数值更新。

## 核心公式

论文第 3 节 Algorithm 1 评估候选参数 $\theta+\sigma\epsilon_i$，并标准化奖励：

$$
\bar r=\frac1N\sum_i r_i,\quad
s=\sqrt{\frac1N\sum_i(r_i-\bar r)^2},\quad
z_i=\frac{r_i-\bar r}{\max(s,\varepsilon)}.
$$

$$
\theta_{new}=\theta+\frac{\alpha}{N}\sum_i z_i\epsilon_i.
$$

本题明确使用总体标准差并加入正数下界，使单个候选、常数奖励都有定义。与论文更新式一致，`alpha` 已吸收通常的噪声尺度倒数，不要再次除以 sigma。

## 公式到代码

`rewards` 为 `[N]`；参数 `theta` 的形状为 `S`；`noise` 为 `[N,*S]`。每个候选对应一个奖励权重，将它广播到参数坐标后，仅对候选轴求均值。传入的是尚未乘 sigma 的标准正态方向。

例如标量 `theta=2`、`noise=[-1,1]`、`rewards=[0,2]`、`alpha=0.1`，标准化奖励为 `[-1,1]`，新参数为 `2.1`。

## 本题约定

- `N>=1`，参数可为标量或非空张量块。输入为同 dtype/device 的有限 CPU float32/float64；允许非连续视图，中间运算保持有限。
- `eps>0`、`alpha>=0`。给标准差设置下界，不是在方差上加 eps；不用样本方差或奖励排名。
- 两个函数都返回与输入同 dtype/device 的 detached 张量，且不修改输入。黑箱优化器不保留反向传播图。
- 常数奖励产生全零权重，不改变参数；零学习率不更新。随机采样与奖励评估由调用方负责。

## 交互图与实验

Lab 使用你的更新函数，从 `[-2,3]` 出发，最大化 `-||theta-[1,-1]||²`。共进行 100 次真实更新，每次 30 个候选，sigma 为 `0.1`、alpha 为 `0.05`。CPU float64、种子 42、两条线程下，参考平方距离从 25 降至约 0.01254，验收阈值为 0.08。图展示实际搜索轨迹，不涉及反向传播；这不是大模型实验或训练成本比较。

## 运行方式

完成两个函数后点击 **Run tests**。T0–T2 验证独立标量计算、候选复制、奖励平移缩放、方向符号、退化输入和梯度隔离；T3 全程使用你的更新函数。

```bash
paperleet test optimization/es_at_scale --impl problems/optimization/es_at_scale/solution.py
paperleet mutate optimization/es_at_scale
```

## 延伸

为什么把每个候选复制一次不应改变更新？为什么 30 个方向可能给出有用更新，却无法精确恢复十亿维梯度？与 Neural Thickets 的专家筛选、预测集成有何区别？
