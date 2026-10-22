# 105. VDP：视觉提示与稳态正则

## 问题

[Visual Domain Prompt](https://arxiv.org/abs/2212.04145) 固定源模型，通过改变图像输入适应域变化。本题实现提示叠加、跨历史域累积的稳态因子，以及域无关提示的惩罚。教师推理、域检测、重要性估计和教师 EMA 不在本题范围内。

## 核心公式

论文 Method 的式 (1)、(3)、(4) 对应：

$$
x'=x+p_{specific}+p_{agnostic},\qquad
\Lambda_i=\sum_{v=1}^{T}\frac{\eta_{v,i}}{\delta_{v,i}^2+\xi},
$$

$$
P(p)=\alpha\sum_i\Lambda_i(p_i-a_i)^2.
$$

`importance` 提供 $\eta$，`displacement` 提供 $\delta$，`anchor` 提供 $a$。每个已结束域单独贡献一项，惩罚中没有二分之一。原文式 (4)–(6) 附近的提示符号与文字存在不一致；本题使用语义名称，按方法说明将惩罚施加在域无关提示上。

## 公式到代码

`apply_prompts(images, specific, agnostic)` 将两个 `[C,H,W]` 提示画布广播到 `[B,C,H,W]`。位置已处理好，提示覆盖之外的画布为零。

`homeostatic_factor(importance, displacement, xi=0.01)` 将 `[T,*S]` 历史记录变为形状 `S` 的 detached 因子。例如某坐标的重要性为 `[2,3]`，位移为 `[1,0]`，xi 为 `0.5`，因子为 `2/1.5 + 3/0.5`。

`homeostatic_penalty(prompt, anchor, factor, strength)` 返回标量，仅当前 `prompt` 接收梯度。

## 本题约定

- 输入为同 dtype 的有限 CPU float32/float64，中间运算保持有限。不要求输入检查，不得修改输入；支持非连续视图并保留 dtype/device。
- 图像 `B>=0`，`C,H,W` 为正。只叠加、不裁剪，保留图像和两个当前提示的梯度。
- 历史记录 `T>=0`，参数形状 `S` 非空，重要性非负、`xi>0`。空历史返回零；两个历史记录均冻结。
- 惩罚的三个张量形状完全相同，因子和 strength 非负，对坐标求和。即使 anchor/factor 开启梯度，也必须冻结它们。零强度或零因子返回可微的零。

## 交互图与实验

固定的恒等读出读取两个带提示的像素。提供的教师特征目标为 `[1,-1]`，历史对应因子为 `[4,0]`。仅训练域无关提示，域专用提示保持为零。两组分别使用强度 0 和 1，以学习率 `0.1` 进行 120 次 float64 SGD 更新。

为便于独立求解，本实验用特征平方一致性 `0.5*||p-target||²` 替代论文的分类损失，解析最优点分别为 `[1,-1]`、`[1/9,-1]`。Lab 展示适应与保持的权衡；不包含教师 EMA、图像损坏数据或在线域检测，不声称完整复现 CTTA 或保证不遗忘。

## 运行方式

完成三个函数后点击 **Run tests**。测试覆盖独立标量结果、广播梯度、历史可加性、空历史、历史冻结，以及真实提示训练与解析最优点的对齐。

```bash
paperleet test finetuning/visual_domain_prompt --impl problems/finetuning/visual_domain_prompt/solution.py
paperleet mutate finetuning/visual_domain_prompt
```

## 延伸

为什么历史贡献大但移动小的坐标应受到更强约束？此处因子与 EWC 的 Fisher 对角有什么差别？实现无标签在线适应还需要哪些模块？
