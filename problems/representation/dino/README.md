# DINO：中心化教师与多视图自蒸馏

## 1. 题目

实现 `teacher_probabilities(teacher, center, temperature=.04)`、`dino_loss(student, teacher, center, student_temperature=.1, teacher_temperature=.04)` 和 `update_center(center, teacher, momentum=.9)`。

[DINO](https://arxiv.org/abs/2104.14294) 第 3.1 节及算法 1 使用中心化、锐化的教师概率跨视图蒸馏。教师 logits 为 `[2,B,K]`，对应两个全局视图；学生为 `[V,B,K]`，`V>=2`，前两个视图与教师全局视图一一对应，其余为额外裁剪。本题从 logits 出发实现损失和中心更新，不要求教师网络参数 EMA、图像裁剪、骨干网络或下游评测。

## 2. 核心公式

给定 `[K]` 中心 `c`、教师温度 `τt` 和学生温度 `τs`：

$$
q_{t,i}=\mathrm{sg}\left[\mathrm{softmax}\left((u_{t,i}-c)/\tau_t\right)\right],
\quad \log p_{v,i}=\mathrm{logsoftmax}(s_{v,i}/\tau_s).
$$

使用所有教师—学生组合，排除同一个全局视图：

$$
L=-\frac1{2(V-1)B}\sum_{t=0}^1\sum_{\substack{v=0\\v\ne t}}^{V-1}
\sum_{i,k}q_{t,i,k}\log p_{v,i,k}.
$$

当前损失计算完成后，用**原始教师 logits** 更新中心：

$$
c^{new}=m c+(1-m)\frac1{2B}\sum_{t,i}u_{t,i}.
$$

新中心停止梯度，并拥有独立于两个输入的存储。

## 3. 从公式到实现

先减中心，再除教师温度、取 softmax。阻断教师和中心梯度，损失只更新学生 logits。学生部分用 `log_softmax` 保持稳定。对 `2*(V-1)` 个视图对和 batch 取平均，类别轴的交叉熵项求和。

`V=2` 时只有 `(teacher0,student1)`、`(teacher1,student0)`；`V=3` 时再加入 `(teacher0,student2)`、`(teacher1,student2)`，共四对。中心均值同时包含两个教师视图和 batch，不使用学生、概率或已中心化的 logits。调用者在当前目标计算后安装新中心。

## 4. 约定与边界

输入为有限、非空的 float32/float64 张量，dtype/device 一致。教师 `[2,B,K]`、中心 `[K]`、学生 `[V,B,K]`，其中 `V>=2`、`B>=1`、`K>=1`。输入可不连续，不得原地修改。概率保持教师形状，损失为标量；中心更新返回 `[K]`，即使 momentum 为端点也需独立存储且无梯度。

温度是有限正 Python 数字；momentum 是 `[0,1]` 内有限 Python 数字。不接受 bool 或 Tensor 标量控制参数。校验辅助函数已提供。温度由调用者传入，函数内不隐藏按 epoch 变化的计划。

测试使用独立标量概率和损失、解析学生梯度，检查教师/中心停止梯度，并覆盖视图计数、同视图排除、不变性、中心权重及端点存储。

## 5. 交互图与 CPU 小实验

两组固定 `6×3` 教师 logits 编码三个类别并带有共同偏置。学生为 `3×6×3` 的可学习表，共 54 个参数。执行 80 步 SGD，学习率 `.6`，学生温度 `.3`，教师温度 `.15`；每次损失之后以 momentum `.9` 更新中心。每八步记录交叉熵与中心范数，并统计最终学生类别一致率和最大中心误差。

每步实际调用你提交的损失和中心函数，并以独立标量 oracle 检查概率函数。教师固定，因此中心还有闭式 EMA oracle。通过条件：最终损失低于 `.08` 且低于初始的 10%，18 个学生预测全部符合合成类别，标量损失/概率误差小于 `1e-11`，中心误差小于 `1e-12`，教师无梯度。CPU float64，种子 42。这个表格蒸馏实验展示目标与移动中心，不训练 ViT，也不衡量图像表征质量。

## 6. 运行方式

```bash
paperleet test representation/dino --impl problems/representation/dino/solution.py
paperleet mutate representation/dino
```

将 `--impl` 指向你完成的 starter。完整 CPU/WASM 测试无需图像数据、GPU 或预训练权重。

## 7. 延伸阅读

中心化在输出坐标间起作用，锐化控制分布的集中程度。分析排除同视图后每个学生实际匹配的教师混合目标。比较 BYOL 的教师参数 EMA 与 DINO 的 logits 中心 EMA：二者更新对象不同，完整 DINO 同时使用两者。

来源：[论文](https://arxiv.org/abs/2104.14294)、[官方 DINOLoss 与训练循环](https://github.com/facebookresearch/dino/blob/main/main_dino.py)。固定温度、单进程张量契约与合成实验为独立编写的教学改编。
