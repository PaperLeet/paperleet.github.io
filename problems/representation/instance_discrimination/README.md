# 实例判别：Binary NCE 与记忆库


## 1. 问题


实现 `instance_nce_loss` 和 `update_memory`。非参数实例判别为每张训练图像保留一个记忆库条目，当前查询与正实例及噪声样本进行比较。原论文采用二元噪声对比估计（binary NCE）；候选 softmax 的 InfoNCE 是另一种目标。


## 2. 核心公式


设查询为 $q_b$，候选 ID 为 $j_{bt}$，配分常数 $Z>0$ 固定，温度为 $T$，从 $N$ 个条目均匀抽取 $K$ 个噪声候选：
$$a_{bt}=q_b^T v_{j_{bt}}/T-\log Z-\log(K/N),$$
$$L=\frac1B\sum_b[\operatorname{softplus}(-a_{b0})+\sum_{t=1}^K\operatorname{softplus}(a_{bt})].$$
第 0 列为正例。所有 $a=0$ 时得到 $(K+1)\log2$。记忆更新为 $v_i\leftarrow\operatorname{normalize}(m v_i+(1-m)q_i)$。


## 3. 公式到代码


从 `memory.detach()` 取候选，点积后减去固定 `log_z` 和噪声先验的对数。在对数域用 softplus 保持稳定；K 个噪声项求和，再对 B 个查询求平均。不能把 `K/N` 改为 `1/N`，也不能忽略 Z；整体平移模型对数似然会改变 binary NCE。记忆更新返回独立、脱离梯度的副本。


## 4. 本题约定


查询 `[B,D]`、记忆 `[N,D]`、候选 int64 `[B,K+1]`，K≥1；浮点数据有限且 dtype/device 一致。编码器通常先归一化，但本损失直接使用输入值，不额外归一化。`log_z` 为固定有限 Python 标量；官方一次性估计 Z 的步骤不在范围内。均匀噪声可重复抽取，也可碰到正实例。损失仅对查询回传梯度。记忆更新 ID 必须互异，动量在 `[0,1]`，范数下界默认 `1e-12`；未选中行保持原样，零混合向量保持为零。


即使 momentum 为 1，选中条目仍需归一化；返回值始终拥有独立存储，即使数值无需更新。

## 5. 交互图与 CPU 实验


六个六维归一化实例向量执行 80 次 SGD，学习率 `.35`，T=`.2`，固定 Z=`exp(5)+5`，记忆动量 `.95`。每次枚举另外五个合成实例作为噪声候选，是确定性教学日程，并非复现原始随机采样。实验调用两个待实现函数；T3 要求损失低于 `.8` 且低于初值的 40%，查询与自身记忆平均余弦大于 `.99`，记忆保持单位长度且无梯度。没有图像编码器或下游指标复现。


## 6. 运行方式


网站运行完整 CPU/WASM 测试，或使用下方命令。测试独立计算二元概率，检查查询有限差分与非连续梯度、记忆冻结、重复噪声、EMA 端点、实例重标号和极端分数。


```bash
paperleet test representation/instance_discrimination --impl problems/representation/instance_discrimination/solution.py
paperleet mutate representation/instance_discrimination
paperleet vizdata representation/instance_discrimination
```



## 7. 延伸与来源


与 InfoNCE 的候选分类比较，可以看到二元“数据还是噪声”分类依赖校准后的似然尺度。与 MoCo 比较：MoCo 对编码器做动量更新并维护 FIFO 队列，本方法按数据集实例 ID 更新固定记忆条目。本题对数域公式与数学 binary NCE 等价；不加入官方概率分母中防止数值异常的微小 epsilon，以保持给定概率的精确含义。记忆更新显式执行，不隐藏在自定义 backward 内。


[Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://arxiv.org/abs/1805.01978) · [Official implementation](https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/NCECriterion.py)
