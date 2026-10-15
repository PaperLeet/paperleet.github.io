# MoCo：动量编码器与负样本队列

实现 `moco_loss`、`momentum_update` 和 `enqueue`，把跨 batch 的负样本字典和缓慢更新的 key 编码器连起来。本题基于 [MoCo 论文](https://arxiv.org/html/1911.05722v3) 的式 1、式 2 与 Algorithm 1。

## 1. 一个正样本，加 K 个字典负样本

`query` 和 `key` 为同形状 `[B,D]`，`queue` 为 `[K,D]`，`B,K,D>=1`。它们是同 dtype/device 的有限 float32/float64，允许非连续视图。逐行除以 `max(L2范数, 1e-12)`；零向量保持为零。

对第 i 个 query，第 i 个 key 是唯一正样本，字典中 K 行都是负样本。它不把当前 batch 的其他 key 自动当作负样本。设归一化后的余弦分数除以温度为 s：

$$L=\frac1B\sum_i\left[\operatorname{logsumexp}(s_i^+,s_{i,0}^-,\ldots,s_{i,K-1}^-)-s_i^+\right].$$

正样本保留在分母。返回同 dtype/device 的零维均值，不修改输入。温度为有限、正的 Python int/float；bool 和 Tensor 不接受。提供的验证函数处理非法输入。

`query` 保留通过归一化的梯度；正样本 `key` 和 `queue` 必须 detach，其梯度应为 None。用稳定的 logsumexp，避免小温度时直接指数溢出。

手算：q=k=[1,0]，queue=[[-1,0],[0,1]]，则损失为 `log(1+exp(-2/T)+exp(-1/T))`。所有非零向量相同时为 `log(K+1)`。

## 2. 动量编码器

`momentum_update(query,key,momentum=.999)` 在参数张量上计算：

$$\theta_k^{new}=m\theta_k+(1-m)\theta_q.$$

接受任意相同形状（含标量、空张量）的有限 float32/float64 张量。m 为有限 Python 数值 `[0,1]`，不接受 bool/Tensor。返回 detached 的独立存储，不改写也不共享输入存储。m=0 复制 query，m=1 冻结 key；m=1 是本题增加的边界约定，论文正常训练用 m<1。

## 3. 环形字典

`enqueue(queue,keys,pointer)` 接收 `[K,D]` 的旧字典与 `[B,D]` 的新 key。K,D>=1，B 可为 0 或大于 K。pointer 是 `[0,K)` 的 Python int（不接受 bool）。共享有限浮点 dtype/device。

将新 key 逐行按同一 epsilon 归一化并 detach，从 pointer 开始顺序覆盖环形槽位，返回 `(新字典, (pointer+B)%K)`。输出独立且无计算图；旧字典和 keys 均不改写。空 batch 返回独立副本和原指针。B>K 时必须保留顺序写入后的最后 K 个 key。

本题使用行优先 `[K,D]`，并支持任意批大小的多次回绕，是明确的教学接口约定，不要求照搬原项目队列的内存布局。

## 4. 和 SimCLR 有何不同

SimCLR 在当前两个视图的全部样本之间构造候选，并对两个视图反向传播；这里使用历史字典，只对 query 反向传播。EMA 让跨 batch 的 key 编码器变化平缓，FIFO 逐渐淘汰较旧的表示。队列应存储本步编码得到的旧 key，不能在 EMA 后重新编码覆盖本步 key。

## 5. 实际 CPU 小实验

已提供 6 对固定的 4 维合成特征、4×3 的 query 线性编码器（12 个可训练数值）、相同初始化的 key 编码器和 5 行字典。执行 12 次真实 SGD，batch=2，lr=.05，m=.8，T=.3：编码 q/k → detach key 的损失 → 更新 query → EMA 更新 key 参数 → 入队本步缓存的 key。

独立 Python 标量目标与中心有限差分驱动另一套参数、EMA 和字典轨迹，逐步核对所有状态。图表展示实际 batch 损失、编码器参数距离和队列指针；不要求损失单调或达到人为准确率阈值。测试还覆盖两组梯度截断、query 有限差分、手算分母、置换/尺度不变性、空 batch、多次回绕和不共享存储。

这是 PaperLeet 原创合成教学改编，不使用图像、ResNet、数据增强、分布式 shuffling BN 或下游评估，不能据此声称复现论文结果。无需 GPU、账号、下载数据或预训练权重。

## 6. 运行

```bash
paperleet init representation/moco
paperleet test representation/moco
paperleet test representation/moco --impl problems/representation/moco/solution.py
paperleet vizdata representation/moco
```

可在 CPU/WASM 浏览器运行完整 T0–T3。浏览器不支持的 PyTorch 用法保留原生服务端判题。
