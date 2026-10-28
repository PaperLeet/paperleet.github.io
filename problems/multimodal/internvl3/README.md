# InternVL3 可变视觉位置增量

## 1. 问题

实现 `variable_visual_positions(visual: Tensor, image_ids: Tensor, deltas: Tensor) -> Tensor`。

每图像可变增量的位置递推、跨模态累计与梯度；原点明确固定为 -1，不实现完整旋转嵌入或 InternVL 训练。

## 2. 核心公式

$$
p_i=-1+\sum_{j=0}^{i}\left[(1-v_j)+v_j\delta_{\operatorname{image}(j)}\right].
$$

## 3. 公式到代码

递推依赖当前 token 的模态，等价于所有增量的前缀和再减一。同一图像使用同一个 delta，跨图像或文本边界都不能重置累计位置。delta 的梯度由该图像 token 在后续前缀中出现的次数决定。delta=1 应恢复普通 0 到 L-1 的位置；本题首 token 若是视觉，则位置是 delta-1，这是显式原点约定。

## 4. 本题约定

visual=[L] bool；image_ids=[L] int64，视觉位置给出 [0,M) 图像编号，文本位置使用 -1；deltas=[M] 浮点且 0<delta<=1。同一图像共享 delta。递推从 p_-1=-1 开始，当前 token 是文本则加 1，是视觉则加相应 delta。输出 [L] 与 deltas 同 dtype/device；允许空序列和 M=0 的纯文本，保留 delta 梯度。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/internvl3 --impl problems/multimodal/internvl3/solution.py
paperleet mutate multimodal/internvl3
paperleet vizdata multimodal/internvl3
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 2.1: Variable Visual Position Encoding, Equations 3 and 4

- [https://arxiv.org/abs/2504.10479](https://arxiv.org/abs/2504.10479)
