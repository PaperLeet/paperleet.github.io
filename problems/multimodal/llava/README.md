# LLaVA 视觉 Token 注入与标签对齐

## 1. 问题

实现 `pack_visual_tokens(text: Tensor, visual: Tensor, labels: Tensor, image_index: int) -> tuple[Tensor, Tensor]`。

单图单样本的视觉序列打包与监督标签对齐；输入特征已经投影，不实现视觉编码器、语言模型或训练数据构建。

## 2. 核心公式

$$
E=[E_{<i};V;E_{>i}],\qquad y=[y_{<i};\underbrace{-100,\ldots,-100}_{P};y_{>i}].
$$

## 3. 公式到代码

图像占位是打包指令，不是额外保留的 token。将其左右切片与视觉块拼接，标签使用相同切片边界。视觉 embedding 参与上下文计算，但没有对应的文本预测标签，因此置为 -100。concat 的反向传播自然把梯度分配回各来源，删除的占位不会影响输出。

## 4. 本题约定

单条序列：text=[T,D] 包含一枚图像占位，visual=[P,D] 是已投影特征，labels=[T] int64。用全部 P 个视觉 token 替换 image_index 处的一个 token；视觉标签填 -100，保留其它原标签。返回长度 T+P-1 的 embedding 与 labels。梯度流向保留文本及视觉，删除的占位梯度为零。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/llava --impl problems/multimodal/llava/solution.py
paperleet mutate multimodal/llava
paperleet vizdata multimodal/llava
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 4.1 architecture; official llava_arch.py prepare_inputs_labels_for_multimodal

- [https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)
- [https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py)
