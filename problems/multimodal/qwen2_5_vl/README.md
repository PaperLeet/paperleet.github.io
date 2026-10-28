# Qwen2.5-VL 视觉窗口重排

## 1. 问题

实现 `window_permutation(frames: int, height: int, width: int, merge_size: int, window_size: int) -> tuple[Tensor, Tensor]`。

动态分辨率下的窗口分组、空间合并顺序与累计长度；不实现完整 ViT 或 GPU 内核。

## 2. 核心公式

$$
\operatorname{index}(t,h,w)=(tH+h)W+w,\quad c_{j+1}=c_j+\#\{\text{valid original patches in window }j\}.
$$

## 3. 公式到代码

先把原始 H×W 网格视为由 merge_size×merge_size 小格组成的合并网格，再按窗口分组这些小格。每个小格内部恢复 row-major patch 顺序；累计边界必须以原始 patch 数量计量。边缘窗口只收集真实格子，不能让 padding 出现在 permutation 中。实验利用返回索引做窗口均值，再散射回原网格，和空间切片 oracle 对照。

## 4. 本题约定

所有输入为正 int；height/width 是原 patch 网格且可被 merge_size 整除；window_size 以合并后的格子为单位。逐帧、逐窗口行/列、窗口内逐合并格行/列、格内逐 patch 行/列输出原始 row-major patch 索引；边界窗口不输出 padding。返回 CPU int64 permutation=[T*H*W] 与以原 patch 个数计数的 cu_seqlens，首项 0。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/qwen2_5_vl --impl problems/multimodal/qwen2_5_vl/solution.py
paperleet mutate multimodal/qwen2_5_vl
paperleet vizdata multimodal/qwen2_5_vl
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 2.1.1 Fast and Efficient Vision Encoder: window attention and spatial merging

- [https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)
- [https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
