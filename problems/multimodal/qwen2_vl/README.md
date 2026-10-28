# Qwen2-VL 分段 MRoPE

## 1. 问题

实现 `chunked_mrope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor`。

连续频率分段的三轴旋转位置编码；使用等宽三轴的教学配置，不包含动态分辨率编码器。

## 2. 核心公式

$$
a(r)=\lfloor r/(R/3)\rfloor,\quad\theta_{nr}=p_{a(r),n}\omega_r,\quad y=[x_A\cos\theta-x_B\sin\theta;\ x_B\cos\theta+x_A\sin\theta].
$$

## 3. 公式到代码

将 R 个频率分成三个连续区间。频率索引 r 保持不变，只选择其所属轴的位置值；不要给每个轴重新从第一个频率开始。旋转是二维正交变换，配对坐标是 r 和 r+R，因此保持范数；使用负位置可精确逆转。等宽配置让连续分段与后续交错布局的差别可直接比较。

## 4. 本题约定

x=[N,2R]，positions=[3,N] 依次为时间/高度/宽度，inv_freq=[R]；R 为 3 的正倍数。本题采用三轴等宽教学配置，前/中/后 R/3 个频率分别属于时间/高度/宽度。特征按前半与后半配对旋转，不是相邻元素配对；位置允许浮点且保留梯度；输入有限，dtype/device 相同。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/qwen2_vl --impl problems/multimodal/qwen2_vl/solution.py
paperleet mutate multimodal/qwen2_vl
paperleet vizdata multimodal/qwen2_vl
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 2.1: Multimodal Rotary Position Embedding

- [https://arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191)
