# Qwen3-VL 交错 MRoPE

## 1. 问题

实现 `interleaved_mrope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor`。

交错频率轴分配与半维配对旋转；使用等宽三轴的教学配置，不包含 DeepStack 或完整视频模型。

## 2. 核心公式

$$
a(r)=r\bmod3,\quad\theta_{nr}=p_{a(r),n}\omega_r,\quad y=[x_A\cos\theta-x_B\sin\theta;\ x_B\cos\theta+x_A\sin\theta].
$$

## 3. 公式到代码

与连续三段不同，轴序列为时间、高度、宽度并循环，使每个轴跨越多个频段。位置乘对应频率后，仍采用前后半维旋转。因为三轴位置一般不同，错误的连续分段不会被普通的一维 RoPE 测试发现；测试使用各轴不同坐标并逐坐标计算 oracle。

## 4. 本题约定

x=[N,2R]，positions=[3,N] 依次为时间/高度/宽度，inv_freq=[R]；R 为 3 的正倍数。本题采用三轴等宽教学配置，频率索引 r 对应轴 r%3。特征按前半与后半配对旋转，不是相邻元素配对；位置允许浮点且保留梯度；输入有限，dtype/device 相同。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/qwen3_vl --impl problems/multimodal/qwen3_vl/solution.py
paperleet mutate multimodal/qwen3_vl
paperleet vizdata multimodal/qwen3_vl
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 2.1 Interleaved MRoPE

- [https://arxiv.org/abs/2511.21631](https://arxiv.org/abs/2511.21631)
