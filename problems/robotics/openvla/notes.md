# OpenVLA: 题解与易错点

编码使用末端词表的倒序编号；np.digitize默认right=False使恰在边界上的值进入右侧桶。最大边界会得到最后一个特殊编号，但解码夹到最后中心。反归一化只作用于mask指定的坐标，未指定坐标原样保留。

编码使用末端词表的倒序编号；np.digitize默认right=False使恰在边界上的值进入右侧桶。最大边界会得到最后一个特殊编号，但解码夹到最后中心。反归一化只作用于mask指定的坐标，未指定坐标原样保留。

先按公式确定归约轴、有效掩码和梯度路径，再将手算小例与实现逐项比较。测试中的独立预期值不调用参考答案。

## 测试捕获的错误

- Exact edges use the wrong digitize boundary convention.
- Action IDs are not reversed from the vocabulary end.
- Unmasked coordinates are incorrectly converted to physical units.

## 实验范围

复现官方ActionTokenizer的离散约定及动作反归一化；不加载7B模型。边界数组可缩小以便手算，默认论文规模为256个边界、255个中心。

## 可重复实验记录

33个边界、99个输入值组成33组3维动作，测试[-1.2,1.2]范围。量化归一化误差<=1/32，物理误差<=1/16，超过80个编码可经中心值一致回编码；无需训练器。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
