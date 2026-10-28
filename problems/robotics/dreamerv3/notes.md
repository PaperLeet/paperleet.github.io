# DreamerV3: 题解与易错点

先symlog压缩原始标量，再在相邻两个中心间线性分配质量。编码期望等于裁剪后的变换值；解码时先在变换空间求期望，再symexp。将symexp分别作用于中心后平均不是同一操作。

先symlog压缩原始标量，再在相邻两个中心间线性分配质量。编码期望等于裁剪后的变换值；解码时先在变换空间求期望，再symexp。将symexp分别作用于中心后平均不是同一操作。

先按公式确定归约轴、有效掩码和梯度路径，再将手算小例与实现逐项比较。测试中的独立预期值不调用参考答案。

## 测试捕获的错误

- Negative rewards lose their sign under symlog.
- Two-hot weights assume uniform spacing even for a nonuniform support.

## 实验范围

实现跨数量级标量变换与离散支持编码；不训练RSSM、想象策略或Minecraft。two_hot输入已经位于symlog空间，中心由调用者指定。

## 可重复实验记录

6个跨数量级标量、25个[-6,6]中心；零logits做150步交叉熵梯度下降(lr=3)。变换往返误差<1e-9，交叉熵<初始40%，最终标签KL<0.1。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
