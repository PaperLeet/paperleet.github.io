# AWQ 激活感知缩放校准

## 1. 问题

实现分组反量化，并以校准输出误差选择激活感知缩放指数。

单线性层AWQ网格搜索与分组非对称量化，使用平均绝对激活统计；不包含权重clipping搜索或TinyChat内核。

原文: [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) · Section 3.2: activation-aware scaling search; official auto_scale.py _search_module_scale.

## 2. 核心公式

$$
a_j=\operatorname{mean}_i|X_{ij}|,\quad s_j=\frac{\max(a_j^\alpha,10^{-4})}{\sqrt{\max_j s^{raw}_j\min_j s^{raw}_j}},\quad \widehat W=Q(Ws)/s,\quad \alpha^*=\arg\min_{\alpha\in G}\operatorname{mean}(X\widehat W^T-XW^T)^2.
$$

## 3. 公式到代码

- `group_quantize(weight: Tensor, bits: int, group_size: int) -> Tensor` — Detached dequantized weights under asymmetric groupwise quantization.
- `awq_search(x: Tensor, weight: Tensor, bits: int, group_size: int, alphas: list[float]) -> tuple[Tensor, Tensor, float, float]` — Best input-channel scales, dequantized weights, exponent and output MSE.

每个候选指数使用同一份原始权重，先乘通道scale、分组量化、再除回scale。用校准输入经过线性层的输出MSE选择候选，不能用权重自身MSE替代。

## 4. 本题约定

x:(N,D)，weight:(O,D)，D须被group_size整除。bits整数2..8，alphas非空且在[0,1]。组沿每行连续输入通道划分，scale=max((max-min)/(2^bits-1),1e-5)，zero=clip(round(-min/scale),0,2^bits-1)，ties-to-even。缩放raw先clamp1e-4，再除以sqrt(max*min)。按输入alpha顺序打破并列误差。返回(scale,反量化weight,alpha,float MSE)，离线校准不传梯度。

## 5. 交互图

16个确定性校准样本含8倍显著通道，3×4权重用3bit分组量化；扫描0到1的11个alpha并比较首个alpha=0基线和选中候选。

T3 使用实际待测函数生成实验，网页参考图使用参考实现。数值来自教学样本，不表示GPU吞吐或完整模型质量。

## 6. 运行方式

```bash
paperleet test ai_infra/awq --impl problems/ai_infra/awq/starter.py
paperleet test ai_infra/awq --impl problems/ai_infra/awq/solution.py
paperleet mutate ai_infra/awq
```

## 7. 延伸

结合所引章节，解释本题保留了哪些维度、状态或调度约束，再扩展实验样本并保持已有契约。更大模型、真实设备与通信代价需要单独测量。
