# GPTQ Cholesky 误差补偿量化

## 1. 问题

按固定列顺序量化权重，并用逆Hessian的Cholesky因子补偿剩余列误差。

给定上三角Cholesky因子的单块GPTQ核心；量化网格固定，Hessian估计、阻尼与分解由题外提供。

原文: [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) · Section 4, Algorithm 1; official gptq.py fasterquant inner column loop.

## 2. 核心公式

$$
H^{-1}=U^TU,\qquad q_j=\Delta\,\operatorname{clip}(\operatorname{round}(w_j/\Delta),q_{min},q_{max}),\qquad e_j=(w_j-q_j)/U_{jj},\qquad W_{:,j:}\leftarrow W_{:,j:}-e_jU_{j,j:}.
$$

## 3. 公式到代码

- `gptq_quantize(weight: Tensor, upper: Tensor, scales: Tensor, qmin: int = -7, qmax: int = 7) -> Tensor` — Dequantized weights after sequential second-order error compensation.

拷贝工作权重，每列先保存网格化结果，再用归一化残差更新剩余权重。必须除以Cholesky对角而不是Hessian对角；补偿符号是减法。

## 4. 本题约定

weight:(O,D)，upper:(D,D)，scales:(O,)；upper严格上三角以下为零、对角>0，表示H^{-1}=U^T U。round为ties-to-even，整数界闭区间。从左到右量化，补偿包含当前列与所有后列。返回反量化float权重，不修改输入；离线PTQ不传梯度。

## 5. 交互图

使用两维相关校准输入和两行权重，扫描固定网格间距；比较GPTQ与独立逐元素round-to-nearest的输出重建MSE。

T3 使用实际待测函数生成实验，网页参考图使用参考实现。数值来自教学样本，不表示GPU吞吐或完整模型质量。

## 6. 运行方式

```bash
paperleet test ai_infra/gptq --impl problems/ai_infra/gptq/starter.py
paperleet test ai_infra/gptq --impl problems/ai_infra/gptq/solution.py
paperleet mutate ai_infra/gptq
```

## 7. 延伸

结合所引章节，解释本题保留了哪些维度、状态或调度约束，再扩展实验样本并保持已有契约。更大模型、真实设备与通信代价需要单独测量。
