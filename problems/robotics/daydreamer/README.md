# DayDreamer 离散世界模型KL平衡

## 1. 问题

DayDreamer 离散世界模型KL平衡：实现下列接口并通过数值、边界和实验检查。

提取DayDreamer继承DreamerV2的KL balancing核心，固定alpha默认0.8；计算free-nats截断之前的KL，不实现RSSM或机器人在线训练。

## 2. 核心公式

$$
L=\alpha D_{KL}(\operatorname{sg}(q)\Vert p)+(1-\alpha)D_{KL}(q\Vert\operatorname{sg}(p)).
$$

## 3. 公式到代码

- `balanced_kl(posterior_logits: Tensor, prior_logits: Tensor, alpha: float=0.8) -> Tensor`

用log_softmax得到logq/logp。动力学项冻结整个q，包括概率与logq；表征项只冻结logp。alpha=1只更新prior，alpha=0只更新posterior。不能用一个未detach的KL乘权重代替。

## 4. 本题约定

posterior/prior logits为(B,K,C)，C>=2，alpha在[0,1]。K个独立分类潜变量的KL求和，然后按B平均。前向值与alpha无关，梯度分配依赖alpha。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

`experiment.py` 在固定输入上调用提交的全部接口，记录学习或采样曲线与末端指标。实验逻辑和阈值见 T3，输出同样用于可视化。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/daydreamer --impl problems/robotics/daydreamer/starter.py
paperleet mutate robotics/daydreamer
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/abs/2206.14176) — Appendix D: DreamerV2 world model, KL balancing 0.8
- [Verified source](https://arxiv.org/html/2206.14176)
- [Verified source](https://openreview.net/pdf/67aa53972119733780dfb9a38ef34181f4d2f261.pdf)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
