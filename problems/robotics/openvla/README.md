# OpenVLA 动作token与分位数还原

## 1. 问题

OpenVLA 动作token与分位数还原：实现下列接口并通过数值、边界和实验检查。

复现官方ActionTokenizer的离散约定及动作反归一化；不加载7B模型。边界数组可缩小以便手算，默认论文规模为256个边界、255个中心。

## 2. 核心公式

$$
d=\sum_j\mathbf1[\operatorname{clip}(a,-1,1)\ge e_j],\quad t=V-d,\quad \hat a=(e_i+e_{i+1})/2,\ i=\operatorname{clip}(V-t-1,0,N-2).
$$

## 3. 公式到代码

- `encode_actions(actions: Tensor, edges: Tensor, vocab_size: int) -> Tensor`
- `decode_actions(tokens: Tensor, edges: Tensor, vocab_size: int) -> Tensor`
- `unnormalize_actions(actions: Tensor, q01: Tensor, q99: Tensor, mask: Tensor) -> Tensor`

编码使用末端词表的倒序编号；np.digitize默认right=False使恰在边界上的值进入右侧桶。最大边界会得到最后一个特殊编号，但解码夹到最后中心。反归一化只作用于mask指定的坐标，未指定坐标原样保留。

## 4. 本题约定

actions为任意非空浮点张量，edges为严格递增一维且两端为-1/+1，vocab_size>=边界数。token为int64；解码token必须在[0,V)，超出动作区间的词表token按官方clip映射端点中心。反归一化输入(B,D)，q01/q99为(D,)且q99>q01，mask为(D,)布尔。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

`experiment.py` 在固定输入上调用提交的全部接口，记录学习或采样曲线与末端指标。实验逻辑和阈值见 T3，输出同样用于可视化。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/openvla --impl problems/robotics/openvla/starter.py
paperleet mutate robotics/openvla
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) — Official action_tokenizer.py digitize / reversed IDs; OpenVLA.predict_action quantile unnormalization
- [Verified source](https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py)
- [Verified source](https://github.com/openvla/openvla/blob/main/prismatic/models/vlas/openvla.py)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
