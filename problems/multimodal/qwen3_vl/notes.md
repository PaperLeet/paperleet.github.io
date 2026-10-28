# Qwen3-VL 交错 MRoPE — 题解

与连续三段不同，轴序列为时间、高度、宽度并循环，使每个轴跨越多个频段。位置乘对应频率后，仍采用前后半维旋转。因为三轴位置一般不同，错误的连续分段不会被普通的一维 RoPE 测试发现；测试使用各轴不同坐标并逐坐标计算 oracle。

$$
a(r)=r\bmod3,\quad\theta_{nr}=p_{a(r),n}\omega_r,\quad y=[x_A\cos\theta-x_B\sin\theta;\ x_B\cos\theta+x_A\sin\theta].
$$

参考实现按上述布局和归约逐步计算：

```python
def interleaved_mrope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    _floating(x,positions,inv_freq)
    if x.ndim!=2 or x.numel()==0 or x.shape[1]%6 or positions.shape!=(3,x.shape[0]) or inv_freq.shape!=(x.shape[1]//2,):
        raise ValueError("expected (N,2R), (3,N), (R,) with positive R divisible by 3")
    half=x.shape[1]//2
    phases=torch.stack([positions[r % 3]*inv_freq[r] for r in range(half)],dim=-1)
    a,b=x[:,:half],x[:,half:]
    return torch.cat((a*torch.cos(phases)-b*torch.sin(phases),b*torch.cos(phases)+a*torch.sin(phases)),-1)
```

错误实现检测覆盖这些常见问题：

- Uses the other MRoPE axis-allocation layout
- Flips the sine sign in the first half

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

交错频率轴分配与半维配对旋转；使用等宽三轴的教学配置，不包含 DeepStack 或完整视频模型。

- [https://arxiv.org/abs/2511.21631](https://arxiv.org/abs/2511.21631)
