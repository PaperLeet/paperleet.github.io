# Qwen2-VL 分段 MRoPE — 题解

将 R 个频率分成三个连续区间。频率索引 r 保持不变，只选择其所属轴的位置值；不要给每个轴重新从第一个频率开始。旋转是二维正交变换，配对坐标是 r 和 r+R，因此保持范数；使用负位置可精确逆转。等宽配置让连续分段与后续交错布局的差别可直接比较。

$$
a(r)=\lfloor r/(R/3)\rfloor,\quad\theta_{nr}=p_{a(r),n}\omega_r,\quad y=[x_A\cos\theta-x_B\sin\theta;\ x_B\cos\theta+x_A\sin\theta].
$$

参考实现按上述布局和归约逐步计算：

```python
def chunked_mrope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    _floating(x,positions,inv_freq)
    if x.ndim!=2 or x.numel()==0 or x.shape[1]%6 or positions.shape!=(3,x.shape[0]) or inv_freq.shape!=(x.shape[1]//2,):
        raise ValueError("expected (N,2R), (3,N), (R,) with positive R divisible by 3")
    half=x.shape[1]//2
    phases=torch.stack([positions[r // (half//3)]*inv_freq[r] for r in range(half)],dim=-1)
    a,b=x[:,:half],x[:,half:]
    return torch.cat((a*torch.cos(phases)-b*torch.sin(phases),b*torch.cos(phases)+a*torch.sin(phases)),-1)
```

错误实现检测覆盖这些常见问题：

- Uses the other MRoPE axis-allocation layout
- Flips the sine sign in the first half

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

连续频率分段的三轴旋转位置编码；使用等宽三轴的教学配置，不包含动态分辨率编码器。

- [https://arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191)
