# Qwen3-VL interleaved MRoPE — Solution

Cycle time, height, and width across frequency indices so each axis spans multiple frequency bands. Multiply each selected position by its matching frequency and use half-split rotations. Different axis positions make contiguous allocation observably wrong; one-dimensional RoPE tests would miss this, so the oracle uses distinct coordinates per axis.

$$
a(r)=r\bmod3,\quad\theta_{nr}=p_{a(r),n}\omega_r,\quad y=[x_A\cos\theta-x_B\sin\theta;\ x_B\cos\theta+x_A\sin\theta].
$$

The reference implementation follows the declared layout and reduction:

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

Common errors caught by mutation tests:

- Uses the other MRoPE axis-allocation layout
- Flips the sine sign in the first half

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Interleaved frequency-axis allocation and half-split rotary pairs in an equal-axis teaching configuration; exclude DeepStack and the full video model.

- [https://arxiv.org/abs/2511.21631](https://arxiv.org/abs/2511.21631)
