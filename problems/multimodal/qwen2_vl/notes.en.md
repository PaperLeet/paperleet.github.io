# Qwen2-VL chunked MRoPE — Solution

Split the R frequencies into three contiguous ranges. Keep the original frequency index r and select only its axis-specific position; do not restart frequencies for each axis. Each orthogonal rotation pairs coordinates r and r+R, preserving norm and inverting under negative positions. Equal axis widths isolate the difference from interleaved allocation.

$$
a(r)=\lfloor r/(R/3)\rfloor,\quad\theta_{nr}=p_{a(r),n}\omega_r,\quad y=[x_A\cos\theta-x_B\sin\theta;\ x_B\cos\theta+x_A\sin\theta].
$$

The reference implementation follows the declared layout and reduction:

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

Common errors caught by mutation tests:

- Uses the other MRoPE axis-allocation layout
- Flips the sine sign in the first half

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Contiguous frequency chunks for three-axis rotary positions in an equal-axis teaching configuration; exclude the dynamic-resolution encoder.

- [https://arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191)
