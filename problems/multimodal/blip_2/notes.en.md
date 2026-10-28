# BLIP-2 maximum-query image-text matching — Solution

Build normalized similarities of shape [B,Q,N], then reduce Q. Each text selects its own winning query; averaging queries beforehand loses this slot structure. Away from ties, gradients flow only through the winning query and then through normalization. Finite-difference tests avoid nondifferentiable ties.

$$
S_{ij}=\frac{1}{\tau}\max_{q}\hat z_{iq}^{T}\hat t_j.
$$

The reference implementation follows the declared layout and reduction:

```python
def max_query_similarity(queries: Tensor, texts: Tensor, temperature: Tensor) -> Tensor:
    _floating(queries, texts, temperature)
    if queries.ndim!=3 or queries.numel()==0 or texts.ndim!=2 or texts.numel()==0 or queries.shape[-1]!=texts.shape[-1] or temperature.ndim!=0 or temperature.item()<=0:
        raise ValueError("expected (B,Q,D), (N,D), positive scalar temperature")
    q=F.normalize(queries,dim=-1,eps=1e-12);t=F.normalize(texts,dim=-1,eps=1e-12)
    return (q@t.T).max(dim=1).values/temperature
```

Common errors caught by mutation tests:

- Averages query scores instead of selecting the best query
- Uses raw query magnitudes

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Normalize Q-Former outputs, compute per-text maximum-query similarities, and route gradients through winning queries; omit Transformer and frozen large-model execution.

- [https://arxiv.org/abs/2301.12597](https://arxiv.org/abs/2301.12597)
