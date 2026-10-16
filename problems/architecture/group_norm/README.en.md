# GroupNorm: Grouped Statistics and Batch Independence

## 1. Problem

### Return requirements

#### `group_norm` — Required

Return one Tensor with exactly `x`'s `[N,C,*spatial]` shape, dtype and device. Preserve gradients to `x`, `weight` and `bias`; do not change any input. Normalize within each sample and contiguous channel group, then apply the separate affine parameters of each channel.

### Formula and conventions

Implement one function from [Group Normalization](https://arxiv.org/html/1803.08494v3#S3), by Yuxin Wu and Kaiming He, ECCV 2018. Group membership follows Equation (7); Figure 3 shows the reshape-and-normalize structure. Read the formula and derive the tensor operations before opening the solution.

## 2. Core formula

For a group containing $m$ channel/spatial values from **one sample**:

$$
\mu=\frac1m\sum_j x_j,\qquad
v=\frac1m\sum_j(x_j-\mu)^2,\qquad
y_i=\gamma_{c(i)}\frac{x_i-\mu}{\sqrt{v+\varepsilon}}+\beta_{c(i)}.
$$

The variance denominator is $m$, not $m-1$. The group owns the moments; each channel owns its scale and bias. The batch axis is never a reduction axis. [PyTorch's GroupNorm documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html) also specifies population variance and per-channel affine parameters.

## 3. Formula to code

Expose a group axis by reshaping channels into `[G,C/G]`. Keep the sample and group axes while reducing the within-group channels and all spatial axes. Restore the original shape before broadcasting `weight` and `bias` along the channel axis.

For channels `[0,2]`, `[4,6]`, `[8,10]`, `[12,14]` and `G=2`, the group means are 3 and 11; both variances are 5. Different channel weights must still produce different affine transforms within the same group.

## 4. Exercise contract

| Item | Contract |
| --- | --- |
| Activations | `[N,C,*spatial]`, rank at least 3; `N>=0`, positive `C` and spatial sizes |
| Affine | `weight [C]`, `bias [C]`, matching dtype/device |
| Groups | Integer `G` excluding bool, `1<=G<=C`, `C%G==0`; invalid counts raise `ValueError` |
| Precision | float32/float64; tests use well-scaled finite activations and finite positive epsilon |
| Statistics | Contiguous channels within each sample; population variance; epsilon inside sqrt |
| Differentiation | Keep both moments connected to the input graph |

This is a functional, stateless exercise. The paper's image layout is extended to one or more spatial axes. Empty batches retain their shape. Singleton groups follow the mathematical formula: normalized values and input/scale gradients are zero, so the output equals the channel bias. This explicit singleton convention is tested directly rather than delegated to a library's input-size guard.

## 5. Experiment

Keep an anchor feature map fixed while adding companions with controlled offsets and scales. The first chart measures changes in that anchor's output: your GN should remain unchanged, while the supplied training-batch-statistics baseline responds to the companions. A second chart compares the same sample at `G=1`, `G=2` and `G=C`.

**Provenance:** a small synthetic teaching adaptation. It demonstrates normalization axes and sample independence. It does not train ImageNet/COCO models or reproduce their accuracy or speed. Submitted-run charts call your implementation; reference previews use the provided reference.

## 6. Run

Complete the web editor and run all tests, or use:

```bash
paperleet test architecture/group_norm --impl problems/architecture/group_norm/solution.py
paperleet mutate architecture/group_norm
```

## 7. Further study

T0 checks shape, precision, strided inputs and preservation. T1 is hand-computed. T2 adds scalar and library oracles, analytical gradients, finite differences, epsilon/singleton boundaries and group validation. T3 measures your implementation's batch-context behavior.

Explain why `G=1` uses layer-wide statistics while `G=C` uses instance-wise statistics. With spatial dimensions, layer-wide statistics include channels **and space**, unlike Transformer token-local LayerNorm. Then derive the gradient coupling between different channels in the same group.
