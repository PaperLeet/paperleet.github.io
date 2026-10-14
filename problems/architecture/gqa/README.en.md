# GQA and KV Head Conversion

Grouped-query attention shares one key/value head among a group of query heads, connecting MHA and MQA. Checkpoint conversion averages KV projections within groups; the paper then continues training. Read [GQA, EMNLP 2023, Sections 2.1–2.2 and Figures 1–2](https://aclanthology.org/2023.emnlp-main.298.pdf). This module implements grouping and conversion without model downloads or uptraining.

## Your core module

Implement `mean_pool_kv_heads` and `grouped_query_attention`. Scaled dot-product attention is a useful prerequisite; the new pieces are group routing and accumulated shared-head gradients.

| Input/output | Convention |
| --- | --- |
| `q` | `[B,H,Q,D]`, H query heads |
| `k`, `v` | `[B,G,K,D]`, `[B,G,K,Dv]`, G KV heads |
| `output`, `weights` | `[B,H,Q,Dv]`, `[B,H,Q,K]` |
| `weight` | One K or V projection `[H,D_model,D_head]`; return `[G,D_model,D_head]` |
| Groups | Positive H and G; H divisible by G; contiguous query groups |
| Mask | Optional boolean tensor broadcastable to `[B,H,Q,K]`; True allows attention |

Inputs are finite float32/float64 CPU tensors. Preserve precision and input values. K is at least one; Q may be zero and need not equal K. Key width D and value width Dv may differ.

## Convert projection heads

With $r=H/G$, separately convert each K or V projection:

$$\overline W_g=\frac{1}{r}\sum_{j=0}^{r-1}W_{gr+j}.$$

Average only consecutive heads, not feature axes. G must be a positive Python integer dividing H; invalid group counts raise `ValueError`. Four scalar heads `[0,2,10,14]` become `[1,12]` for G=2. G=H preserves every projection; G=1 makes one shared head.

## Route contiguous groups

Query head h reads KV head $g(h)=\lfloor h/r\rfloor$. For H=4 and G=2, the mapping is `[0,0,1,1]`, not `[0,1,0,1]`. Scale by the key feature width:

$$A_h=\operatorname{softmax}_{keys}\left(Q_hK_{g(h)}^\top/\sqrt D+M_h\right),\qquad O_h=A_hV_{g(h)}.$$

Return both outputs and weights. An entirely masked row has exact zero output and weights, with finite zero gradients. That padding convention is explicit to this exercise. Causal visibility must be passed as a mask; no autoregressive alignment is implicit. Invalid head counts or incompatible dimensions raise `ValueError`.

Sharing preserves differentiability: each KV head accumulates contributions from every query head in its group. Projection mean pooling distributes each pooled gradient equally across the original heads. Differentiable expansion, indexing and loops are all acceptable; no particular kernel is required.

## Toy experiment

The supplied experiment converts eight synthetic projection heads over six tokens to G=1,2,4,8. Charts show persistent KV element counts relative to MHA, and converted-output RMSE relative to the original MHA output. Your submitted functions produce the graded experiment; the reference Lab preview is labeled separately.

Cache counts exclude queries, attention matrices and temporary expanded computation tensors. The storage fraction G/H is not a runtime speedup. Synthetic output error is not language-model quality: the paper also uses uptraining, which this exercise does not run. G=H must preserve the original output and G=1 stores one eighth of the KV elements; intermediate error need not be monotonic.

## Validation

T0 covers shape, precision, strides and empty queries. T1 uses hand-computed routing and pooling. T2 uses scalar softmax, an independent library call per query head, analytical derivatives and numerical Jacobians to check masks, shared gradients, MHA/MQA endpoints and invalid groups. T3 runs the complete synthetic experiment. Browser CPU/WASM and server judging use identical tests and tolerances.
