# Kimi K3: Bounded Decay and Delta Memory

Implement two small operators from [Kimi K3: Open Frontier Intelligence, v2, §2.1.1, Eqs.1/5](https://arxiv.org/html/2607.24653v2#S2.SS1.SSS1). This exercise isolates the recurrent memory at the heart of Kimi Delta Attention. The report is pinned to the August 7, 2026 revision; the [official K3 repository](https://github.com/MoonshotAI/Kimi-K3) and [FlashKDA kernels](https://github.com/MoonshotAI/FlashKDA) provide implementation context.

## Your core module

### Return requirements

#### `bounded_decay` — Required

Return one floating retention Tensor `[B,T,H,K]`, matching logits: B batch size, T sequence length, H head count, K key channels. Values lie between exp(lower_bound) and 1; return retention factors, not their logarithms. Preserve logits' dtype/device and gradients into logits and log_scale; do not change inputs.

#### `recurrent_kda` — Required

Return `(output, final_state)` as floating tensors `[B,T,H,V]` and `[B,H,K,V]`. B is batch size, T token count, H head count, K key channels and V value channels (K and V may differ). Each output reads the state after that token's update; final_state can resume the scan. With T=0, output is `[B,0,H,V]` and final_state equals the supplied initial state, or zeros when it is None. Preserve shared dtype/device and gradient paths through nonempty scans to q/k/v/alpha/beta/initial_state; do not detach the returned state or mutate inputs.

### Formula and conventions

1. `bounded_decay` turns per-channel logits into retention factors.
2. `recurrent_kda` consumes prepared queries, keys, values and gates, returning every output and the final memory.

With $A$ the per-head log-scale and $g_{\min}<0$:

$$g=g_{\min}\sigma(e^A z),\qquad \alpha=e^g.$$

For each token, first forget, then correct the memory's prediction, then read:

$$
\bar S_t=\operatorname{Diag}(\alpha_t)S_{t-1},\qquad
S_t=\bar S_t+\beta_t k_t(v_t-k_t^\top\bar S_t)^\top,\qquad o_t=S_t^\top q_t.
$$

These operations are differentiable. Updating by `beta * outer(k, v)` alone accumulates conflicting values instead of correcting the stored prediction.

| Argument/result | Contract |
|---|---|
| `logits`, `log_scale` | `[B,T,H,K]`, `[H]`; `lower_bound=-5` by default |
| `q`, `k`, `v` | `[B,T,H,K]`, `[B,T,H,K]`, `[B,T,H,V]`; prepared vectors, possibly `K != V` |
| `alpha`, `beta` | `[B,T,H,K]`, `[B,T,H]`; retention and write factors, **not logits** |
| `initial_state` | Optional `[B,H,K,V]`; omitted means zeros |
| Returned output/state | `[B,T,H,V]`, `[B,H,K,V]`; output reads the current token's updated state |
| Empty sequence | Output has zero time length; final state equals the input state, or zeros |
| Numerical scope | Finite floating inputs; preserve shared dtype/device, autograd and all inputs; support noncontiguous views |

The scan performs no query scaling or key normalization. Its inputs are already prepared by the surrounding model. Input projections, short convolutions, head RMSNorm, output gating and the full hybrid KDA–MLA backbone are outside this exercise. State storage here follows the paper's key-by-value orientation; some kernels store its transpose. Gate endpoints 0/1 are accepted as useful analytic limits even though finite sigmoid parameters lie strictly between them.

## Toy experiment

The first chart redraws the log-decay comparison in Figure 3(a), with $A=0$, $g_{\min}=-5$, and logits from −10 to 10. It compares K3's bounded gate against Kimi Linear's negative-softplus gate. This is a **paper-derived figure**, not a GPU throughput reproduction.

The second chart is a separate **PaperLeet teaching adaptation**: write value 2 at key 0, write 7 at key 1, overwrite key 0 with 5, then retain one quarter of key 0 without writing. The exact outputs are `[2, 7, 5, 1.25]`. No training or downloads are needed. T3 calls the submitted functions; the preview and your actual run are displayed separately.

## Validation

T0 checks shapes, dtype/device, empty scans and input preservation. T1 evaluates the literal matrix equation independently. T2 checks the bounded gate, all six scan input gradients, segmented continuation, causal prefixes, zero writes and strided inputs. T3 validates both charts against independent scalar targets. No performance claims about CUDA kernels are tested on CPU.
