# SGLang Prefix KV Cache and Leaf-Only LRU — Solution notes

## Derivation and implementation

Match consecutive tokens from the root, never arbitrary suffixes. Copy the cache and update timestamps. When full, remove the oldest unprotected node with no children; an ancestor becomes evictable only after its children disappear.

$$
h(x)=\max\{j:x_{:j}\in C\},\qquad e=\arg\min_{p\in\operatorname{leaves}(C)\setminus P}(\operatorname{lastUse}(p),p).
$$

## Boundary and gradient ownership

cache maps nonempty token tuples to integer last-use times and must be prefix-closed. prompt is a nonempty tuple of nonnegative ints; capacity counts nonroot token nodes. clock is at least existing timestamps. pinned names cached nodes whose ancestors are also protected. Protect the current prompt path too; break LRU ties lexicographically by token tuple. Return (new_cache, reused_tokens), never mutating input even on failure.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Replay six prompts sharing system prefixes with capacities seven and twelve; compare prefill token counts with no caching and verify node capacity.

A token-node expansion of radix-tree semantics modeling shared prefixes, LRU and active-request pins, without storing actual KV tensors.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| cached_prefill_tokens | 10 |
| uncached_prefill_tokens | 21 |
| reused_tokens | 11 |
| maximum_nodes | 10 |

## Common mistakes

- m1_evict_internal: Evicting a parent with live children destroys prefix closure.
- m2_ignore_pins: Evicting active prefixes invalidates running-request references.

[SGLang](https://arxiv.org/abs/2312.07104) · Section 3: RadixAttention, longest-prefix reuse, leaf-first LRU and reference counters.
