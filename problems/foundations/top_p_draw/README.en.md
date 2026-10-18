# Top-p: Draw Tokens from the Nucleus

## Problem

Implement `sample_top_p(logits, p, temperature, generator)` to return one vocabulary token ID per row. The supplied `nucleus_probs` handles temperature, stable softmax, nucleus selection and renormalization. Your task completes the path from logits to actual sampled tokens.

Read [Holtzman et al., ICLR 2020, §3.1, Eqs. (2–3)](https://arxiv.org/abs/1904.09751). The paper defines the nucleus and its normalized distribution. The inverse-CDF method and exact random-stream contract below are teaching choices, not claims that the paper mandates this implementation.

## Core equations

For vocabulary-order nucleus probabilities q, form C_i = sum_{j<=i} q_j and divide every C_i by the final C entry. Draw u uniformly in [0,1), then return the smallest vocabulary index i satisfying C_i > u. Each token owns the half-open interval [C_{i-1}, C_i). Zero-mass intervals never select a token.

## From equations to code

Call the provided helper, build the CDF along the last axis, normalize it by its final entry, and use one independent uniform per row. The number of CDF entries <= u is the selected vocabulary ID. Never return a sorted probability rank.

For q=[0,.5,0,.5], u=0 selects token 1; u=.5 selects token 3. The supplied helper uses the shortest nonempty prefix reaching cumulative mass >=p, ties favoring smaller vocabulary IDs. With four equal logits and p=.5, only IDs 0 and 1 remain.

## Contract

- Finite float32/64 CPU logits `[...,V]`, V>=1; empty leading dimensions are allowed. Finite scalar 0<p<=1 and temperature>0.
- Return int64 IDs with shape `logits.shape[:-1]`, on CPU. A single vector returns a scalar tensor. Do not modify inputs; no derivatives are required.
- `nucleus_probs` is provided in the starter and returns float64 probabilities in original vocabulary order. Keep that helper unchanged.
- Consume exactly `torch.rand(logits.shape[:-1], dtype=torch.float64, generator=generator)` once. The caller provides a CPU `torch.Generator`. Do not reseed, replace it with the global generator, or reuse a single draw across rows. An empty batch consumes no draws.
- Normalize the float64 CDF by its last entry to make its endpoint exactly 1. Select using strict C_i>u, including at exact interval boundaries.
- The exact stream is an exercise convention. Other correct categorical algorithms can use different random draws; this task deliberately practices inverse-CDF sampling.
- EOS, multi-step generation, NaN/Inf logits and temperature=0 are outside this single-token exercise.

## Interactive experiment

<viz id="toy_experiment" />

The plot shows 8192 tokens drawn by your function from synthetic logits, seed 42. It compares observed frequencies with an independently specified target [0,2/3,0,1/3]. Excluded tokens must never appear. Frequency error must be below .04; deterministic boundary tests remain exact. This is a small sampling experiment, not a text-quality benchmark.

## Run

```sh
paperleet test foundations/top_p_draw --impl problems/foundations/top_p_draw/solution.py
```

Use the starter in the workspace. T0 checks shape/dtype, T1 an independent scalar oracle, T2 stream ownership, boundaries and input preservation, and T3 actual learner sampling. Browser CPU/WASM uses the same suite and mutants.

## Further practice

Compare with the existing Top-p probability exercise (#89). Explain why sorting is needed to find the nucleus but the final draw must still return the original vocabulary ID. Then describe how a generation loop would stop different batch rows at EOS.
