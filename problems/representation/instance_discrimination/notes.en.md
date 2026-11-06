# Implementation notes

Read these after attempting the exercise.

Gather bank rows with `memory.detach()`, compute dot products, and work in the log domain to avoid exponential overflow. Sum the K negative terms and average only over B. Do not drop `log_z` or replace `K/N` with `1/N`. Unlike softmax, shifting all model log-likelihoods changes binary NCE. Return an independent detached bank copy after indexed EMA and normalization.

Queries `[B,D]`, bank `[N,D]`, candidate IDs `[B,K+1]`, K≥1. Floats are finite float32/64; IDs are int64 and in range on the same device. Features normally come from a normalized encoder, but this function consumes their supplied values without additional normalization. `log_z` is a fixed finite Python scalar; the paper/official code estimate Z once, which is outside this exercise. Uniform noise draws may repeat and may hit the positive ID. Memory gradients stop; query gradients remain. Memory updates require distinct IDs, momentum in `[0,1]`, and norm floor `eps=1e-12`; untouched rows stay exactly unchanged and zero mixtures remain zero.

The maintained mutations isolate these faults:

- Use candidate softmax instead of original binary NCE.
- Use 1/N instead of K/N noise prior.
- Let the persistent memory receive autograd gradients.
- Average the noise terms rather than summing them.
- Swap old/new weights in the memory EMA.
- Store unnormalized EMA rows.
- Ignore the fixed partition constant.

## Actual CPU calibration

Train six normalized six-dimensional instance vectors for 80 SGD steps at learning rate `.35`, using T=`.2`, fixed Z=`exp(5)+5` and bank momentum `.95`. All five other toy IDs are deterministic negative candidates; this schedule illustrates the objective and is not the original stochastic sampler. The lab invokes both submitted functions. T3 requires loss below `.8` and 40% of its initial value, own-bank cosine above `.99`, and unit, detached bank rows. No image encoder or downstream benchmark is reproduced.

All data and initial parameters are deterministic; the test fixture also seeds Python/NumPy/PyTorch to 42. Native calibration uses the repository's PyTorch 2.14.1 CPU environment on Apple silicon. Tests use the same full training loop in the NumPy autograd adapter and exported WASM bundle. The gates validate numerical behavior and learning on this synthetic task, not a published paper benchmark.

Source: [paper](https://arxiv.org/abs/1805.01978) and [official implementation](https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/NCECriterion.py).

Observed native reference metrics: initial_loss=2.34368; final_loss=0.661522; initial_alignment=0.574979; final_alignment=0.999562.

A nonzero EMA mixture shorter than the supplied epsilon must be divided by that epsilon. Fixing the norm floor at 1e-12 silently changes valid nondefault inputs; the dedicated mutant and scalar case cover this.

The momentum-one shortcut must not return an alias. Selected non-unit rows still need normalization, and editing any result row must leave the input bank unchanged.
