# GQA pitfalls

1. **Contiguous sharing differs from cyclic tiling.** H=8 and G=2 requires `[0,0,0,0,1,1,1,1]`. A plain `repeat` often interleaves the KV heads instead.
2. **Scale by feature width.** Divide logits by $\sqrt D$, not the head count, group count or group size.
3. **Mean conversion does not preserve general MHA outputs.** Averaging projections before softmax differs from averaging independent attention results. G=H is the unchanged endpoint. Synthetic deviation is not model accuracy.
4. **Shared KV gradients add.** For query heads $\mathcal H_g$ sharing KV head g, $\partial L/\partial K_g=\sum_{h\in\mathcal H_g}\partial L_h/\partial K_g$; the same applies to V. Do not detach or divide this gradient by group size.
5. **Projection pooling has the averaging factor.** Each original projection head gets 1/r of its pooled output gradient. Distinguish this conversion derivative from the sum in shared attention.
6. **Fully masked rows need an explicit convention.** Avoid softmax over all negative infinities, then zero forbidden weights. Incorrect mask broadcasting can leak future information. For unequal Q and K, the caller specifies visibility explicitly.
7. **A smaller cache does not promise faster exercise code.** A teaching implementation may expand KV for computation. The experiment counts persistent KV elements; it does not measure kernels, bandwidth or throughput.

Source: [GQA, Sections 2.1–2.2](https://aclanthology.org/2023.emnlp-main.298.pdf). Mean pooling converts a checkpoint; full uptraining is outside this exercise.
