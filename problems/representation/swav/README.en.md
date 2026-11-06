# SwAV: balanced assignments and swapped prediction

## 1. Problem

Implement `sinkhorn_assignments(scores, epsilon=.05, iterations=3)` and `swapped_prediction_loss(scores1, scores2, assignments1, assignments2, temperature=.1)`. Turn prototype scores into balanced soft assignments, then predict each view's assignments from the other view's scores.

The source is [SwAV](https://arxiv.org/abs/2006.09882), Section 3. This exercise implements the single-process, two-view objective. The input tensors are already prototype scores; an image encoder, distributed synchronization, queue and prototype normalization are outside its scope.

## 2. Core equations

Let scores have shape `[B,K]`. Initialize a `K×B` matrix and normalize its total mass:

$$
Q=\exp(S^\top/\epsilon),\qquad Q\leftarrow Q/\sum_{ki}Q_{ki}.
$$

Repeat exactly `iterations` times, with prototype rows first and example columns second:

$$
Q_{ki}\leftarrow\frac{Q_{ki}}{\sum_jQ_{kj}}\frac1K,
\qquad Q_{ki}\leftarrow\frac{Q_{ki}}{\sum_lQ_{li}}\frac1B.
$$

Return assignments `A=(BQ).T`, detached from autograd. For prediction probabilities `P_v=softmax(S_v/temperature)`, use

$$
L=-\frac1{2B}\sum_{i,k}\left[\mathrm{sg}(A_{2,ik})\log P_{1,ik}
+\mathrm{sg}(A_{1,ik})\log P_{2,ik}\right].
$$

The assignment epsilon and prediction temperature have separate roles.

## 3. Equations to code

Perform Sinkhorn in log space with `logsumexp` to avoid overflowing exponentials. Subtract each normalization's log sum and `log(K)` or `log(B)` in the specified order; exponentiate after adding `log(B)` at the end. The result must match the finite iteration procedure, rather than an unspecified fully converged transport solution.

Use stable `log_softmax` for prediction. Cross the two views and average the two directional cross-entropies. Stop gradients on assignments in both functions, including assignments supplied directly by the caller.

## 4. Conventions

Inputs are finite nonempty `[B,K]` float32/float64 tensors. The loss's four tensors share shape, dtype and device. Inputs may be strided and must remain unchanged. Assignments are nonnegative with each row summing to one within `1e-5`. The loss is scalar; only scores receive gradients.

`epsilon` and `temperature` are finite positive Python numbers. `iterations` is a positive Python integer; bool and Tensor scalar controls are rejected. Validation helpers are supplied.

After every finite iteration, returned example rows sum to one. Prototype column sums approach `B/K` as Sinkhorn converges; finite iterations need not balance them exactly. `B` need not equal `K`, and `K=1` is valid. Tests cover finite-iteration scalar values, both marginal axes, permutation and global-shift invariance, extreme scores, analytic score gradients and stopped assignments.

## 5. Interactive plots and CPU lab

Fixed trigonometric `6×3` evidence tensors include a shared prototype bias. Compute assignments with epsilon `.2` and 40 Sinkhorn iterations, then train two independent `6×3` score tables for 80 Adam updates at learning rate `.04`, using prediction temperature `.3`. There are 36 learned parameters; initial predictions oppose the other view's evidence. Record loss and assignment entropy every eight steps, and plot the fixed assignment mass of each prototype.

The experiment calls both submitted functions. Independent scalar calculations check assignments and each loss. Its acceptance requires final cross-entropy minus the assignment-entropy lower bound below `.002`, a loss decrease above `.1`, scalar errors below `1e-11`, and prototype imbalance below `1e-8`. CPU float64, seed 42. This bounded experiment demonstrates balanced targets and swapped prediction; it does not measure learned image representations.

## 6. Run

```bash
paperleet test representation/swav --impl problems/representation/swav/solution.py
paperleet mutate representation/swav
```

Replace `--impl` with your completed starter. The complete CPU/WASM suite needs no image data, GPU or pretrained weights.

## 7. Further reading

Compare the assignment epsilon with the prediction temperature. Inspect a single Sinkhorn iteration to see why its final column normalization guarantees per-example probability rows but can perturb prototype balance. Swapping views changes which scores predict which targets, while detached assignments prevent an extra gradient path through clustering.

Sources: [paper](https://arxiv.org/abs/2006.09882) and [official training loop and distributed Sinkhorn](https://github.com/facebookresearch/swav/blob/main/main_swav.py). This exercise specializes the algorithm to one process and two views; its tests and vector experiment are independently authored.
