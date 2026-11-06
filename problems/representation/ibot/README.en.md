# iBOT: centered teacher and masked patch distillation

## 1. Problem

[iBOT](https://arxiv.org/abs/2111.07832) combines image-level self-distillation with masked patch prediction from an online teacher. Its patch targets are distributions over learned prototypes, rather than fixed tokenizer IDs. This exercise isolates the **same-view, single-global-crop patch branch** on supplied logits.

Implement `teacher_patch_targets(teacher_logits, center, teacher_temp=.04)` and `ibot_patch_loss(student_logits, teacher_logits, mask, center, student_temp=.1, teacher_temp=.04)`. Student and teacher logits are `[B,N,K]`, the center is `[K]`, and Boolean mask `[B,N]` selects student patches to supervise. Return detached `[B,N,K]` teacher probabilities and one scalar loss.

## 2. Core equations

Center the teacher's prototype scores before temperature-softmax, and stop the complete teacher branch:

$$
q_{bnk}=\operatorname{sg}\!\left[\operatorname{softmax}_k\left(\frac{t_{bnk}-c_k}{\tau_t}\right)\right],\qquad
\log p_{bnk}=\operatorname{logsoftmax}_k\left(\frac{s_{bnk}}{\tau_s}\right).
$$

The [official `iBOTLoss.forward`](https://github.com/bytedance/ibot/blob/main/main_ibot.py) computes a masked mean **within each image**, then averages all images. With $M_b=\sum_n m_{bn}$:

$$
\ell_{bn}=-\sum_k q_{bnk}\log p_{bnk},\qquad
\mathcal L=\frac1B\sum_b\frac{\sum_n m_{bn}\ell_{bn}}{\max(M_b,1)}.
$$

An image with zero masked patches contributes zero and still counts in `B`. For a selected student score, the gradient is $(p_{bnk}-q_{bnk})/(B M_b\tau_s)$. Teacher logits and center have no gradient. Matching `p=q` yields teacher entropy, generally a positive value; zero cross entropy is not the target for a soft teacher.

## 3. From equations to code

1. Subtract the `[K]` center, divide by teacher temperature, apply final-axis softmax and detach the entire target.
2. Divide student logits by student temperature and use stable final-axis log-softmax.
3. Multiply by teacher probabilities, sum over prototypes and negate.
4. Mask the patch losses. Divide each image's sum by its masked count clamped to one, then average all images.

Use the helper in this file; keep supplied validation. Do not replace teacher probabilities with argmax labels, swap image correspondence or average over all masked patches globally. The CLS branch's cross-view pairing is not the pairing of this patch branch.

## 4. Exercise conventions

- Matched nonempty `[B,N,K]` student/teacher tensors and `[K]` center share float32/float64 dtype, device and finite values. Mask is Boolean `[B,N]` on the same device. Preserve inputs, dtype/device and support strided views.
- Both temperatures are finite positive Python int/float; reject bool and Tensor scalars. Defaults follow the nominal official temperatures; no temperature schedule is applied here.
- The teacher target, including the center, is stopped. Visible student patches receive zero gradients; empty whole masks return differentiable zero. `K=1` has target probability one and zero CE.
- The center is supplied and fixed during this call. Center EMA, teacher EMA, multi-crop averaging, shared projection-head architecture and the image-level CLS objective remain outside this module. `[K]` simplifies the official broadcast center shape `[1,1,K]`.
- Student score offsets and teacher score offsets may vary per patch without changing probabilities. Adding the same prototype vector to teacher logits and center also preserves targets.

Tests use independent scalar probabilities and CE, exact and finite-difference student gradients, stopped teacher/center checks, unequal and empty mask rows, consistent permutations, soft-target entropy, singleton vocabulary and extreme scores.

## 5. Interactive charts and CPU experiment

The Lab directly optimizes a **`3×4×4` student logit table** against fixed deterministic teacher logits and a center. Images contain `1,3,0` selected patches. Adam performs **120 actual updates at `.08`**, with student/teacher temperatures `.7/.5` to make the soft distributions easy to inspect. Teacher and center are included in the optimizer as a gradient-boundary check; they must remain fixed.

Charts show your CE alongside the teacher entropy floor, independently computed image-balanced KL, and teacher/student probabilities for one masked patch. Reference CPU KL decreases from about `.10581` to `1.8e-7`; CE approaches `.81916`, not zero. T3 requires KL below `1e-5` and `.001` of baseline, oracle agreement within `1e-12`, and zero drift in teacher, center and visible student logits. This bounded synthetic distillation task does not train an encoder, update an online tokenizer or reproduce iBOT benchmark results.

## 6. Running

Fill in both functions and select **Run tests**. The unchanged tests and real optimization loop run in CPU/WASM without images, GPUs or pretrained weights.

```bash
paperleet test representation/ibot --impl problems/representation/ibot/solution.py
paperleet mutate representation/ibot
paperleet ci --problem representation/ibot
paperleet vizdata representation/ibot
```

## 7. Further reading

Read iBOT Section 3 and both branches of the official `iBOTLoss`. Why does patch matching use the same global view while the class-token objective uses cross-view terms? What distinct roles do teacher centering and temperature play? Why must teacher weights and center be updated outside gradient descent on this objective?

Compare [BEiT](https://arxiv.org/abs/2106.08254) hard token targets and [DINO](https://arxiv.org/abs/2104.14294) image-level self-distillation. Derive when image-balanced masked means equal the global masked-token mean and when they differ.
