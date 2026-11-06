# DINO: centered teacher and multi-view self-distillation

## 1. Problem

Implement `teacher_probabilities(teacher, center, temperature=.04)`, `dino_loss(student, teacher, center, student_temperature=.1, teacher_temperature=.04)`, and `update_center(center, teacher, momentum=.9)`.

[DINO](https://arxiv.org/abs/2104.14294), Section 3.1 and Algorithm 1, distills a teacher's centered, sharpened predictions across views. Teacher logits have shape `[2,B,K]` for two global views. Student logits have shape `[V,B,K]`, where `V>=2`; the first two views match the teacher's global views, and later views are additional crops. This exercise starts at logits and implements the loss and center update. Teacher-network parameter EMA, image crops, a backbone and downstream evaluation are outside its scope.

## 2. Core equations

For center `c` of shape `[K]`, teacher temperature `τt` and student temperature `τs`:

$$
q_{t,i}=\mathrm{sg}\left[\mathrm{softmax}\left((u_{t,i}-c)/\tau_t\right)\right],
\quad \log p_{v,i}=\mathrm{logsoftmax}(s_{v,i}/\tau_s).
$$

Use every teacher/student pair except its matching global view:

$$
L=-\frac1{2(V-1)B}\sum_{t=0}^1\sum_{\substack{v=0\\v\ne t}}^{V-1}
\sum_{i,k}q_{t,i,k}\log p_{v,i,k}.
$$

After computing the current loss, update the center from **raw teacher logits**:

$$
c^{new}=m c+(1-m)\frac1{2B}\sum_{t,i}u_{t,i}.
$$

The new center is detached and stored independently of both inputs.

## 3. Equations to code

Subtract the center before applying teacher temperature and softmax. Stop both teacher and center gradients; only student logits receive loss gradients. Use `log_softmax` for stable student log probabilities. Average over `2*(V-1)` view pairs, examples and soft-label cross-entropy, while summing the K class terms.

For `V=2`, pairs are `(teacher0,student1)` and `(teacher1,student0)`. For `V=3`, also include `(teacher0,student2)` and `(teacher1,student2)`: four pairs in total. Center averaging includes both teacher views and the batch axis, but never students, probabilities or centered logits. The caller installs the new center after evaluating the current objective.

## 4. Conventions

Inputs are finite, nonempty float32/float64 tensors with matching dtype/device. Teacher `[2,B,K]`, center `[K]`, student `[V,B,K]`, `V>=2`, `B>=1`, `K>=1`. Inputs may be strided and must remain unchanged. Probabilities preserve teacher shape; the loss is scalar; center updates return `[K]` with independent storage and no gradient even at momentum endpoints.

Temperatures are finite positive Python numbers; momentum is a finite Python number in `[0,1]`. Bool and Tensor scalar controls are rejected. Validation helpers are supplied. Temperatures are caller supplied: no epoch-dependent schedule is hidden inside these functions.

Tests compare independent scalar probabilities and losses, analytic student gradients and stopped teacher/center paths. They cover view-pair counting, matched-view exclusions, invariances, center weighting and endpoint storage.

## 5. Interactive plots and CPU lab

Two fixed `6×3` teacher views encode three groups plus a shared logit bias. A `3×6×3` student table has 54 trainable parameters. Run 80 SGD updates at learning rate `.6`, student temperature `.3` and teacher temperature `.15`; after each loss, update the center with momentum `.9`. Record cross-entropy and center norm every eight steps; also measure final student class agreement and the maximum center error.

Every step calls the submitted loss and center functions, and the submitted probability function is checked against an independent scalar oracle. The teacher remains fixed, so the center also has a closed-form EMA oracle. Acceptance requires final loss below `.08` and below 10% of its initial value, all 18 student predictions matching their synthetic classes, scalar loss/probability errors below `1e-11`, center error below `1e-12`, and no teacher gradients. CPU float64, seed 42. This table distillation lab demonstrates the objective and moving center; it does not train a ViT or establish image representation quality.

## 6. Run

```bash
paperleet test representation/dino --impl problems/representation/dino/solution.py
paperleet mutate representation/dino
```

Replace `--impl` with your completed starter. The complete CPU/WASM suite runs without image data, a GPU or pretrained weights.

## 7. Further reading

Centering acts across output coordinates, while sharpening controls the distribution's concentration. Analyze how same-view exclusions change a student's target mixture. Compare BYOL's teacher-parameter EMA with DINO's logit-center EMA: they update different quantities, and the full DINO method uses both.

Sources: [paper](https://arxiv.org/abs/2104.14294) and [official DINOLoss and training loop](https://github.com/facebookresearch/dino/blob/main/main_dino.py). The fixed-temperature, single-process tensor contract and toy experiment are independently authored teaching adaptations.
