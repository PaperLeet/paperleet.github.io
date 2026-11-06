# BYOL: normalized regression and momentum targets

## 1. Problem

Implement `byol_loss(p1, p2, z1, z2, eps=1e-12)` and `ema_update(online, target, momentum=.99)`. The `p` tensors are predictions from two online views; the `z` tensors are target projections. Match each online prediction to the other view's target, then update target parameters using an exponential moving average of online parameters.

The source is [BYOL](https://arxiv.org/abs/2006.07733), Section 3.1 and Algorithm 1. Compared with the SimSiam exercise, this objective uses normalized squared distances and a separate target-parameter EMA. Targets receive no optimizer gradients.

## 2. Core equations

Use the official JAX helper's **squared-norm** floor:

$$
\bar v_i=\frac{v_i}{\sqrt{\max(\sum_d v_{id}^2,\epsilon)}}.
$$

Follow the official implementation's sum of directions, averaged over examples:

$$
L=\frac1B\sum_i\left[\|\bar p_{1i}-\overline{\mathrm{sg}(z_{2i})}\|_2^2
+\|\bar p_{2i}-\overline{\mathrm{sg}(z_{1i})}\|_2^2\right].
$$

Outside the epsilon region, a directional distance is `2−2*cosine`. Two opposite directional pairs give total loss `8`, not `4`. The squared-distance expression also defines zero-vector behavior.

$$
\theta_{target}^{new}=m\theta_{target}^{old}+(1-m)\theta_{online}.
$$

At `m=1`, retain the old target; at `m=0`, copy the online parameters.

## 3. Equations to code

Normalize along features, stop both target projections, and cross the views. Sum squared feature differences, then average examples; do not divide by two or by feature width. The EMA function receives parameter tensors, so it must not normalize them. Detach both inputs and return new storage. The training loop updates online parameters before updating the targets.

## 4. Conventions

Loss inputs share a finite nonempty `[B,D]` shape. EMA supports arbitrary matching nonempty parameter shapes. All inputs share float32/float64 dtype and device, may be strided, and must remain unchanged. Loss is scalar. Every EMA result has independent storage and no gradient, including at endpoints.

`eps` is a finite positive Python number applied to the squared norm. The default `1e-12` means a norm floor of `1e-6`. Momentum is a finite Python number in `[0,1]`; bool and Tensor scalar controls are rejected. Validation helpers are supplied.

Tests cover a scalar oracle, finite differences of both predictions, stopped targets, view swapping, paired permutations, positive scaling, near-zero vectors, and EMA weighting, endpoints and storage independence.

## 5. Interactive plots and CPU lab

Two `6×3` online tables share a `3×3` linear predictor: 45 online parameters. Fixed trigonometric vectors initialize the tables, and targets initially copy them. Run 80 actual SGD updates with learning rate `.18`; after every update, use `m=.9` for the corresponding target tables. Record loss and target lag every eight steps. Independent scalar arithmetic checks each loss and EMA update.

Acceptance requires loss below `.05` and below 5% of its initial value, total target movement above `.1`, scalar loss error below `1e-11`, and EMA error below `1e-12`. CPU float64, framework seed 42. This table experiment demonstrates predictions interacting with moving targets; it is not shared-encoder image pretraining or a universal no-collapse guarantee.

## 6. Run

```bash
paperleet test representation/byol --impl problems/representation/byol/solution.py
paperleet mutate representation/byol
```

Replace `--impl` with your completed starter. The browser runs the complete CPU/WASM suite without image data, a GPU or pretrained weights.

## 7. Further reading

Compare SimSiam's stopped targets with BYOL's moving targets. Detaching leaves forward values unchanged but determines gradient ownership. Explore how momentum changes target lag, and distinguish online predictions from target encoder parameters: averaging prediction outputs is not a parameter EMA.

Sources: [paper](https://arxiv.org/html/2006.07733v3), [official normalization and regression helper](https://github.com/google-deepmind/deepmind-research/blob/master/byol/utils/helpers.py), and [official symmetric objective/training loop](https://github.com/google-deepmind/deepmind-research/blob/master/byol/byol_experiment.py). Exercise code, tests and table experiment are independently authored.
