# REPA Representation Alignment

Implement the loss that connects noisy diffusion features to clean semantic representations in [REPA](https://arxiv.org/abs/2410.06940), ICLR 2025 Oral. A frozen image encoder supplies clean-image targets; a trainable projector maps the diffusion model's hidden patches to the same feature width.

## Implementation requirements

### Return requirements

#### `repa_alignment` (required)

Return one scalar floating `Tensor`, shape `()`, preserving feature dtype/device. It is negative patch cosine similarity averaged across B examples and N patches of `[B,N,D]` features, after reducing the D channels. Keep `projected` gradients, freeze `teacher`, and leave both inputs unchanged; matching patches whose norms are at least eps give -1, not zero.

#### `repa_objective` (required)

Return one scalar floating `Tensor`, shape `()`: all-element mean velocity MSE plus `weight` times the scalar alignment loss. Preserve the common input dtype/device and gradients to `prediction` and `projected`; detach both `velocity_target` and `teacher`. Return the combined loss only, not a tuple of components; leave inputs unchanged.

### Formula and conventions

For projected patches `p` and teacher patches `z`, both `[B,N,D]`, define

$$
\bar p_{bn}=\frac{p_{bn}}{\max(\lVert p_{bn}\rVert_2,\epsilon)},\quad
\bar z_{bn}=\frac{\operatorname{sg}(z_{bn})}{\max(\lVert z_{bn}\rVert_2,\epsilon)},\qquad
L_{align}=-\frac1{BN}\sum_{b,n}\bar p_{bn}^{\mathsf T}\bar z_{bn}.
$$

Implement `repa_alignment(projected, teacher, eps=1e-8)` and `repa_objective(prediction, velocity_target, projected, teacher, weight=0.5, eps=1e-8)`:

$$L=\operatorname{mean}((v_{pred}-\operatorname{sg}(v_{target}))^2)+\text{weight}\,L_{align}.$$

The negative sign matters: perfect agreement has alignment loss **−1**. Normalize each patch across its last dimension before reducing. Clamp the two norms separately; zero teacher patches contribute zero. Both teacher and velocity targets are fixed, even if their tensors arrive with `requires_grad=True`. Student and velocity-prediction gradients must remain connected.

## Contract

Features have equal, nonempty `[B,N,D]` shapes. Velocity tensors have equal nonempty shapes; they need not share the feature shape. Inputs are finite CPU float32/float64 tensors, including noncontiguous views. Preserve dtype/device and input values. `eps` must be finite and positive, and `weight` finite and nonnegative; validation is supplied. The projector, pretrained encoder and full diffusion model are outside this exercise.

Example: `p=[[[1,0],[0,2]]]`, `z=[[[3,0],[0,-1]]]` yields alignment loss `0`: one aligned patch and one opposite patch.

## Toy experiment

**Teaching adaptation.** A fixed synthetic clean feature table acts as the teacher. We add small noise to its input representation and optimize a 3×3 projector for 80 updates using your combined objective. Charts display the measured mean cosine and submitted loss. The cosine metric is recomputed independently using scalar arithmetic. This is a small demonstration of the objective, not the paper's image-generation benchmark.

The preview uses the reference; T3 uses your submitted functions. Tests also check scalar formula values, analytic derivatives, target freezing and independent patch scaling. [Author implementation](https://github.com/sihyun-yu/REPA/blob/main/loss.py).
