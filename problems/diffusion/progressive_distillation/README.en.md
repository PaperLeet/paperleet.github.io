# 26. Progressive Distillation: Two Teacher Steps to One

## 1. Exercise goal

Derive the exact clean prediction that makes one student DDIM step match two teacher steps, then learn it.

## 2. Interfaces and returns

### Return requirements

#### `ddim_step` (required)

Return the sample Tensor at the less noisy time s, matching noisy shape [B,...] and floating dtype/device; B is batch size and trailing axes are sample coordinates. Return only the updated sample, not clean_prediction or coefficients. Keep differentiable paths; time coefficients [B] broadcast over sample axes.

#### `distilled_target` (required)

Return the clean-prediction target Tensor that makes one student step match the teacher endpoint, with noisy shape [B,...] and floating dtype/device. Detach the entire result: no gradient reaches noisy, teacher_endpoint or time coefficients. This is neither the teacher endpoint itself nor the first teacher prediction.

#### `distillation_loss` (required)

Return a floating scalar Tensor of shape (), matching prediction dtype/device. Broadcast each sample weight over coordinates, then average weighted squared errors over the batch and all coordinates; do not divide by the weight sum. Preserve prediction gradients and detach target internally; weight is not explicitly detached.

#### `teacher_pair` (provided)

Return the endpoint Tensor after exactly two teacher DDIM steps, matching noisy shape [B,...] and floating dtype/device. Return only the sample at end, not the midpoint or a pair of predictions. The output is gradient-free and noisy is unchanged.

#### Provided local helpers

- `_time`: Provided: return t reshaped from [B] to [B,1,...,1], a Tensor with the same rank as x; B counts samples and every non-batch axis is singleton. Preserve t dtype/device (not x dtype/device) and gradients. Do not mutate inputs; the result may share storage with t.

### Equations and interface notes

### Contract

- `ddim_step(noisy: Tensor, clean_prediction: Tensor, alpha_t: Tensor, sigma_t: Tensor, alpha_s: Tensor, sigma_s: Tensor) -> Tensor`
- `distilled_target(noisy: Tensor, teacher_endpoint: Tensor, alpha_t: Tensor, sigma_t: Tensor, alpha_s: Tensor, sigma_s: Tensor) -> Tensor`
- `distillation_loss(prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor`
- `teacher_pair(teacher, noisy: Tensor, t: Tensor, midpoint: Tensor, end: Tensor) -> Tensor` (provided)

Inputs alpha and sigma are amplitudes, with alpha²+sigma²=1; they are not cumulative variance alpha_bar. Require sigma_t>0 and a strictly lower end time so alpha_s-(sigma_s/sigma_t)alpha_t is nonzero. The teacher performs two denoising evaluations, including a new prediction at the intermediate state. Freeze its solved target. Weight is supplied per sample (the toy uses 1); parameterization rotations remain in module #7.

## Runnable experiment

**Provenance: PaperLeet teaching adaptation.** A fixed affine denoiser teacher takes two cosine-schedule DDIM steps; a tiny linear student learns the solved target and matches the endpoint in one evaluation. Demonstrates the exact distillation algebra, not image quality or repeated multi-stage halving.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2202.00512
- https://github.com/google-research/google-research/tree/0a1b0fe7298a58bca6dcb47112379d829bc0c0a5/diffusion_distillation

Anchors: arXiv v1 §3 Progressive distillation; Algorithm 2 and Appendix B distillation target.
