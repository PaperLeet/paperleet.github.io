# 19. DMD: Score-Difference Distillation

## 1. Exercise goal

Implement the distribution-matching generator gradient, paired regression, and separate fake-score update.

## 2. Interfaces and returns

### Return requirements

#### `score_difference` (required)

Return one frozen gradient-target Tensor with real_score/fake_score shape [B,...] and floating dtype/device. Broadcast the B weights per example without reducing any axes. Detach the complete result, so no gradient reaches real_score, fake_score or weight.

#### `dmd_loss` (required)

Return a scalar Tensor of shape (), using generated floating dtype/device. Both matching and paired-regression errors average over B and all coordinates; matching has a factor of 1/2. Keep the left-hand generated gradient path, but freeze the target constructed from generated-gradient and paired_target. Do not return the gradient tensor itself.

#### `fake_score_loss` (required)

Return a floating scalar Tensor of shape (), matching generated dtype/device, averaging denoising-score error over the batch and all coordinates. Detach generated inside the function while retaining model-parameter gradients; do not detach the loss. This interface does not explicitly stop gradients through noise or sigma.

#### Provided local helpers

- `_time`: Provided: return t reshaped from [B] to [B,1,...,1], a Tensor with the same rank as x; B counts samples and every non-batch axis is singleton. Preserve t dtype/device (not x dtype/device) and gradients. Do not mutate inputs; the result may share storage with t.

### Equations and interface notes

### Contract

- `score_difference(real_score: Tensor, fake_score: Tensor, weight: Tensor) -> Tensor`
- `dmd_loss(generated: Tensor, gradient: Tensor, paired_target: Tensor, regression_weight: float = .1) -> Tensor`
- `fake_score_loss(model, generated: Tensor, noise: Tensor, sigma: Tensor) -> Tensor`

We use a VE noise convention y=x+sigma*noise. The fake/real inputs are scores ∇y log q_sigma and ∇y log p_sigma, not epsilon predictions. The explicit positive weight is supplied by the caller. The matching surrogate has derivative weight*(s_fake-s_real)/numel; target scores are frozen. DMD retains paired regression; the Gaussian toy uses MSE as a CPU proxy for the paper’s image regression metric. Fake-score DSM must not train the generator. The linked DMD2 repository implements the shared score-gradient machinery, not evidence that this exercise is DMD2.

## Runnable experiment

**Provenance: PaperLeet teaching adaptation.** A shifted 2D Gaussian generator and trainable Gaussian fake score expose score signs, gradient isolation, and paired regression. This is not the paper’s Figure 3 bimodal source code; it isolates the same distribution-matching mechanism with analytic scores.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2311.18828
- https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/edm/edm_guidance.py

Anchors: arXiv v1 §3 distribution matching gradient and regression regularization; Figure 3 distribution matching illustration.

The second T3 experiment and the reference preview add a **Figure-3-inspired bimodal adaptation**. An analytic two-Gaussian student begins with both means near the right mode. Compare real-score-only, score-difference distribution matching, and matching plus paired MSE regression. Teacher and fake scores are exact mixture scores here, so this experiment isolates generator gradients; the first Gaussian T3 separately trains the fake score. Use noise sigma 1.5→0.6 over 600 CPU updates, target centers (-1.8,0)/(1.8,0), and component std .25. Report both mode proportions and covered-mode count. These settings and curves are PaperLeet’s reimplementation, not original Figure 3 data or source code.
