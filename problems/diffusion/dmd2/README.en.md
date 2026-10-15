# 20. DMD2: Two-Time-Scale and Adversarial Updates

## 1. Exercise goal

Implement the critic-to-generator update ratio and non-saturating GAN contribution on top of distribution matching.

## 2. Interfaces and returns

### Return requirements

#### `generator_due` (required)

Return a Python bool, not an integer or Tensor. Return True only for a positive completed critic-update count that lands on a generator-update interval; count 0 returns False. This function performs no update and has no tensor shape or gradients.

#### `adversarial_losses` (required)

Return (discriminator_loss, generator_loss), in that order, as two floating scalar Tensors of shape (). Preserve logits dtype/device; real and fake counts may differ. Average real and fake discriminator terms separately before adding them; average the generator term over fake logits. Do not detach logits here: both losses retain their respective gradient paths.

#### `generator_loss` (required)

Return one floating scalar Tensor of shape (), using generated/logits dtype/device. DM matching averages all generated elements with a factor of 1/2; the GAN term averages fake_logits. Freeze the target constructed from generated-gradient while retaining training gradients through generated and fake_logits. There is no paired-regression term and no tuple of losses.

#### Provided local helpers

- `_time`: Provided: return t reshaped from [B] to [B,1,...,1], a Tensor with the same rank as x; B counts samples and every non-batch axis is singleton. Preserve t dtype/device (not x dtype/device) and gradients. Do not mutate inputs; the result may share storage with t.

### Equations and interface notes

### Contract

- `generator_due(completed_critic_updates: int, critic_per_generator: int = 5) -> bool`
- `adversarial_losses(real_logits: Tensor, fake_logits: Tensor) -> tuple[Tensor, Tensor]`
- `generator_loss(generated: Tensor, gradient: Tensor, fake_logits: Tensor, gan_weight: float = .1) -> Tensor`

The critic/fake diffusion model updates more often than the generator (5:1 here). Counts are completed critic updates, so update 0 does not trigger a generator step. Logistic discriminator and generator losses have opposite signs on fake logits. Caller detaches fake data during critic training and freezes critic parameters, while retaining input gradients, during generator training. Unlike DMD, the generator objective has no paired teacher regression. This exercise covers the stabilization/real-data contribution, not the paper’s multistep backward simulation.

## Runnable experiment

**Provenance: PaperLeet teaching adaptation.** An unpaired shifted-Gaussian toy jointly trains a fake score and discriminator for five updates per student update. It demonstrates scheduling, GAN gradient direction, and removal of paired regression. Uses scalar Gaussian score parameters rather than pretrained diffusion features.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2405.14867
- https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/edm/edm_guidance.py

Anchors: arXiv v1 §3 two-time-scale updates and GAN loss; Official EDM compute_generator_clean_cls_loss and compute_guidance_clean_cls_loss.
