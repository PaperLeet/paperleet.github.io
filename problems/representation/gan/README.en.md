# GAN: Stable Adversarial and Non-saturating Losses

## 1. What problem does the paper solve?

GAN trains a discriminator against a generator. Section 3 introduces both the minimax game and a non-saturating alternative for stronger early generator gradients. This exercise selects that alternative explicitly and implements the losses directly from logits.

[原始来源 / Original source](https://arxiv.org/html/1406.2661v1#S3) · Section 3, Equation (1), Algorithm 1 and the non-saturating generator paragraph.

## 2. Core equations and return contract

$$L_D=\operatorname{mean}(\operatorname{softplus}(-r))+\operatorname{mean}(\operatorname{softplus}(f)),\qquad L_G=\operatorname{mean}(\operatorname{softplus}(-f)),\quad D=\sigma(\mathrm{logit}).$$

Required functions: `discriminator_loss`, `generator_loss`.

## 3. From equations to code

Real targets are one and fake targets zero. Average each discriminator batch separately, then add. The generator pushes fake logits upward. Express binary log losses with stable softplus rather than materializing probabilities.

## 4. Exercise conventions

Both functions consume finite float32/float64 logits and return a scalar with the same dtype/device, preserving every logit gradient. Inputs may have any shape, including scalars, but must be nonempty. Real/fake discriminator batches may have different shapes/counts and must share dtype/device. **Do not divide the sum by two** or smooth labels. Empty inputs or invalid dtypes raise ValueError. Do not modify inputs.

The specified `L_G=-mean(log D(fake))` is non-saturating. The original minimax `mean(log(1-D(fake)))` is a legitimate different objective, but does not satisfy this contract. Never detach logits inside the losses. The caller detaches generated samples before D during its update; during G updates, freeze D parameters while preserving the input-to-G gradient.

## 5. Experiment

Real data are Gaussian with mean 2 and variance 1. The generator is `G(z)=z+shift`, and D is an affine logit. The harness runs 240 actual D/G update pairs, batch 64, Adam lr=0.03 for both, seed=179, float64. Your losses drive backward passes with G detached during D steps and D parameters frozen during G steps. Charts show the learned location and both losses. Acceptance requires final location error<0.5. This does not reproduce image quality, FID or general GAN convergence.

## 6. Run your implementation

Complete the starter and run all tests in the workspace to unlock Notes. Maintainers can also run:

```bash
paperleet test representation/gan --impl problems/representation/gan/solution.py
```

## 7. Further exploration

Compare minimax and non-saturating gradients at fake logit=-1000. Change the D/G learning-rate ratio and inspect oscillations rather than assuming both losses decrease monotonically.
