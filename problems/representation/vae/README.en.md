# VAE: Reparameterization and the Bernoulli ELBO

## 1. What problem does the paper solve?

Auto-Encoding Variational Bayes jointly learns a probabilistic decoder and approximate posterior. Its pathwise estimator makes a sample differentiable through posterior parameters. This exercise selects the diagonal-Gaussian/Bernoulli case from [Section 3, Equations (9)–(10), Appendices B and C.1](https://arxiv.org/html/1312.6114v11#S3).

## 2. Core equations and return contracts

Write `logvar = log(sigma²)` and use caller-supplied standard-normal noise:

$$z=\mu+\exp(\tfrac12\operatorname{logvar})\odot\epsilon,$$

$$K_b=\tfrac12\sum_j[\mu_{bj}^2+\exp(\operatorname{logvar}_{bj})-1-\operatorname{logvar}_{bj}],$$

$$L=\frac1B\sum_b\left[\sum_f(\operatorname{softplus}(\ell_{bf})-x_{bf}\ell_{bf})+K_b\right].$$

- `reparameterize(mu, logvar, noise)` returns `[B,Z]`; all three inputs share that shape, with `Z>0`. Do not draw noise internally.
- `gaussian_kl(mu, logvar)` returns **nonnegative** KL to `N(0,I)`, with a positive sign in the loss, shape `[B]`, summed over latent dimensions.
- `negative_elbo(logits, target, mu, logvar)` returns a scalar in nats per observation. Logits/targets share nonempty `[B,F]`, posterior batch sizes agree, and target values lie in `[0,1]`.

Preserve matching float32/float64 dtype/device, input values and live gradients. Sampling and KL allow `B=0`; the loss requires `B>0`. Incompatible shapes raise `ValueError`.

## 3. From equations to code

Convert log variance to **standard deviation** for sampling. Sum the analytic KL over `Z`. Compute Bernoulli NLL stably from logits, sum over `F`, add KL once per observation, then average the batch. This minimizes the negative lower bound; it does not maximize the expression returned by your loss function.

## 4. Exercise conventions

Use one posterior sample and the original KL coefficient 1, without beta weighting, annealing or clipping probabilities. Feature sums and batch means are distinct: changing either changes the reconstruction/KL tradeoff. A correct forward KL value with a detached gradient is incorrect. The supplied noise remains differentiable as an explicit tensor input, even though the experiment draws it as a constant.

## 5. Experiment

A linear encoder produces two means and log variances; a linear Bernoulli decoder reconstructs two complementary six-bit patterns. Adam, learning rate 0.04, performs 120 real updates with 32 observations and seed 107. Your three functions supply sampling, the loss and KL diagnostics.

Charts show sampled negative ELBO, reconstruction NLL and KL. The mean of the final 20 losses must be below 80% of the initial loss, and the decomposition must agree. This is a synthetic teaching experiment, not MNIST reproduction, an exact marginal-likelihood measurement or proof of useful latent representations.

## 6. Run your implementation

Complete all three functions and select **Run tests**, or run:

```bash
paperleet test representation/vae --impl problems/representation/vae/solution.py
```

Tests include independent scalar likelihood/KL calculations, closed-form derivatives, extreme logits, reduction identities, pure sampling, shape errors and actual encoder/decoder training.

## 7. Further exploration

Derive why negative ELBO bounds negative log likelihood. Compare posterior sampling to using its mean, then explain which estimator you are optimizing. The Bernoulli observation model suits this binary toy; continuous images require a separately specified likelihood.
