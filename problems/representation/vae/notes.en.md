# Signs, scales and pathwise derivatives

The paper maximizes reconstruction log likelihood minus positive KL. An optimizer minimizing a loss needs reconstruction **negative** log likelihood plus positive KL. The analytic KL is a sum over latent coordinates, while the outer objective averages observations.

For fixed noise, `dz/dmu=1` and `dz/dlogvar=0.5*exp(logvar/2)*noise`. For the positive KL, derivatives are `mu` and `0.5*(exp(logvar)-1)`. The loss's Bernoulli-logit derivative is `(sigmoid(logits)-target)/B`. These independent identities catch forward-correct gradient detachment and incorrect reductions.

Seven mutants use variance as standard deviation, detach the mean, average latent coordinates, reverse KL's sign, average observation coordinates, omit KL or detach its contribution. Changing a reduction can define another objective, but not the one declared here. `log(sigmoid(logits))` also loses numerical stability for extreme logits; work directly in logit space.

The training loss is a one-sample estimate of the negative variational bound. A decreasing toy loss is not an exact marginal-likelihood estimate, a guarantee of noncollapsed latent variables or evidence of image-generation quality. The fixed coefficient 1 is deliberate; beta weighting and annealing require a separate exercise contract.
