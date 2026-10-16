# Representation alignment and distribution preparation

A new four-module curriculum connects VICReg, SIGReg, VISReg and REPA. Three
original bilingual guides add nine explain/implement/verify prompts. The previous
51 guides and all ten paths are retained, bringing the handbook to 54 guides,
162 prompts and eleven preparation paths. This is an editorial curriculum for ML
and research preparation, not a claim about company interview frequency.

The prompts follow the owned exercises' bilingual READMEs, interfaces, reference
implementations and independent scalar/gradient tests:

- SIGReg retains the sine component, Gaussian window and N scaling, integrates
  supplied positive grids with the symmetry factor, and averages only after each
  slice statistic is formed. Supplied directions and nodes are fixed; features
  remain differentiable. Exact zero collapse has positive loss and zero gradient.
  This exercise supports nonuniform supplied grids and finite fixed slices.
- VISReg separates population scale, sorted projected shape and centering. Only
  the shape denominator's standard deviation is detached. Global-view means
  remain differentiable and all views enter invariance. Tied projections and
  zero variance require finite forward values without a unique-gradient claim.
- REPA uses negative patch cosine, separate norm clamps, frozen teacher/velocity
  targets and mean velocity MSE plus weighted alignment. Perfect alignment may
  give a negative total. The exercise returns one weighted scalar; the author
  implementation returns separate components to its training caller.

Primary context was checked in [LeJEPA §§4.2.3 and 5.1](https://arxiv.org/html/2511.08544v1),
[VISReg §3.1 and Algorithm 1](https://arxiv.org/html/2606.02572v1), and
[REPA](https://arxiv.org/html/2410.06940v4). The pinned author implementations for
[SIGReg](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/univariate/epps_pulley.py)
and [REPA](https://github.com/sihyun-yu/REPA/blob/9d60489e823bd9b356319b6d30e4cf7f333119a9/loss.py)
were inspected. The older VISReg repository README only redirects; it is not
algorithm-code evidence. No image-pretraining benchmark or universal finite-slice
identification result is claimed by these exercises.

## Verification

The mounted curriculum check first fails on the preceding source because this
path does not exist. It then checks the actual selected three-plus-one module
preview, latest draft/progress, every workspace next transition, exact
company/role/stage/saved return, completed return and missing-module handling.
The handbook check opens all four actual guides and every native hint in both
languages, retaining source/login context and deferred imports. The public hub
also exposes the eleventh path without changing contributor evidence.

The three unchanged numerical exercises pass 375 executions over three repeats,
with all twelve maintained mutants detected. Full frontend, local/hosted builds,
private API checks and a real account/session HTTP/temporary SQLite story should
pass before publication. Larger guide text remains deferred. Production
verification is read-only; no browser visual QA is claimed.
