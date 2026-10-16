# Policy distillation and context alignment preparation

The new four-module path connects SFT loss masks, student-prefix OPD, OPSD
dual-context alignment and OPSD vocabulary-wise clipping. Three original
bilingual guides add nine explain/implement/verify prompts. The existing 54
guides and 11 paths retain their previous content and order, bringing the
handbook to 57 guides / 171 prompts and 12 paths. These are teaching self-checks,
not collected interview reports or company-frequency evidence.

## Sources and bounded scope

The local Chinese/English READMEs, problem metadata, reference functions and
maintained tests define each exercise's exact interface and reduction. The
guides distinguish these contracts instead of treating every distillation loss
as interchangeable.

- [GKD v3](https://arxiv.org/html/2306.13649v3), Section 3 equations (2–4) and
  Algorithm 1: student-prefix distillation, lambda=1 and stopped discrete samples.
- [OPSD v3](https://arxiv.org/html/2601.18734v3), Algorithm 1 and Section 3.2:
  privileged teacher context on the same student rollout, stopped teacher targets
  and vocabulary-wise upper clipping.
- [Pinned author trainer](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/opsd_trainer.py),
  `generalized_jsd_loss` and the causal student/teacher extraction in
  `compute_loss`: independent context offsets, `no_grad` teacher and pointwise
  clipping before reduction. Its current masked `batchmean` uses valid-token
  normalization. The exercise's equal-response average follows its declared
  contract and paper formulation; it is not claimed identical to that trainer.

The OPD toy samples a fixed-length categorical Markov model with no EOS and
compares full vocabulary distributions. It is not a sampled-token policy-gradient
estimator. Its loss averages all valid tokens. The OPSD alignment exercise uses
each branch's `prompt_length-1+j`, safe masked gathers and teacher detachment.
The clipping exercise averages within each response and then over all responses,
including empty ones as zero; capped signed terms can yield a negative result.
The content makes no large-model benchmark, EMA or teacher-adapter claim.

## Validation

The new curriculum test first fails on the preceding source because the path is
absent. Maintained mounted tests then exercise its ordered four-module overview,
progress and recent-draft continuation, three-plus-one disclosure, all next-step
transitions, completion and missing-module behavior, deferred guide loading,
every bilingual hint and exact company/role/stage/saved source context through
practice and sign-in. Existing handbook scope and editor-preservation coverage
remains in place.

Independent native examples check a 1:3 BOS distribution with absorbing token
states, graph-connected zero for masked nonfinite OPD logits, causal positions
[1,2,3] versus [4,5,6], the approximately -0.147404 signed clipping example,
teacher gradient isolation and unequal-length/empty response weighting.
Run the four current modules through the original three-repeat reference/mutant
judge, catalog checks, full frontend suite, both build profiles, private core
checks and a genuine account/session HTTP/temporary SQLite preparation story
before publication. No exercise, runtime, API, storage schema or company evidence
is modified. Cloud and exact live delivery are separate release gates; no browser
visual QA or operating-system key automation is claimed.
