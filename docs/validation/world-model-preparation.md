# Latent world models and planning preparation

A four-module curriculum connects I-JEPA, Causal-JEPA object trajectory masking, LeWM joint prediction and LeWM CEM planning. It appears for ML/research roles, with a three-module preview and the existing native disclosure for its fourth module. The selected path retains full progress, resumes a current draft, and scopes the workspace picker and next-module actions in curriculum order. Company, role, stage, period, saved-list and source returns are preserved. Missing modules cannot count as completion.

Three new, original bilingual guides add nine explain/implement/verify prompts. All previous 48 guides and nine paths remain unchanged; the total is 51 guides, 153 prompts and ten paths. This is a curriculum addition, separate from contributor counts or company frequency. No community reports are fabricated.

## Content review

Each prompt is grounded in its owned exercise's bilingual README, interfaces, reference implementation and all eight tests, including mutation cases. Primary references were checked on 2026-10-16:

- [Causal-JEPA §4.2](https://arxiv.org/html/2602.11389v2#S4.SS2), the pinned [input constructor](https://github.com/galilai-group/cjepa/blob/412337d0210bf98cee2ca90c3586ab2ea7ca519e/src/cjepa_predictor.py) and [CLEVRER training loss](https://github.com/galilai-group/cjepa/blob/412337d0210bf98cee2ca90c3586ab2ea7ca519e/src/train/train_causalwm_from_clevrer_slot.py). The exercise keeps first-frame identity visible, excludes anchor supervision, separates history/future means and freezes targets. Its per-example mask is a teaching extension; the released loop also supervises selected anchors. This distinction is stated explicitly.
- [LeWorldModel §§3.1–3.2 and Appendix B](https://arxiv.org/html/2603.19312v2), pinned [training objective](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py) and [SIGReg implementation](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py). Prediction keeps both encoding paths differentiable, regularization receives all frames in time-major order, and the exercise rollout isolates one-state context. Planning uses terminal cost, stable lowest-cost elites and population variance. The supplied sampling loop owns action bounds and replanning is outside the exercise.

The guides explain the existing exercise contracts; they do not change numerical interfaces, implementations, tests, browser bundles, pretrained assets or execution limits. Benchmark reproduction and causal identification are outside their scope.

## Verification

- The new curriculum assertion first fails on the previous source because no connected world-model path exists.
- `check-interview-study.mjs` covers curriculum order, selected three-plus-one disclosure, full progress, most recent draft, each picker/next transition, exact source context, completion and unavailable modules.
- `check-preparation-guides.mjs` opens all four guides through native controls, checks every new bilingual hint, language continuity, exact practice/login returns and deferred loading. Existing guides, scope switching, editor preservation, timeout/retry and stale imports remain covered.
- `check-interview-ui.mjs` verifies all ten named preparation paths. The public catalog contract and the three existing modules' full native judge run, with three repetitions and all twelve mutants, provide independent checks for the implementation hints.
- Before publication, run the complete frontend suite, public and hosted builds with unchanged budgets, private interview/deployment checks, and the genuine account/session HTTP and temporary SQLite flow. That flow must preserve contributor evidence and make no UI writes while reading the guides or returning from practice.
- Production checks compare exact released assets and runtime delivery. No browser visual QA is claimed.
