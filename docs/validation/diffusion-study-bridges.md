# Diffusion study bridges

The existing Training & experimentation path jumped from forward noising to
reverse sampling without a guided prediction-target exercise, then skipped
implicit sampling on the way to Flow Matching. It now includes the maintained
v-parameterization and DDIM modules in that sequence. The path keeps its existing
`training` identity and its optimizer and policy-objective modules, for nine
modules in total. Existing drafts, accepted results and source-return filters
continue to use the same workspace and routes.

Two bilingual guides add explain, implement and verify prompts. The handbook now
contains 39 guided concepts and 117 prompts, loaded when the reader opens them.
The source contracts remain the maintained exercise statements:

- `diffusion/param_conversion`: implement only `make_v`, `v_to_x0` and `v_to_eps`.
  Distinguish square-root cumulative signal coefficients, angular velocity,
  division-free inverse rotation and information lost at division endpoints.
- `diffusion/ddim`: implement only `ddim_step`. Distinguish current and target
  times, the direction/random variance budget, deterministic noise independence,
  the final clean boundary and approximate finite-step inversion. Timestep
  selection, sampling and inversion helpers remain supplied.

No algorithm, reference, mutant, training step, quality threshold or browser
deadline changes. The README coverage correction follows the current exported
catalog and module YAML: 78 modules, 619 learner tests and 342 mutants.

## Verification

The shared-catalog test and complete frontend checks pass. The existing study-flow
checks now exercise recent DDIM-draft resume, the four diffusion transitions,
nine-module picker, Unicode source filters and completion back to the selected
path. Handbook checks cover both new guides, closed hints, locale changes and
practice/sign-in return links. Hosted, connected-public and standalone-public
production builds pass their existing asset budgets.

Actual standalone production UI on runtime 0.20 passes parameter conversion 7/7
and DDIM 9/9. The latter calibrates nine reference tests and catches five mutants.
Download-before/reload/download-after comparisons preserve the exact UTF-8 drafts
(1,778 and 2,716 bytes). Saved acceptance survives reload; full notes stay locked
before acceptance. The current-code DDIM experiment shows all eight modes and
MMD² approximately 0.00052444 under the unchanged 0.00078666 threshold. This is
the existing synthetic oracle experiment, not pretrained-model reproduction.

English/Chinese prompts render in the real workspace. At 390px, document width
and scroll width are both 390px; the temporary viewport override is reset.
No account, community or formal submission records are created. Full cloud gates
and production rollout are tracked by the PR checks separately from this local
UI evidence.
