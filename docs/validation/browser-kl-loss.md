# KL loss compatibility in browser judging

The browser adapter adds the functional
[`kl_div`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.kl_div.html)
and [`KLDivLoss`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html)
APIs. An equivalent DistiLLM-2 learner using both APIs passes all nine native
checks; the prior committed browser runtime raises `BrowserUnsupported` in
eight checks. Neither the lesson nor its reference, tests or mutants changes.

## Scope

Matching float32/float64 input and target tensors support arbitrary broadcast
shapes, scalar/strided/empty arrays, probability or log-probability targets,
and all four reductions. Mean divides by the broadcast output element count;
batchmean divides by the input's leading dimension, with native scalar and
zero-batch behavior. Raw NumPy values and analytical VJPs independently preserve
both input and target graphs, including target contributions through a mixture.
Saved storage is checked before backward, including frozen operands.

Probability targets of zero have a zero xlogy term but retain native undefined
target gradients. Negative, nonfinite and extreme targets preserve native CPU
values and gradients, including NaNs; no epsilon is added. Finite JVP supports
positive probability targets or finite log targets with representable
exponentials. Reverse differentiation through both primal and tangent channels
retains real graphs. Mixed dtypes, legacy reduction arguments, generic
higher-order backward and unsupported JVP boundaries use the existing native
server path.

## Verification

- 53 focused checks pass in 2.80 seconds against native PyTorch 2.14.1, plus an
  independent scalar likelihood and central differences for both arguments.
  Checks include all reductions, broadcast accumulation, empty/scalar JVP,
  finite JVP/reverse-over-forward, storage mutation, no-grad and isolated public
  function/module dispatch. The existing TorchScript deprecation warning remains.
- The alternate DistiLLM-2 learner passes the complete unchanged nine-test suite
  under native CPU and the exported NumPy adapter. Swapping the KL arguments is
  rejected by assertions rather than treated as unsupported compatibility.
- Exported runtime/manifests/details update the 66 autodiff-dependent modules;
  Schedules is unchanged. All 1,916 existing immutable manifest/detail assets
  remain available. All other problem index fields, papers, categories and
  lesson sources are unchanged.

## Full regression and actual CPU/WASM

- Full public-only regression: **1,213 passed, one optional Docker skip**, in
  132.52 seconds. This includes all unchanged maintained browser reference and
  mutation calibrations in isolated NumPy processes; it does not claim to run
  every module in a real browser.
- Complete frontend checks and the standalone production build pass in an
  isolated snapshot excluding concurrent UI work. The initial graph is
  **399,885 raw / 128,581 gzip bytes**, within unchanged 400 KiB / 130 KiB limits.
  The snapshot explicitly uses the public Python 3.12 environment for syntax
  preflight; the system Python 3.9 cannot run that existing null-byte fixture.
- The hosted production build also passes at **405,261 raw / 130,402 gzip bytes**,
  unchanged from its previous measured graph. The new Python operator stays in
  the on-demand shared runtime.
- Actual Pyodide CPU/WASM executes the alternate learner at **9/9**, including
  all four maintained fault calibrations and the original 60 gradient updates.
  Its own Lab curve decreases from 0.524419 to 0.00646713. Additional module-level
  assertions verify probability and log-target JVPs and all four reverse gradients
  against independent native float64 constants before the unchanged suite runs.
- Swapping the KL direction returns **6/9**; restoring the correct direction
  returns **9/9**. A valid native-only legacy argument shows the unsupported-API
  notice, preserves the exact source and marks the prior successful report as
  older code. Restoring the source returns **9/9** again.
- Initial localhost phases: preparation **1.80 s**, compatibility calibration
  **0.50 s**, complete tests **0.12 s**. These observations are not internet cold
  starts or a comparison with Oracle hardware. Browser warning/error logs are
  empty. The static QA server sees zero API requests and zero POSTs; no live
  authenticated native fallback or new Docker image is exercised.

No public push, production app/image deployment or account credit is claimed.
Private pinned-core/packaged-worker validation is recorded in the private repo.
