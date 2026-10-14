# CPU/WASM distillation modules — 2026-10-14

Added full browser support for four existing lessons. Problem statements, required/provided functions, reference solutions, tests, mutants, model sizes, experiment loops and thresholds are unchanged.

| Module | Complete browser suite | Calibration | Original behavioral experiment |
| --- | --- | --- | --- |
| GLM-5 off-policy correction | 7/7 | All 4 mutants rejected | 100 stale-rollout bandit updates; strict ratio endpoints and frozen importance weights |
| OPD | 6/6 | All 4 mutants rejected | 120 Markov student-prefix training updates, with fresh student sampling and frozen teacher |
| Consistency Models | 7/7 | All 4 mutants rejected | 350 updates of the original 48-wide SiLU network and independently copied EMA target |
| Progressive Distillation | 6/6 | All 4 mutants rejected | 180 SGD updates; a one-step student matches the independently evaluated two-step teacher |

## Implementation and independent checks

- Fixed scalar comparisons that previously rounded a Python float to float32 before comparing against float64 tensors. Adjacent representable boundaries and scalar-tensor promotion now match native PyTorch.
- Added bounded Linear, positional Sequential, SiLU and SGD. Identical-array forward results and input/weight/bias derivatives are checked against native PyTorch. SGD parity includes multiple updates, missing gradients, zeroing, momentum, dampening, decay, Nesterov and maximize.
- Leaf parameter deep copies register their own storage-version counters, retain shared-module identity, and support target freezing. A detached alias mutation invalidates a saved backward graph; EMA never silently reuses modified tensors.
- Added named no-grad updates, unweighted bincount, and single-draw categorical sampling by independent exponential races. Both replacement flags have the correct single-draw distribution. Reproducibility, zero-probability categories and empirical frequencies are tested, independently of native sampling. Single-draw and normal RNG streams need not match native draws exactly; the maintained learning criteria are unchanged.
- The testing adapter keeps native float32/float64 tolerances, shape/dtype checks, NaN policy and exact integer comparisons. Tests import their original `torch.testing.assert_close`; assertions are not rewritten. Unimplemented testing/optimizer execution options continue to fall back.
- Exported problem bundles include each original `toy.py` as well as `experiment.py`; content checks prevent substituting these helpers. Previous immutable assets are retained for cached clients.

## Evidence

- `.venv/bin/python -m pytest tests/test_browser*.py -q`: **486 passed**. Includes all 41 reference suites and all 166 maintained mutants, unchanged-source export checks, failure/unsupported semantics and independent native parity.
- `.venv/bin/paperleet ci --problem rl/glm5_offpolicy --problem rl/opd --problem diffusion/consistency --problem diffusion/progressive_distillation --repeats 2 --budget 90`: **PASS**, 4 problems, 260 test cases, 2 runs, all 16 native mutants caught.
- `cd web && npm run check`: passed, including filtering and dispatch for the four new routes, complete bundle hydration, private workspaces and mixed-runtime badges.
- `cd web && npm run build`: passed TypeScript and initial graph budget; 366,345 bytes raw / 117,263 bytes gzip. No new optional feature became an initial import.
- Real in-app browser, disposable production-build server on loopback: all four reference suites passed in **Browser only**, each with 4 calibrated mutants. Server logs confirmed zero judge submissions from these browser runs.
- Submitted OPD's unfrozen-teacher mutant in the actual browser: **5/6**, with the expected teacher-gradient failure. This was learner failure feedback and did not create a server job.
- Consistency's actual browser report opened in **My run results**, displaying submitted/target point clouds, its loss curve and endpoint MSE **0.001276**, below the unchanged 0.04 limit.
- Progressive Distillation’s actual browser lab displayed its original loss curve and endpoint MSE **2.2049e-14**, below the unchanged 1e-5 limit, with two teacher calls and one student call per sample.
- A trusted Progressive Distillation fixture using `SGD(foreach=True)` was rejected as unsupported in Browser only with zero server submissions. Auto then created exactly one queued job, ran the identical source with native CPU PyTorch in a disposable directory, and returned **Accepted 6/6**. This exercised real queue/runner/report flow; it did not exercise the production Docker service.

The supported catalog is now **41/60 modules, 308 tests, 166 mutants**. Active exported assets total 771,767 bytes versus 4,716,257 bytes for equivalent self-contained bundles (83.6% less), excluding the Python runtime and retained historical hashes. Remaining modules use the server.
