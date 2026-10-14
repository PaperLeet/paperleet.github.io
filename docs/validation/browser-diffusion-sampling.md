# Full CPU/WASM diffusion sampling — 2026-10-14

Extended the existing browser judge to DDPM reverse sampling, DDIM and conditional Flow Matching. The maintained problem files, YAML calibration values, implementations, experiments, tests and mutants are unchanged. No dataset, network, sample count, training loop, precision tolerance or acceptance threshold was reduced.

| Module | Browser suite | Calibrated mutants | Retained workload |
| --- | --- | --- | --- |
| DDPM reverse sampling | 10/10 | All 5 caught | Both variance modes; 1,000 reverse steps and 2,000 samples per experiment |
| DDIM | 9/9 | All 5 caught | Eta limits, trailing time order, inverse/step convergence and 2,000-sample distribution checks |
| Flow Matching | 8/8 | All 4 caught | Three 128-wide hidden layers, 2,000 updates, batch 256, 50-step Euler sampling and 2,000 generated/target examples |

## Fidelity and compatibility

- Bundles carry the original Gaussian-mixture distributions, analytic score/velocity oracles, MMD/coverage metrics, TimeMLP helper and `problem.yaml`. Export tests compare every dependency byte with its maintained source. PyYAML is pinned by the Pyodide lockfile, self-hosted and loaded only for bundles using that configuration resource. Public resource validation still rejects traversal and arbitrary YAML files.
- DDIM initially failed its unchanged MMD threshold with NumPy's different normal RNG stream. Added the CPU Box–Muller filling rules, cached scalar pairs and overlapping 16-value tail refill. Independent float32/float64 checks cover scalar/vector boundaries, 4,000-value sampling arrays, mixed draws and exact following uniform streams. Float32 SIMD rounding remains architecture-dependent; bitwise equality across all CPU libraries is not claimed.
- The harness now resets both NumPy and the installed torch adapter's generator before each test, matching the native deterministic fixture. RNG forks restore pending normal samples as well as MT state after nested contexts and exceptions.
- Added direct Euclidean `cdist` with analytical gradients, broadcast reduction, zero-distance handling and an allocation bound that also covers zero-dimensional feature vectors. Other norms and oversized allocations fall back. Logit/clamp/NaN behavior is checked against native PyTorch. Registered buffers remain outside the parameter iterator.
- Fused Linear and SiLU analytical VJPs remove unnecessary Python graph nodes. Identical-array values and input/weight/bias gradients retain independent native parity.
- Real Flow Matching calibration exceeded the old 90-second limit. It now has a 240-second calibration phase, 60-second learner phase and five-minute absolute cap. The UI explains the complete training cost and offers Stop and server mode. Detailed progress cannot extend the caps; other lessons retain their original limits. Training and mutant checks are complete, rather than shortened to fit a timeout.
- Reports and labs now show parameterized names such as `test_gmm_sampling[posterior]` and `[beta]`, including restored reports. Raw test identity and scoring are unchanged.

## Verification

- `.venv/bin/python -m pytest tests/test_browser*.py -q`: 533 passed, including all 44 reference suites, all 180 maintained mutants, RNG parity, source fidelity, allocation bounds and unsupported/failure semantics.
- `.venv/bin/paperleet ci --problem diffusion/ddpm_sampling --problem diffusion/ddim --problem diffusion/flow_matching --repeats 2 --budget 120`: PASS, 3 problems, 308 test cases, 2 runs; all 14 native mutants caught.
- `cd web && npm run check`: passed, including immutable bundle hydration, YAML/path validation, browser dispatch, UI state and non-renewable normal/heavy execution caps.
- `cd web && npm run build`: TypeScript and initial graph checks passed, 367,243 bytes raw / 117,642 bytes gzip; optional editor, community, accounts and labs remain deferred.
- Real in-app browser against a disposable production-build server: all three references passed in **Browser only**, with all 14 calibrated mutants. No server judge submission was made by these complete browser runs.
- Actual Flow Matching artifacts displayed eight recovered modes, MMD² **0.012716** against the unchanged **0.019074374055781274** threshold, its original loss curve and **32.326 seconds** for the learner's T3 experiment. Complete per-run reference/mutant calibration adds several minutes on the tested browser; these are observed values, not a cross-device benchmark or an ETA.
- DDPM's actual two variance experiments displayed MMD² approximately **0.00181** and **0.00179**, both passing. After refresh, the report and lab retained the distinct posterior/beta labels.
- A trusted DDIM fixture using unsupported `cdist(p=1)` produced a compatibility message and zero server submissions in Browser only. Auto created exactly one queued job for the identical source and returned native CPU **Accepted 9/9** in a disposable directory. This did not exercise the production Docker service.

The catalog now supports **44/60 modules, 335 maintained tests and 180 mutants** in CPU/WASM. Active shared exports total **867,766 bytes**, versus **5,388,655 bytes** for equivalent self-contained bundles (83.9% less), excluding runtime wheels and retained historical hashes. Slow or incompatible environments keep the server path. Production Docker deployment and other browser/device coverage are outside this local verification.
