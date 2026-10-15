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

## Page-local calibration reuse — 2026-10-14

The immutable worker and full hydrated problem bytes now key a bounded in-memory compatibility cache. References and every mutant still run on first use; cached evidence expires after 15 minutes, never renews on a hit, and disappears on reload. Python/NumPy ABI checks precede reuse. Every learner submission starts a fresh opaque-origin worker and executes the unchanged complete suite. Worker/helper changes regenerate the worker hash, including during Vite development. No learner code, trained state or result is cached as calibration.

Actual in-app browser observations against an isolated localhost production build:

| Module / run | Preparation | Reference + mutant calibration | Complete learner tests | Result |
| --- | ---: | ---: | ---: | --- |
| LoRA first | 1.36 s | 0.42 s | 0.07 s | 6/6; all 4 mutants calibrated |
| LoRA same-page repeat | 1.36 s | displayed 0.00 s, reused | 0.17 s | 6/6 |
| Flow Matching first | 1.40 s | 162.39 s | 32.90 s | 8/8; all 4 mutants calibrated |
| Flow Matching same-page repeat | 1.45 s | displayed 0.00 s, reused | 32.35 s | 8/8 |

The Flow Matching total decreased from approximately **196.7 to 33.8 seconds** on this device. The 2,000-step model training, sample counts, tests, distribution checks and numerical thresholds were unchanged. These are paired single observations on localhost, not cross-device or Oracle benchmarks. Displayed zero calibration time is rounded, rather than a claim that validation/comparison costs nothing. Preparation includes bundle delivery, Python initialization, controller import, hashing and relay overhead.

Changing LoRA's scaling while reusing calibration returned **3/6**, and restoring the original source returned **6/6**; a cached compatibility record never supplied a learner pass. Reload retained the saved code/report but forced fresh calibration (0.44 seconds in the observed rerun). A separate 390px iframe completed Attention **7/7**, reference 7/7 and all 3 mutants; document and viewport widths were both 390px. Initial dynamic editor/rich-text requests failed transiently in one tab; the readable editor fallback retained the code and tests, and reload restored the normal editor.

With a temporary real account, an exact reviewed Attention fixture using unsupported `cdist(p=1)` was reported as incompatible in Browser only with **zero judge POSTs**. Auto then captured that same source, created **one job / one native execution** and passed **7/7**. It did not import a guest pass into the account. The QA adapter accepts only fixed reviewed fixtures; this is not production Docker execution.

The full frontend checks cover exact calibration counts/names, full bundle and worker byte identities, runtime fingerprint mismatch, non-renewing expiry, sleep/clock anomalies, LRU bounds, immutable public-only records, late-message rejection, stop/error cleanup, fresh source, and hashing-unavailable full-calibration fallback. A real filesystem-watcher test verifies that editing a helper regenerates the worker asset and reloads development clients. Original numerical source/bundle exports are unchanged. Same-origin hosted, browser-only Pages and split-origin hosted builds remain under the unchanged initial-JavaScript limits.
