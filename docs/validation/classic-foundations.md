# ResNet, BatchNorm and VAE

Validated on 2026-10-17. Three previously absent classic papers add three CPU/browser exercises, 30 learner tests and 19 localized mutants. After integrating concurrent public main, the catalog contains 63 papers plus one introductory course, 85 core modules, 679 learner tests and 378 mutants.

Paper IDs 64–66 and module IDs 97–99 follow the maximum published IDs in public main `c79995fe`. The three unpublished exercises were renumbered when concurrent sampling/continual-learning work occupied their original branch IDs. All 82 upstream module records and 61 upstream paper/course records remain identical after export; published IDs and existing learner drafts are unchanged.

## Sources and selected scope

| Paper | Source | Exercise |
| --- | --- | --- |
| ResNet, CVPR 2016 | [Section 3.2, Equations (1)–(2), Figure 2](https://arxiv.org/html/1512.03385v1#S3.SS2) | Two matrix layers, identity/projection shortcut and post-addition ReLU; no convolutional network or BN implementation. |
| Batch Normalization, ICML 2015 | [Algorithms 1–2 and Section 3.2](https://arxiv.org/html/1502.03167v3#S3), [conference proceedings](https://proceedings.mlr.press/v37/ioffe15.html) | Differentiable batch moments, unbiased running-variance estimates and detached frozen inference, using an explicit pure-state interface. |
| Auto-Encoding Variational Bayes, ICLR 2014 | [Section 3, Equations (9)–(10), Appendices B/C.1](https://arxiv.org/html/1312.6114v11#S3) | Pathwise Gaussian sampling, analytic nonnegative KL and a stable Bernoulli negative ELBO, with original KL coefficient one. |

Each package has bilingual seven-section statements and notes, exact starter contracts, a reference, T0–T3 tests, mutants, and a visualization generator. Independent scalar values and analytic/finite-difference gradients supplement reference agreement. Browser configuration discovers every test automatically. The three-guide, nine-prompt bilingual vision/variational path is present in shared interview metadata and lightweight route validation.

## Numerical and browser checks

- New-module three-repeat native gate: 660 test executions; 6/6 residual, 6/6 BatchNorm and 7/7 VAE mutants caught by genuine assertions.
- Final integrated native gate: `paperleet ci --budget 600`, 85 modules, 11,349 test executions, three repeats, passed in 301.93 seconds with no issues. No per-test budget, assertion, tolerance or training loop was weakened.
- Public infrastructure and StrideTorch regressions before concurrent curriculum integration: 41,599 passed, one skipped. After integration, 300 affected catalog/interview/browser registration and contract tests passed. Public and private applications use separate virtual environments so the open-edition boundary check remains effective.
- Before concurrent four-module integration, full actual WASM catalog: Pyodide 0.29.4 / NumPy 2.2.5, 81/81 modules passed reference, all-mutant calibration and learner suites. The three new modules took approximately 9.30/8.01/5.91 seconds respectively on this run, with no browser-budget overruns. The existing Flow Matching module exceeded a browser phase budget on this concurrently loaded host; the verifier retained the full computation and recorded that warning. Full-catalog correctness does not establish device latency or browser-deadline compliance.
- After concurrent curriculum integration and renumbering, the seven affected modules passed complete actual WASM reference/mutant/learner suites at their final IDs, without phase-budget overruns. Together with the unchanged module records verified against upstream, these runs cover every current module; the earlier full-catalog run remains a separate observation.
- Actual Chrome production-build workspace: ResNet starter 0/9, completed implementation 9/9; BatchNorm implementation 11/11; VAE implementation 10/10. The Lab displays learner-executed curves separately from reference curves, with measured artifacts and correct source identity. Formula rendering and practice/source navigation were inspected.
- Matching production worker dependencies: Python 3.12, PyTorch 2.6.0, Linux ARM64, exact `Dockerfile.judge` allowlisted source. Non-root containers ran without network, with read-only root, dropped capabilities, two CPUs and a 1 GiB memory limit. All three references passed and all 19 mutants produced genuine failing assertions. Account code and credentials were absent from the worker. The final 85-module COPY image was rebuilt, and all seven affected references passed its actual isolated worker protocol; Python implementation/test sources are identical to the native/WASM-validated sources.

Timings are local observations, not CPU/WASM/server speed comparisons. Docker image and detailed raw reports remain local development artifacts.

Complete frontend checks at the integrated 85-module catalog passed. The branch also integrates public main `4db2b71b`, retaining its profile-policy/avatar dependency separation, production Pages build checks and moderation work. The repeated field validation in lightweight routing is consolidated without importing deferred curriculum metadata; existing unknown/duplicate-filter checks remain effective. Path/handbook count assertions derive their expected sizes from the maintained catalogs. Hosted (including the release workflow's split API origin), connected-public, standalone-public and local production builds all pass after integration (hosted initial JavaScript: 406,690 bytes raw / 132,089 bytes gzip). The fixed 400 KiB raw / 130 KiB gzip initial-JavaScript limits are unchanged.

## Experiments

| Module | Actual experiment | Native reference observation | Unchanged acceptance |
| --- | --- | --- | --- |
| ResNet | Four width-eight blocks, 32 positive three-dimensional samples, 80 Adam updates, seed 73 | MSE 0.06155 → 0.0009984 | Final MSE < 0.003, <5% of initial, learned residual nonzero |
| BatchNorm | 80 sampled `[32,2]` minibatches, mean 3/variance 4, affine Adam fit, seed 91 | MSE 3.380 → 0.0007124; inference companion change 0 | Final MSE < 0.003; affine errors < 0.04; frozen inference invariant |
| VAE | Two complementary six-bit patterns, two latent dimensions, 120 encoder/decoder Adam updates, seed 107 | Initial negative ELBO 4.1671; final-20 mean 1.6942 | Final-20 mean <80% of initial; NLL + KL decomposition preserved |

All experiments use learner functions. They are labeled synthetic teaching adaptations: no ImageNet/CIFAR/MNIST reproduction, model-accuracy ranking, exact marginal-likelihood claim or universal optimization claim.

## Independent blind implementation

An independent agent received only the three starters and Chinese/English statements, without references, tests, mutants, notes or website access. All three independently written implementations passed on their first attempts (9/9, 11/11, 10/10). Agent wall-clock observations were 11, 18 and 19 seconds, not estimates of human difficulty. Review feedback clarified that the BatchNorm toy's second normal-distribution parameter is variance and that KL is nonnegative, with a positive loss sign.

## Release coordination

The paired private change adds the three modules to the judge allowlist and image COPY rules, with native job/report and exact copied-worker tests. The private `core` pin must resolve to the reviewed, published public commit when released; replace a feature-branch pin if a squash/rebase changes the published SHA. Follow both repositories' explicit release guides. Ordinary PRs/pushes do not publish. No production data, permissions, workflow enablement, version tag or deployment is changed by this contribution.
