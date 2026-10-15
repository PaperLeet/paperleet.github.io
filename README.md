# PaperLeet

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="web/public/brand/paperleet-logo-dark.svg">
  <img src="web/public/brand/paperleet-logo-light.svg" alt="PaperLeet" width="320">
</picture>

**English** | [简体中文](README.zh-CN.md)

**Website source:** [PaperLeet/paperleet.github.io](https://github.com/PaperLeet/paperleet.github.io) contains this website, its practice catalog, and the browser/local judging tools. The site's GitHub icon opens this repository.

**Standalone offline edition:** [Hongyang-Du/PaperLeet](https://github.com/Hongyang-Du/PaperLeet) is the main open-source project for local CPU practice, with the full learning guide, saved local solutions, and an MIT license. Star that repository to support the project.

Read papers, implement the methods, and test your understanding through mathematical properties and actual behavior.

The catalog includes **56 papers, one introductory course, and 77 focused core modules**. Start with a paper, read its contribution and selected scope, then implement its core modules in a LeetCode-style split workspace. Supporting code is supplied so each exercise focuses on the paper’s key mechanism. Completing a paper means passing its selected core modules. Paper organization lives in `papers.yaml`; statements, implementations, tests, and visualization data live in `problems/`.

## Run locally

The product is named **PaperLeet**. Use the `paperleet` CLI; the earlier `leetpaper` alias remains supported.

Requires Python 3.10+ and Node.js 20.19+. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -e ./packages/paperleet-torch
paperleet export
npm --prefix web ci
make dev
```

Open the URL shown in the terminal (default: **http://127.0.0.1:5173**). On subsequent runs, use **`make dev`** from the repository root to start both the website and the judge service. Press Ctrl-C to stop both processes.

## Repository scope

This public repository contains the frontend, paper catalog, browser runtime, native local judge, and offline maintainer tools. `make dev` works without an account, email provider, or private repository. Signed-out browser practice automatically saves code and test results in the current browser, so they can be restored after refresh or reopening the site.

The NumPy-based browser tensor/autodiff implementation is also available as the independently buildable [paperleet-torch library](packages/paperleet-torch/README.md). Its standalone package needs NumPy only; existing PaperLeet installations include the same library source and retain their current entry points. Keep ordinary `import torch`, `from torch import nn` and `import torch.nn.functional as F` in your code: run standalone scripts with `paperleet-torch train.py`. PaperLeet's browser Worker activates those aliases automatically.

The account/community screens and API contracts are public frontend code. Their server implementations, online storage, hosted judge orchestration, and production deployment tools live in the separate private `PaperLeet/backend` repository. That repository pins this public core as a Git submodule; no private service is installed or fetched by the open edition.

Git history is intentionally retained. Earlier commits contain the former combined backend; deleting it from the current tree does not make that earlier code confidential. New online backend changes belong only in the private repository.

The online practice site is https://paperleet.com/, with the public frontend published through GitHub Pages. The local bridge binds only `127.0.0.1`; hosted services use a separate entrypoint.

## Solve your first problem

For an introduction, start with [LLM Training & Generation](https://paperleet.com/#/papers/llm-training-generation): six standalone exercises on shifted losses, SFT masks, sampling, gradient accumulation, and clipping, all runnable on browser CPU/WASM.

1. Choose a paper, such as **6 · DDPM**, then open its **Forward Noising** module.
2. Read the goal and complete only the required functions in the Python editor. Schedules and other supporting functions are already implemented.
3. Click **Run tests** to check your code in the browser and wait for the results.
4. Use each test layer's failure explanations, expected values, and actual values to revise your code, then run it again.
5. On the online application, sign in and click **Submit** to have the server judge your solution and save the submitted code and result to your account.

Passing the full browser test suite updates browser learning progress and unlocks notes on common pitfalls. Account completion requires a successful server submission. Running only a subset of tests does not qualify as passing the full suite. Use the Chinese / English toggle in the upper-right corner to switch the interface, problem statements, and explanations. Paper and module numbers, numeric aliases, and internal directory names stay stable for existing drafts and CLI commands. A later failed submission does not erase a previously accepted result.

Signed-out code and browser test results are saved locally in the browser, separately from signed-in accounts. These drafts stay on the same device, browser, and site origin; clearing site data or ending a private-browsing session can remove them. Signing in does not automatically import guest drafts. After sign-in, account drafts synchronize automatically to the account. **Submit** saves a separate record of the exact submitted source and its server result, including unsuccessful attempts. Open **My submissions** for the current problem to review and copy those solutions or load older attempts; later edits to the draft do not change submitted code. Submission history is private to its owner.

**Run tests** uses the browser runtime and does not automatically submit code to the server when a browser run fails or uses an unsupported API. Use **Submit** explicitly when you want server judging and a saved account submission. Signed-in browser test results and learning progress can remain locally in that browser, scoped to the account; they do not become formal account submissions.

Restoring the starter asks for confirmation before replacing the current draft. It does not automatically download a backup; copy or download your current code first if you want to keep it. Saved submissions and earlier accepted results remain available in the account, while old test results are marked as belonging to previous code.

If the code editor cannot load, plain-text mode still supports editing and browser tests, with account draft synchronization when signed in. **Reload editor** refreshes the page after account draft and local learning saves finish; it stays disabled during tests, failed saves or draft conflicts. Reloading restores successfully saved guest drafts and browser test results, synchronized account drafts, and locally saved signed-in browser feedback. Local save failures are shown in the editor; retry saving or download your code before leaving.

Failed shared numerical assertions show the zero-based mismatch coordinate, expected/actual values, absolute error and test tolerances, or both tensor shapes when they differ. Full error details remain expandable. These optional diagnostics work with browser and server reports and do not affect grading; server submission reports are retained with the account. NaN/Infinity experiment data remain readable failure evidence rather than breaking browser result transport; older reports retain their original error text.

**Go to code line** moves the cursor to a current syntax error or an explicit learner-code traceback frame. It opens the code panel from the lab and works in plain-text mode too. Modified or unverifiable source has no navigation action; test/reference-only failures keep their diagnostics without inventing a code location.

Use **Light / Dark** in the top bar to choose an appearance. The first visit follows your system preference; an explicit choice is remembered in this browser. The paper library puts categories, search, filters, and module progress together; paper details contain the technical summary and expandable practice scope.

Drag the workspace dividers to resize the description/code columns and the code/results rows. Sizes are remembered in this browser. Double-click a divider or press Enter while it is focused to restore its default; arrow keys fine-tune the split. Run controls stay visible, and narrow screens stack the description above the editor.

The **Featured collections** carousel groups papers into LLM Basics, RL, Diffusion & Flow, Post-training, JEPA / World Models, and Visual Generation. Colorful cards show paper and problem counts; compact category chips below them filter the current collection. Collections can overlap and have shareable routes such as `#/collections/llm-basics` (existing `#/lists/...` links still work). Switching collections clears filters. Membership and bilingual card text live in `web/src/paperCollections.ts`.

Library search, category, topic, module difficulty, run location and paper status stay in the hash URL, for example `#/collections/diffusion?q=DDPM&difficulty=E&runtime=browser`. Refresh, browser Back/Forward, paper details, module navigation and sign-in keep the library return context. Returning also tries to restore the paper you opened, even when the all-paper order changes after refresh. Filters use the active workspace's current progress; a shared link carries filters rather than another person's code or grades. Progress-dependent results wait for workspace loading before declaring the list empty. See [navigation validation](docs/validation/library-navigation.md).

**Random problem** opens an unattempted module within the current collection and filters. It excludes any saved run, accepted result, or edited draft; an unchanged starter is still eligible. The button waits for practice progress to load and is disabled when no eligible problems remain.

The website displays browser test and server submission results automatically, with no report files to manage manually. For native local judging, the local service listens on `127.0.0.1:8765`, and Vite proxies `/api` requests to it. Native local runs save code to `workspace/` and execute it with Python/PyTorch on your machine. This local setup is intended for personal use; online services use the separate private application.

## Online services

Online accounts, synchronized drafts, profiles and community discussions require the separately operated application. The standalone local edition hides these routes and makes no account API requests. Browser and native local practice remain available without sign-in.

Each account can submit **1 paper proposal**, publish **5 discussion posts**, and publish **50 replies** per day. These are separate quotas that reset at **00:00 UTC**. Deleting published content does not restore the day's quota.

AI assistance can help draft and improve content. Review the text, verify its claims and sources, and take responsibility for the final publication. Automated bulk posting and replying are not allowed. Paper proposals enter the maintainer review process; submitting one does not automatically publish a new exercise.

Discussion pages offer read-only refresh/retry without discarding reply drafts. A reply acknowledged by the server appears immediately even if the follow-up read fails, with an action to focus that reply. Retrying the read does not submit the reply again; returning to the list keeps the module, search and filters.

With a compatible hosted backend, authors can edit a post's title, body and category while keeping its link, replies and reactions. Edit drafts survive closing the editor or reloading the same tab when storage is available. Concurrent changes require review of the latest content before saving; a response-lost save can be retried safely. [Validation and limits](docs/validation/community-post-editing.md).

Discussion and reply links can be copied without carrying feed searches or drafts. Reply links focus and highlight the selected reply, including after sign-in; unavailable replies leave the rest of the discussion readable. A manual-copy field is available when automatic copying fails. [Link validation](docs/validation/discussion-links.md).

Posts and replies retain a pending publication when a success response is lost. Retrying unchanged content reuses the same request identity; changed drafts offer an explicit new-content action or retry of the earlier publication while preserving new work. This requires the matching updated hosted backend. Pending recovery survives reload in the same tab when tab storage is available; closing the tab or clearing storage can lose it.

## Submit paper

In **local maintainer mode**, click **Submit paper / 提交论文** in the top navigation and provide a paper URL, an optional official code URL, and what you want to practice. The model reads these sources and generates Chinese and English problem statements, starter code, a reference implementation, layered tests, and buggy variants. Candidates proceed to human review after passing isolated validation. Only the current version approved by a human reviewer receives a numeric problem ID and enters the paper library as a module. Export groups modules that cite the same arXiv paper or source URL under one paper. The generator currently creates one candidate module per submission. It looks for author toys and paper figures, labels teaching adaptations explicitly, and generates T3 tests that record measured curves and metrics through a bounded `artifact` fixture. Human reviewers can inspect those results before approval.

Generation requires `OPENAI_API_KEY` and `PAPERLEET_MODEL` to be configured in the terminal used to start the service. Validation requires Docker to be running and the following image to be built from the repository root:

```bash
docker build -t paperleet-harness:local paperleet/submissions
```

You can still save submissions without a configured model or Docker; the website explains what is needed for generation or validation. Candidate materials and review records are stored in `.paperleet/submissions/`. The local workspace owner performs reviews. Hosted members do not receive generation, validation, or publication controls. See the [harness design (in Chinese)](docs/paper-submission-harness.md) for the full workflow, isolation boundaries, and human review criteria.

## For maintainers: CLI and quality checks

Use the CLI to maintain problems, debug the judge, or check reference solutions:

```bash
paperleet list
paperleet init diffusion/param_conversion
# Edit workspace/diffusion/param_conversion.py
paperleet test diffusion/param_conversion
paperleet test diffusion/param_conversion --layers T0,T2 --report reports/my-result.json
paperleet test rl/gae --impl problems/rl/gae/solution.py
paperleet mutate diffusion/param_conversion
paperleet ci
```

`init` does not overwrite existing code by default. Use `--force` when you intend to reset it.

`paperleet ci` checks that every reference solution passes, every mutant triggers at least one genuine test failure, each mutant changes at most 10 lines, results remain consistent across three runs, and the per-test T3 CPU budget and total runtime meet the limits. The expanded 69-module catalog has a 300-second total budget (override with `--budget`); individual T3 limits stay in each package. Results are written to `reports/ci.json`, with full reference reports for each problem also saved under `reports/`. Import and syntax errors do not count as valid mutant detections.

## Papers and selected core modules

| Paper ID | Paper | Core scope | Module IDs |
| --- | --- | --- | --- |
| 1 | GAE | Implement the backward GAE recurrence and episode boundary handling. | 1 |
| 2 | PPO | Implement the clipped surrogate and its gradient behavior; value loss, normalization, and diagnostics are supplied. | 2 |
| 3 | DPO | Implement the preference loss and implicit rewards; token alignment and sequence log-probability calculation are supplied. | 3 |
| 4 | DeepSeekMath · GRPO | Focus on the paper's GRPO module: group-relative advantages, clipped objectives, and response-level aggregation. | 4 |
| 5 | Dr. GRPO | Implement two corrections: remove group standard-deviation scaling and normalize by a fixed generation budget. | 5 |
| 6 | DDPM | Implement closed-form forward noising, the Gaussian posterior, and an epsilon-parameterized reverse step; schedules and sampling loops are supplied. | 6, 8 |
| 7 | Progressive Distillation | Implement v-parameterization conversions, then derive and validate the two-teacher-step to one-student-step distillation target. | 7, 26 |
| 8 | DDIM | Implement the DDIM update and clean endpoint; timestep selection, loops, and inversion exploration code are supplied. | 9 |
| 9 | Flow Matching | Implement the conditional path, target velocity, and CFM regression loss; time sampling, Euler integration, and training helpers are supplied. | 10 |
| 10 | GSPO | Implement sequence ratios and the GSPO loss; test masking, clipping regions, and gradients. | 11 |
| 11 | GKD · On-policy Distillation | Implement student rollouts and full-vocabulary divergence in a tiny autoregressive model. OPD here means on-policy distillation. | 12 |
| 12 | DAPO | Implement group filtering and the token-level objective; sampling infrastructure and distributed training are outside this module. | 13 |
| 13 | LoRA | Implement the low-rank update and linear forward pass; verify frozen parameters and a small regression experiment. | 14 |
| 14 | QLoRA | Implement blockwise NF4 quantization, dequantization, and adapter forward; double quantization and paged optimizers are outside this exercise. | 15 |
| 15 | MeanFlow | Implement the average-flow JVP target and gradient boundaries, then run a small generative experiment. | 16 |
| 16 | JiT | Implement data-to-velocity conversion and the denoising objective with a paper-derived embedded-manifold experiment. | 17 |
| 17 | Drifting Models | Implement the drifting field and stopped-gradient loss, and verify generator behavior on a small two-dimensional distribution. | 18 |
| 18 | DMD | Implement the score-difference target and paired regression term; inspect mode coverage using an analytic bimodal distribution. | 19 |
| 19 | DMD2 | Implement two-timescale updates and auxiliary objectives; verify parameter-update boundaries in a small experiment. | 20 |
| 20 | Consistency Models | Implement the boundary parameterization and adjacent-time consistency-distillation loss. | 21 |
| 21 | Transformer | Implement scaled dot-product attention, multi-head attention, LayerNorm and Add & Norm, sinusoidal positions, and position-wise FFN as five independent modules. | 22, 27–30 |
| 22 | AdamW | Implement moment estimates, bias correction, and a decoupled weight-decay step. | 23 |
| 23 | FlashAttention | Implement blockwise recurrence and rescaling. This CPU algorithm exercise does not benchmark GPU kernels. | 24 |
| 24 | CLIP | Implement feature normalization, temperature scaling, and bidirectional contrastive loss with a tiny paired-retrieval experiment. | 25 |
| 25 | GLM-5 | Implement head-wise Muon Split and double-sided importance corrections for asynchronous RL. | 31–32 |
| 26 | Kimi K3 | Implement bounded KDA recurrence, Block Attention Residuals, and SiTU-GLU; recreate the report’s gate and activation curves. | 33–35 |
| 27 | OPSD | Dual-context causal alignment and pointwise divergence clipping. | 36, 37 |
| 28 | SDPO | Same-question feedback routing, reverse KL, and top-k tail-mass coarsening. | 38, 39 |
| 29 | REPA | Patch-wise representation alignment and gradient boundaries. | 40 |
| 30 | Rotation Trick | Rotation-based gradient transformation with quantized forward values. | 41 |
| 31 | Differential Transformer | Differential attention on projected tensors and lambda parameterization. | 42 |
| 32 | Block Diffusion | Single-forward attention masks for concatenated noisy and clean tokens. | 43 |
| 33 | DistiLLM-2 | Skew KL on teacher trajectories and skew reverse KL on student trajectories. | 44 |
| 34 | PKPO | Continuous-reward transformation, policy loss, and the paper’s one-dimensional Gaussian toy. | 45 |
| 35 | SVDQuant | Smoothing, SVD low-rank decomposition, and a 4-bit residual forward pass. | 46 |
| 36 | MoGe | Robust optimal estimation with truncated weighted L1 residuals. | 47 |
| 37 | DAPS | Linear-Gaussian posterior score, Langevin updates, and independent renoising. | 48 |
| 38 | DiffusionNFT | Positive/negative beta implicit velocities, the two-branch objective, and tiny training. | 49 |
| 39 | VAR | Next-scale conditioning and attention masks; multiscale residual quantization and reconstruction. | 50–51 |
| 40 | FD-Loss | EMA population moments, differentiable Fréchet distance, and stopped-gradient loss normalization. | 52–53 |
| 41 | VICReg | Variance, view invariance, and off-diagonal covariance penalties. | 54 |
| 42 | LeJEPA · SIGReg | Characteristic-function Gaussianity statistic and sliced aggregation. | 55 |
| 43 | VISReg | Scale, sliced shape, centering, and multi-view invariance. | 56 |
| 44 | I-JEPA | Masked teacher targets, prediction loss, and EMA updates. | 57 |
| 45 | LeWorldModel | Joint prediction objective and latent rollout; terminal cost and CEM elite refitting. | 58–59 |
| 46 | Causal-JEPA | Object masking with identity anchors and masked history/future prediction. | 60 |
| 47 | RMSNorm | Full and first-k RMS normalization, scale identities, prefix bias, and cross-feature gradients. | 61 |
| 48 | GQA | Group-wise KV projection conversion, contiguous query routing, shared-head gradients, and persistent cache capacity. | 62 |
| 49 | RoFormer / RoPE | Adjacent-pair rotations, relative-position logits, analytical gradients, and cached-decoding offsets. | 63 |
| 50 | TIGER | Frozen residual quantization, collision suffixes, unconstrained beam search, and valid full-ID catalog lookup. | 64–65 |
| 51 | ColBERT | Normalized token MaxSim, masking, pairwise ranking and synthetic query-vector training. | 66 |
| 52 | SASRec | Causal/padding history visibility, sampled next-item binary loss and history-dependent synthetic training. | 67 |
| 53 | BPR | Per-user unobserved candidates, paired matrix-factorization objective/global prior and personalized synthetic training. | 68 |
| 54 | LightGCN | Symmetric bipartite normalization, linear neighbor propagation, uniform layer aggregation and actual initial-embedding training. | 69 |

RMSNorm, GQA and RoPE are part of LLM Basics. Each includes independent gradient checks, complete CPU/WASM judging and supplied synthetic experiments. See [public-core validation](docs/validation/llm-foundations-public.md).

The homepage groups papers into seven categories. New modules include CPU toy experiments, source/adaptation notes, and T0–T3 checks. Lab distinguishes reference plots from your submitted code’s T3 results. This is selected core practice; full model training and GPU kernel benchmarking remain outside scope.

Each module declares `module.required_functions` and `module.provided_functions`. The starter leaves only required functions unfinished, and the judge exercises them through the supplied supporting code. All modules include bilingual statements and notes, reference implementations, layered tests, and buggy variants. Module IDs 6 and 7 do not require T3 tests; module 5's length trend is illustrative and is not graded.

## Visualizations and static builds

The repository includes pre-generated data, so you can start it directly. The homepage reads the small `data/index.json`; opening a module loads its content-addressed `data/modules/` detail file with statements, references and visualization data. Detail files can be cached across visits. The full `data/catalog.json` remains available for CLI consumers and older clients. Re-export after changing problems or experiments:

```bash
paperleet vizdata all
paperleet export
cd web
npm run check
npm run build
```

`vizdata` uses the reference implementations, a fixed seed, and two PyTorch CPU threads. Each problem's standalone data file is limited to 500 KB. Concept diagrams that can be computed in the browser update immediately as you move their sliders. `npm run check` verifies chart rendering for all maintained modules, trajectory decoding, report charts, completion status, private workspace isolation, draft conflicts, and accepted-result preservation.

The homepage defers the Python editor, Markdown/math renderer, community, account pages, paper submissions, and labs until they are needed. The initial JavaScript graph is approximately 370 KB uncompressed / 118 KB gzip, down from 1.42 MB / 450 KB; these figures exclude CSS, catalog data, and the separately loaded Python runtime. `npm run build` checks the actual manifest graph against a 400 KiB raw / 130 KiB gzip budget and prevents these optional features from becoming initial imports. A failed feature offers retry and page reload controls. If the rich editor fails, plain-text mode preserves the current page's code and supports editing and judging, plus account draft synchronization when signed in; the private-workspace read-only gate still applies. A browser may cache a failed module import, in which case use page reload to retry it; successfully saved guest code is restored after that reload.

GitHub Pages builds the browser-practice edition using `VITE_PUBLIC_SITE=1` and `VITE_APP_EDITION=local`. It supports editing and browser judging without an API. Ordinary local builds also support the loopback native judge. Online account synchronization is provided by the separate private application.

For a standalone static site, run `VITE_PUBLIC_SITE=1 npm run build` from `web/`;
no backend address is required. Enable an external account application with
`VITE_HOSTED_APP_AVAILABLE=1` and its `VITE_HOSTED_APP_URL` HTTPS origin. A hosted
frontend can use `VITE_API_ORIGIN` for a separate HTTPS API origin, or leave it
empty for same-origin requests. Builds reject invalid enabled origins before
emitting JavaScript; diagnostics name the setting without printing its value.
See [standalone deployment validation](docs/validation/standalone-startup.md).

## Repository layout and contributing

```text
paperleet/         CLI, pytest runner, local service, shared mathematical utilities
papers.yaml        Curated papers, contributions, scope, and module ordering
problems/          Self-contained core module packages (RL, fine-tuning, generative models, architecture, optimization, representation)
web/               Vite + React + TypeScript
tests/             Judge framework integration tests
workspace/         Personal code, ignored by Git
reports/           Judge and CI reports, ignored by Git
.paperleet/        Local judge queue and paper submissions, ignored by Git
```

See the [contributing guide (in Chinese)](CONTRIBUTING.md) for the problem authoring workflow and the [10-problem pilot specification (in Chinese)](PaperLeet_第一步_10题试点规格.md) for detailed acceptance criteria. GitHub Actions runs framework tests, three rounds of quality checks, data generation, and the frontend build, then uploads reports and static site build artifacts.

See the [pilot validation record (in Chinese)](docs/validation/pilot.md) for measured results and the scope of validation for this implementation.

See the [expanded curriculum validation record](docs/validation/core-curriculum.md) for the 26-paper release, toy provenance, browser checks, and harness validation.

The Oral/Spotlight expansion adds 12 papers and 14 modules. See the [candidate evidence](docs/research/paper-candidates-2026-10-13.md) and [validation record](docs/validation/oral-spotlight-expansion.md) for sources, toy provenance, and checks.

VAR and FD-Loss add four focused modules with CPU experiments. See the [source and validation record](docs/validation/var-and-fd-loss.md).

Browser judging covers all 63 maintained teaching modules: Schedules, Attention, LoRA, parameter conversion, LayerNorm, positional encoding, DPO, multi-head attention, positionwise FFN, AdamW, GAE, PPO, GRPO, Dr. GRPO, DAPO, GSPO, CLIP, SiTU-GLU, QLoRA, Block Attention Residuals, differential attention, VICReg, Rotation Trick, LeWM joint prediction, LeWM CEM planning, Causal-JEPA, REPA, I-JEPA, DistiLLM-2, OPSD context alignment, OPSD pointwise clipping, SDPO feedback routing, SDPO top-k/tail KL, Muon Split, SVDQuant, FlashAttention numerical core, Kimi KDA, GLM-5 off-policy correction, OPD, Consistency Models, Progressive Distillation, DDPM reverse sampling, DDIM, Flow Matching, DAPS, SIGReg, VISReg, MoGe robust alignment, VAR next-scale prediction, VAR residual tokenization, Block Diffusion, PKPO, DiffusionNFT, FD population statistics, JiT, Drifting, DMD, DMD2, the differentiable FD Loss, MeanFlow, RMSNorm / pRMSNorm, GQA and RoPE. These modules run all 491 maintained tests in local NumPy/WASM, with analytical autodiff and optimizer training where required; the 257 maintained mutants are checked in per-module calibration before learner tests. The original experiments include QLoRA's 220-step adapter training, CLIP's 80-step paired retrieval training, VICReg's feature-geometry optimization, LeWM's 120-step joint training and 14-round planning, Causal-JEPA's 100-step trajectory probe and the complete RL teaching loops. **Run tests** stays in the browser; unsupported APIs and runtime failures are reported without automatic server fallback. Signed-in users choose **Submit** for server judging and saved solutions. Runtime assets load on demand; content-addressed shared code reduces active test-bundle downloads by 87.2% across the current catalog while retaining per-run calibration. Live steps show environment setup, reference/mutation calibration and completed learner tests, with Stop available throughout. Hosted account progress remains server-verified. See [implementation and compatibility notes](docs/browser-judge.md).
