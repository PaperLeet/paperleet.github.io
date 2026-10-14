# PaperLeet

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="web/public/brand/paperleet-logo-dark.svg">
  <img src="web/public/brand/paperleet-logo-light.svg" alt="PaperLeet" width="320">
</picture>

**English** | [简体中文](README.zh-CN.md)

Read papers, implement the methods, and test your understanding through mathematical properties and actual behavior.

The pilot includes **46 papers and 60 focused core modules**. Start with a paper, read its contribution and selected scope, then implement its core modules in a LeetCode-style split workspace. Supporting code is supplied so each exercise focuses on the paper’s key mechanism. Completing a paper means passing its selected core modules. Paper organization lives in `papers.yaml`; statements, implementations, tests, and visualization data live in `problems/`.

## Run locally

The product is named **PaperLeet**. Use the `paperleet` CLI; the earlier `leetpaper` alias remains supported.

Requires Python 3.10+ and Node.js 20.19+. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
paperleet export
npm --prefix web ci
make dev
```

Open the URL shown in the terminal (default: **http://127.0.0.1:5173**). On subsequent runs, use **`make dev`** from the repository root to start both the website and the judge service. Press Ctrl-C to stop both processes.

## Host an invitation-only beta

A deployment package is available for a small online beta. Members can open the website from their own computers and solve the **60 curated modules**. The default institutional mode requires an invitation and verified institutional email. For an initial test group, use administrator-created accounts with `PAPERLEET_AUTH_MODE=test`: each tester receives a login username and access code, with no email registration, Supabase, or SMTP setup. See the [test account guide](docs/test-accounts.md). Hosted judging uses an isolated, resource-limited sandbox; drafts, results, and submission history are stored privately per account. Browser draft caches are separated by account, and conflicting edits from another device require an explicit choice before overwriting a draft.

Follow the [hosting guide](docs/hosting.md) to configure the server, sandbox, persistent storage, and your chosen account mode. Only institutional mode needs an email provider and invitation list. This repository supplies deployment files; it does not deploy a public instance automatically. The local `make dev` workflow below continues to run code on your own machine.

Without a rented server, use [Cloudflare Tunnel with local Docker](docs/cloudflare-tunnel.md) to connect your domain to the same hosted beta on your computer. The computer must stay awake and online; accounts and practice data remain in its private Docker volume.

## Solve your first problem

1. Choose a paper, such as **6 · DDPM**, then open its **Forward Noising** module.
2. Read the goal and complete only the required functions in the Python editor. Schedules and other supporting functions are already implemented.
3. Click **Run all tests** and wait for the test results to appear automatically.
4. Use each test layer's failure explanations, expected values, and actual values to revise your code, then run it again.

Passing the full test suite automatically marks the problem as complete and unlocks its notes on common pitfalls. Running only a subset of tests does not qualify as passing the full suite. In local mode, drafts and progress are saved in your current browser. In hosted mode, signed-in members sync drafts and judging results to their account. Use the Chinese / English toggle in the upper-right corner to switch the interface, problem statements, and explanations. The paper library uses numeric IDs 1–46. Module IDs 1–60 and internal directory names stay stable for existing drafts and CLI commands. A later failed run does not erase a previously accepted result. Existing drafts are preserved; use the editor reset action after saving your code if you want the new starter with supplied helpers.

Failed shared numerical assertions show the zero-based mismatch coordinate, expected/actual values, absolute error and test tolerances, or both tensor shapes when they differ. Full error details remain expandable. These optional diagnostics work with browser and server reports, survive refresh, and do not affect grading. NaN/Infinity experiment data remain readable failure evidence rather than breaking browser result transport; older reports retain their original error text.

Use **Light / Dark** in the top bar to choose an appearance. The first visit follows your system preference; an explicit choice is remembered in this browser. The paper library puts categories, search, filters, and module progress together; paper details contain the technical summary and expandable practice scope.

Drag the workspace dividers to resize the description/code columns and the code/results rows. Sizes are remembered in this browser. Double-click a divider or press Enter while it is focused to restore its default; arrow keys fine-tune the split. Run controls stay visible, and narrow screens stack the description above the editor.

The **Featured collections** carousel groups papers into LLM Basics, RL, Diffusion & Flow, Post-training, JEPA / World Models, and Visual Generation. Colorful cards show paper and problem counts; compact category chips below them filter the current collection. Collections can overlap and have shareable routes such as `#/collections/llm-basics` (existing `#/lists/...` links still work). Switching collections clears filters. Membership and bilingual card text live in `web/src/paperCollections.ts`.

**Random problem** opens an unattempted module within the current collection and filters. It excludes any saved run, accepted result, or edited draft; an unchanged starter is still eligible. The button waits for practice progress to load and is disabled when no eligible problems remain.

Hosted problem pages show public aggregate statistics: unique participants and solvers, total submissions, accepted and failed submissions, and acceptance rate (accepted / total submissions). Each account counts once per problem for participant and solver totals; repeated full submissions count separately. Partial test runs and judge-service failures are excluded. Statistics include existing server-judged history and persist with the database; local development runs do not contribute. No submissions yet means an unset acceptance rate, not a failed submission.

The website runs code through the judge service and displays results automatically, with no report files to manage manually. The local service listens on `127.0.0.1:8765`, and Vite proxies `/api` requests to it. Code is saved to `workspace/` and executed by Python/PyTorch on your machine. This local setup is intended for personal use. Use the separate hosted configuration for isolated multi-user judging; do not expose the development service directly.

## Community and accounts

Open **Community** to browse discussions, post and reply, like or repost, and follow other researchers. Public profiles show posts, replies, likes, reposts, followers, and following. In institutional mode, writing requires a verified email from an institution approved in `institutions.yaml`; the hosted beta additionally requires an administrator invitation. In test mode, members sign in with an administrator-issued username and access code. Their profiles are labeled **Test account** and do not claim institutional email verification.

Use **Discuss** in a practice workspace to browse or start discussions about that module, then **Back to practice** to resume your saved draft. Topic links also appear on associated posts and profile activity. Only text entered in the post composer is published; code drafts and judging reports are not attached automatically.

Sign-in returns you to the module or discussion you came from. Unpublished post drafts survive closing the composer, navigation, and refresh in the same tab, separately for each account and module. Use **Resume draft** to continue; these drafts stay in the tab and are not synced to other devices.

Community board, search, Following and page selections are stored in the page URL. Refreshing, sharing a link or returning from a post restores that list; **Back to discussions** also preserves whether you entered from the full community or a module. Sign-in keeps these filters. A result range appears above the list, unavailable pages recover to the last valid page, and empty searches offer **Clear filters**. These links contain only browsing filters; code and unpublished drafts are not included.

Reply drafts also survive navigation and refresh, separately for each account and post, including the selected reply target. Use **Write / Preview** to review Markdown before posting. Publishing disables editing, keeps failed submissions recoverable, and clears only the successfully submitted draft. A deleted target requires an explicit choice before posting; a delayed response cannot clear a newer draft. Draft storage failures are shown in the composer. Replies become public only after **Post reply** is selected.

Posts, replies, previews and profile reply activity render inline `$...$` and display `$$` formulas alongside highlighted fenced code blocks. Each code block offers **Copy code** with success/failure feedback; inline code stays literal. Both composers include expandable formatting examples. Long display formulas and code scroll within their own regions on narrow screens. Invalid formulas retain readable source, and public Markdown continues to block raw HTML and remote image loading. See [formatting validation](docs/validation/community-math-code.md).

Profiles include unique solved and attempted problem totals, difficulty breakdowns, a yearly activity calendar, active days and streaks, and recent accepted problems. The calendar uses the viewer's browser time zone and separates submission counts, distinct problems attempted or passed that day, and first-ever passes. Only server-judged full submissions count; partial tests and service failures do not. Repeated passes never increase the cumulative solved count. Community statistics include posts, replies, likes and reposts received, and follow relationships.

Display names and generated avatars are always public. Verified institution and country visibility are separate switches, both off by default; test accounts have no institution field, and full email addresses stay private. Country is optional and entered by the user, with no IP lookup. Turn off **Allow learning progress to be public** in account settings to restrict personal learning statistics to yourself. Code, drafts, and detailed reports always stay private. Local practice stays in the browser and is not uploaded to profiles. Per-problem statistics remain anonymous aggregate counts.

Institutional email sign-in uses Supabase Auth and requires server configuration. Test account sign-in uses separately provisioned accounts and does not use the email provider. Copy `.env.example` to `.env`, configure it, and explicitly export it in your shell before starting the service. See the [community setup guide](docs/community.md) for OTP templates, SMTP, institution review, and privacy rules. For an online beta, follow the [hosting guide](docs/hosting.md), which uses a separate isolated judge and invitation controls.

## Submit paper

In the **hosted beta**, signed-in members can submit paper proposals and read their own submissions; administrators can read all proposals. Generation, validation, and publication are performed by maintainers through the offline workflow. Hosted judging is limited to the 60 curated modules included in the deployment image.

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

`paperleet ci` checks that every reference solution passes, every mutant triggers at least one genuine test failure, each mutant changes at most 10 lines, results remain consistent across three runs, and the per-test T3 CPU budget and total runtime meet the limits. The expanded 60-module catalog has a 300-second total budget (override with `--budget`); individual T3 limits stay in each package. Results are written to `reports/ci.json`, with full reference reports for each problem also saved under `reports/`. Import and syntax errors do not count as valid mutant detections.

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

The homepage defers the Python editor, Markdown/math renderer, community, account pages, paper submissions, and labs until they are needed. The initial JavaScript graph is approximately 370 KB uncompressed / 118 KB gzip, down from 1.42 MB / 450 KB; these figures exclude CSS, catalog data, and the separately loaded Python runtime. `npm run build` checks the actual manifest graph against a 400 KiB raw / 130 KiB gzip budget and prevents these optional features from becoming initial imports. A failed feature offers retry and page reload controls. If the rich editor fails, plain-text mode preserves the current code and supports editing, saving, and judging; the private-workspace read-only gate still applies. A browser may cache a failed module import, in which case use page reload to retry it.

You can deploy `web/dist/` as a static preview for reading papers and viewing visualizations. Account workspaces, editing, synchronization, and judging require the configured `/api` service. Use the [hosting package](docs/hosting.md) for an interactive beta; no public service has been deployed by this repository automatically.

## Repository layout and contributing

```text
paperleet/         CLI, pytest runner, local service, shared mathematical utilities
papers.yaml        Curated papers, contributions, scope, and module ordering
institutions.yaml  Reviewed institutional email domains
problems/          Self-contained core module packages (RL, fine-tuning, generative models, architecture, optimization, representation)
web/               Vite + React + TypeScript
tests/             Judge framework integration tests
workspace/         Personal code, ignored by Git
reports/           Judge and CI reports, ignored by Git
.paperleet/        Local community database and paper submissions, ignored by Git
```

See the [contributing guide (in Chinese)](CONTRIBUTING.md) for the problem authoring workflow and the [10-problem pilot specification (in Chinese)](PaperLeet_第一步_10题试点规格.md) for detailed acceptance criteria. GitHub Actions runs framework tests, three rounds of quality checks, data generation, and the frontend build, then uploads reports and static site build artifacts.

See the [pilot validation record (in Chinese)](docs/validation/pilot.md) for measured results and the scope of validation for this implementation.

See the [expanded curriculum validation record](docs/validation/core-curriculum.md) for the 26-paper release, toy provenance, browser checks, and harness validation.

The Oral/Spotlight expansion adds 12 papers and 14 modules. See the [candidate evidence](docs/research/paper-candidates-2026-10-13.md) and [validation record](docs/validation/oral-spotlight-expansion.md) for sources, toy provenance, and checks.

VAR and FD-Loss add four focused modules with CPU experiments. See the [source and validation record](docs/validation/var-and-fd-loss.md).

Browser judging covers 58 modules: Schedules, Attention, LoRA, parameter conversion, LayerNorm, positional encoding, DPO, multi-head attention, positionwise FFN, AdamW, GAE, PPO, GRPO, Dr. GRPO, DAPO, GSPO, CLIP, SiTU-GLU, QLoRA, Block Attention Residuals, differential attention, VICReg, Rotation Trick, LeWM joint prediction, LeWM CEM planning, Causal-JEPA, REPA, I-JEPA, DistiLLM-2, OPSD context alignment, OPSD pointwise clipping, SDPO feedback routing, SDPO top-k/tail KL, Muon Split, SVDQuant, FlashAttention numerical core, Kimi KDA, GLM-5 off-policy correction, OPD, Consistency Models, Progressive Distillation, DDPM reverse sampling, DDIM, Flow Matching, DAPS, SIGReg, VISReg, MoGe robust alignment, VAR next-scale prediction, VAR residual tokenization, Block Diffusion, PKPO, DiffusionNFT, FD population statistics, JiT, Drifting, DMD and DMD2. These modules run all 447 maintained tests in local NumPy/WASM, with analytical autodiff and optimizer training where required; the 235 maintained mutants are checked in per-module calibration before learner tests. The original experiments include QLoRA's 220-step adapter training, CLIP's 80-step paired retrieval training, VICReg's feature-geometry optimization, LeWM's 120-step joint training and 14-round planning, Causal-JEPA's 100-step trajectory probe and the complete RL teaching loops. Unsupported APIs and runtime failures fall back to server judging. The editor offers Auto, Browser only and Server modes. Runtime assets load on demand; content-addressed shared code reduces active test-bundle downloads by 85.9% across the current catalog while retaining per-run calibration. Live steps show environment setup, reference/mutation calibration and completed learner tests, with Stop available throughout. Hosted account progress remains server-verified. See [implementation and compatibility notes](docs/browser-judge.md).
