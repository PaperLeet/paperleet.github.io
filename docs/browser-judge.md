# CPU/WASM browser judge prototype

Ten validated modules use the browser's CPU through Pyodide + NumPy. **No WebGPU or GPU is required.** The editor offers **Auto (browser first)**, **Browser only**, and **Server**. Other modules keep the server queue. Run `make dev` and open `http://127.0.0.1:5173/#/architecture/attention`.

| Module | Backend | Maintained suite | Calibration |
| --- | --- | --- | --- |
| Schedules | NumPy/WASM tensor adapter | 8/8, including genuine float64 and variance preservation | Reference 8/8; all 5 mutants caught |
| Attention | NumPy/WASM reverse-mode autodiff | 7/7, including masked rows, independent q/k/v gradients and prefix averaging | Reference 7/7; all 3 mutants caught |
| LoRA | NumPy/WASM reverse-mode autodiff + Adam | 6/6, including frozen-base/input gradients, rank/merge checks and the 300-step training experiment | Reference 6/6; all 4 mutants caught |
| Parameter conversion | NumPy/WASM + autodiff | 7/7, including endpoint clamps and float64 round trips | All 4 mutants caught |
| LayerNorm | NumPy/WASM + analytical library-oracle VJP | 7/7, including token independence, affine gradients and Add & Norm | All 4 mutants caught |
| Positional encoding | NumPy/WASM | 7/7, including odd widths, offsets, empty sequences and rotations | All 4 mutants caught |
| DPO | NumPy/WASM + autodiff/Adam | 9/9, including shifted masked log-probs, extreme margins and preference training | All 5 mutants caught |
| Multi-head attention | NumPy/WASM + independent SDPA oracle/VJP | 7/7, including all seven input/projection gradients, empty masked rows and two-head routing | All 4 mutants caught |
| Positionwise FFN | NumPy/WASM + ReLU/autodiff/Adam | 7/7, including all five gradients and the unchanged 300-step nonlinear regression experiment | All 4 mutants caught |
| AdamW | NumPy/WASM + independent stateful optimizer oracle | 7/7, including bias correction, decoupled decay, changing learning rates and the 200-step quadratic | All 4 mutants caught |

These are bounded torch-like teaching adapters, **not the native PyTorch wheel or a general replacement for PyTorch**. Existing `import torch`, function signatures, maintained pytest files, reference implementations, mutants and experiment loops are retained. Test counts and precision tolerances are unchanged. Every browser run calibrates its runtime against the reference and every maintained mutant before executing the captured learner source. Unsupported APIs do not count as successful mutation detection. Ordinary assertion failures, syntax errors and unfinished TODOs are learner feedback and do not trigger an automatic server run.

## Differentiation and compatibility

`paperleet/browser/numpy_autograd.py` implements analytical vector-Jacobian products for the supported modules: batched/broadcast matmul, supported two-input einsum, elementwise arithmetic, stable softmax, masks/where, reductions, indexing (including repeated indices), reshape/transpose/permute, expansion and repeat. Leaf gradients accumulate through `backward`; `autograd.grad` returns derivatives without accumulating into leaves. `detach` and `no_grad` disconnect the relevant paths. LoRA uses genuine gradients and Adam moments/bias correction on the original 300-step loop. NumPy supplies CPU linear algebra, including the matrix-rank check.

The independent parity tests compare the adapter and **native PyTorch on identical arrays** for values, batched/vector/broadcast matmul gradients, masked attention gradients, einsum, repeated indices, views, accumulation and Adam steps in float32/float64. CPU MT19937 integer sampling and float32/float64 uniform conversion are explicitly matched to native PyTorch and verified on 20,000 preference pairs. This keeps the unchanged DPO convergence threshold meaningful. Normal samples remain deterministic NumPy draws and are not bitwise identical to PyTorch; browser and native normal-data training curves need not match point for point. Both must meet unchanged experiment thresholds. Stable log-sigmoid/log-softmax, gather gradients, clamp/trigonometric derivatives, variance/std, pairwise distances and the separate LayerNorm oracle also have independent native checks.

The multi-head attention suite's library oracle uses raw NumPy SDPA and its own analytical VJP rather than the learner's Tensor softmax/matmul graph. Native parity covers boolean/additive masks, completely masked rows, broadcast batches, causal/non-square inputs, custom scales, empty query/key sequences, and additive-mask gradients in both precisions. Dropout, grouped-query attention and combined causal/mask options still require the server. Concatenation/stacking preserve gradients through duplicate and strided inputs; ReLU uses a zero derivative at zero. AdamW's separate stateful NumPy oracle exposes per-parameter steps and moments; differential tests cover learning-rate changes, decoupled decay, missing gradients and zeroing behavior. Advanced optimizer options and parameter groups remain unsupported.

Clamp uses subgradient 1 at the bounds, matching the hosted worker's pinned PyTorch 2.6 contract ([upstream implementation](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/autograd/FunctionsManual.cpp)). The local installed PyTorch version chooses differently at these nondifferentiable kinks, so an explicit production-contract test covers the bounds and differential tests compare smooth points.

Higher-order derivatives, CUDA, custom extensions, unimplemented PyTorch modules/operators and differentiable in-place editing are outside this backend. Known unsupported APIs raise `BrowserUnsupported`. Auto mode falls back to the native server queue, using the **same captured source** even if the user continues editing. Browser only reports incompatibility without submitting to the server. Server can always be selected explicitly. PPO/GAE still require unimplemented training/environment dependencies, and other unvalidated modules remain on the server. The supported subset will not accept every valid PyTorch implementation; general PyTorch compatibility and cross-browser/mobile coverage remain beyond this prototype.

## Isolation and account progress

Learner Python executes in a disposable classic Worker inside an opaque-origin `sandbox="allow-scripts"` iframe. It has no DOM or site storage access. The iframe CSP permits only self-hosted public runtime assets, excluding `/api` and third-party destinations. Messages are checked against the frame and a fresh nonce. Stopping, changing accounts, unmounting or exceeding the deadline removes the frame and worker. Phase limits are 90 seconds for initialization/calibration, 30 seconds for execution, and 120 seconds total; bundle downloads are bounded to 30 seconds. The UI remains responsive while Python or training runs.

Hosted browser results are local feedback, held in memory and discarded on account changes/refresh. They never update authoritative account progress, shared statistics or rankings. Signed-in users can select **Verify on server**; that server executes and records its own result. Local development can retain browser completions locally. Existing server authentication, private draft synchronization, draft conflicts and durable queue recovery remain in force.

## Runtime assets and deployment

`npm run dev` and `npm run build` run `scripts/prepare-browser-runtime.mjs` first. It copies Pyodide **0.29.4** from the exact npm dependency and downloads NumPy **2.2.5**, pytest **8.3.5** and dependencies using the official Pyodide lockfile SHA-256 digests. Cached wheels are verified and reused; initial preparation requires network access. Build failures are explicit. No GPU runtime is downloaded or required.

The runtime is self-hosted in `web/public/browser-runtime/` (ignored by Git), loaded only when a browser run is requested. The homepage does not preload Python/NumPy. Versioned runtime assets are immutable; worker scripts revalidate. `paperleet export` emits content-addressed compatibility bundles from the maintained Python files. Re-export after changing adapters, the harness, tests or reference solutions. Vite dev/preview and Caddy provide CORS only on the public runtime/worker asset paths so the opaque sandbox can load them.

Upstream projects and license information: [Pyodide](https://github.com/pyodide/pyodide), [NumPy](https://github.com/numpy/numpy), [pytest](https://github.com/pytest-dev/pytest). Archives retain upstream license metadata. The bounded autodiff adapter is implemented in this repository; it does not depend on a third-party PyTorch port.

## Reproducible checks

- `.venv/bin/python -m pytest tests/test_browser_judge.py tests/test_browser_autograd.py -q`: unchanged source bundles, complete reference/mutant calibration, false-positive prevention, syntax/unsupported API behavior and independent native PyTorch parity. Adapter installation for full suites is isolated in subprocesses so it cannot replace the server's native torch.
- `cd web && npm run check`: transport isolation, restricted network, message validation, report parsing, assertion failures, stop/timeout cleanup and all ten CPU/WASM routes.
- `cd web && npm run build`: TypeScript and production assets.
- Browser: select **Browser only** and run each reference; all 72 tests must pass, including LoRA/FFN training and AdamW optimization. The non-frozen-base LoRA mutant must fail the gradient test (5/6), while unsupported APIs in Auto mode retain native server judging. Run the generated reports' actual experiment charts through the Lab UI.

The paper library exposes **Runs in browser** and **Server judging** filters. Runtime capability intersects search/difficulty at the module level and also scopes random selection. Paper and module cards display browser capability, while anonymous onboarding explains when immediate browser runs are available.
