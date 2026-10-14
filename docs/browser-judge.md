# CPU/WASM browser judge prototype

The three pilots use the browser's CPU through Pyodide + NumPy. **No WebGPU or GPU is required.** The editor offers **Auto (browser first)**, **Browser only**, and **Server**. Other modules keep the server queue. Run `make dev` and open `http://127.0.0.1:5173/#/architecture/attention`.

| Module | Backend | Maintained suite | Calibration |
| --- | --- | --- | --- |
| Schedules | NumPy/WASM tensor adapter | 8/8, including genuine float64 and variance preservation | Reference 8/8; all 5 mutants caught |
| Attention | NumPy/WASM reverse-mode autodiff | 7/7, including masked rows, independent q/k/v gradients and prefix averaging | Reference 7/7; all 3 mutants caught |
| LoRA | NumPy/WASM reverse-mode autodiff + Adam | 6/6, including frozen-base/input gradients, rank/merge checks and the 300-step training experiment | Reference 6/6; all 4 mutants caught |

These are bounded torch-like teaching adapters, **not the native PyTorch wheel or a general replacement for PyTorch**. Existing `import torch`, function signatures, maintained pytest files, reference implementations, mutants and experiment loops are retained. Test counts and precision tolerances are unchanged. Every browser run calibrates its runtime against the reference and every maintained mutant before executing the captured learner source. Unsupported APIs do not count as successful mutation detection. Ordinary assertion failures, syntax errors and unfinished TODOs are learner feedback and do not trigger an automatic server run.

## Differentiation and compatibility

`paperleet/browser/numpy_autograd.py` implements analytical vector-Jacobian products for the operations used by Attention and LoRA: batched/broadcast matmul, supported two-input einsum, elementwise arithmetic, stable softmax, masks/where, reductions, indexing (including repeated indices), reshape/transpose/permute, expansion and repeat. Leaf gradients accumulate through `backward`; `autograd.grad` returns derivatives without accumulating into leaves. `detach` and `no_grad` disconnect the relevant paths. LoRA uses genuine gradients and Adam moments/bias correction on the original 300-step loop. NumPy supplies CPU linear algebra, including the matrix-rank check.

The independent parity tests compare the adapter and **native PyTorch on identical arrays** for values, batched/vector/broadcast matmul gradients, masked attention gradients, einsum, repeated indices, views, accumulation and Adam steps in float32/float64. The adapter's seeded NumPy random stream is deterministic but is not bitwise identical to PyTorch's RNG; browser and native training curves therefore need not match point for point. Both must meet the unchanged experiment thresholds.

Higher-order derivatives, CUDA, custom extensions, unimplemented PyTorch modules/operators and differentiable in-place editing are outside this backend. Known unsupported APIs raise `BrowserUnsupported`. Auto mode falls back to the native server queue, using the **same captured source** even if the user continues editing. Browser only reports incompatibility without submitting to the server. Server can always be selected explicitly. The supported subset will not accept every valid PyTorch implementation; general PyTorch compatibility and cross-browser/mobile coverage remain beyond this prototype.

## Isolation and account progress

Learner Python executes in a disposable classic Worker inside an opaque-origin `sandbox="allow-scripts"` iframe. It has no DOM or site storage access. The iframe CSP permits only self-hosted public runtime assets, excluding `/api` and third-party destinations. Messages are checked against the frame and a fresh nonce. Stopping, changing accounts, unmounting or exceeding the deadline removes the frame and worker. Phase limits are 90 seconds for initialization/calibration, 30 seconds for execution, and 120 seconds total; bundle downloads are bounded to 30 seconds. The UI remains responsive while Python or training runs.

Hosted browser results are local feedback, held in memory and discarded on account changes/refresh. They never update authoritative account progress, shared statistics or rankings. Signed-in users can select **Verify on server**; that server executes and records its own result. Local development can retain browser completions locally. Existing server authentication, private draft synchronization, draft conflicts and durable queue recovery remain in force.

## Runtime assets and deployment

`npm run dev` and `npm run build` run `scripts/prepare-browser-runtime.mjs` first. It copies Pyodide **0.29.4** from the exact npm dependency and downloads NumPy **2.2.5**, pytest **8.3.5** and dependencies using the official Pyodide lockfile SHA-256 digests. Cached wheels are verified and reused; initial preparation requires network access. Build failures are explicit. No GPU runtime is downloaded or required.

The runtime is self-hosted in `web/public/browser-runtime/` (ignored by Git), loaded only when a browser run is requested. The homepage does not preload Python/NumPy. Versioned runtime assets are immutable; worker scripts revalidate. `paperleet export` emits content-addressed compatibility bundles from the maintained Python files. Re-export after changing adapters, the harness, tests or reference solutions. Vite dev/preview and Caddy provide CORS only on the public runtime/worker asset paths so the opaque sandbox can load them.

Upstream projects and license information: [Pyodide](https://github.com/pyodide/pyodide), [NumPy](https://github.com/numpy/numpy), [pytest](https://github.com/pytest-dev/pytest). Archives retain upstream license metadata. The bounded autodiff adapter is implemented in this repository; it does not depend on a third-party PyTorch port.

## Reproducible checks

- `.venv/bin/python -m pytest tests/test_browser_judge.py tests/test_browser_autograd.py -q`: unchanged source bundles, complete reference/mutant calibration, false-positive prevention, syntax/unsupported API behavior and independent native PyTorch parity. Adapter installation for full suites is isolated in subprocesses so it cannot replace the server's native torch.
- `cd web && npm run check`: transport isolation, restricted network, message validation, report parsing, assertion failures, stop/timeout cleanup and all three CPU/WASM routes.
- `cd web && npm run build`: TypeScript and production assets.
- Browser: select **Browser only** and run each reference; all 21 tests pass, including LoRA training. The non-frozen-base LoRA mutant must fail the gradient test (5/6), while unsupported APIs in Auto mode retain native server judging. Run the generated reports' actual experiment charts through the Lab UI.
