# Torch runtime iteration 69: first-order functional VJP

Version 0.69 adds `torch.autograd.functional.vjp` for supported real first-order
operators. The callback runs once with distinct detached input objects. Existing
caller gradient flags, accumulated gradients and hooks remain intact; results
and derivatives are detached. Tensor/tuple structure, implicit scalar seeds,
unused-input zeros and strict-mode rejection are checked against native Torch.
An implicit NumPy escape inside the recorded callback raises an error; explicit
`detach()`, `no_grad()` and `numpy(force=True)` preserve deliberate disconnection.
Higher-order graphs, complex inputs, forward-AD nesting and `torch.func.vjp`
remain unsupported. This release adds no GPU backend.

Registered function attributes on `nn.Module` now survive shallow/deep copy and
pickle, including functional VJP and Transformer activations. Operator references
resolve to their owning distribution in a fresh process. Ordinary payloads do not
have arbitrary `__class__` properties evaluated during state handling. General
user-lambda serialization is outside this contract.

## Verification

- Full application and standalone CPU suite: **41804 passed, 1 skipped, 34 warnings in 496.54s (0:08:16)**.
- Current catalog: **86 modules**, **386 mutants** in actual Pyodide/WASM, no budget exceedance.
- Standalone SDK: **8,165 cases / 66 suites** in Node and Chrome WASM; full exported application also checks the current SDK suite.
- Native model qualification: **368 cases** (336 successful, 32 expected rejections), independently replayed on CPU, Node WASM and Chrome WASM. Coverage includes strided/aliased inputs, float32/float64, opposite seed dtype, linear/normalization/convolution/attention/linalg, mixed constant outputs and caller-state preservation. Maximum observed absolute difference: `2.5033950805664062e-06`.
- Actual wheel/sdist and private application wheel: exact 61 source/license identities, **8,178 installed tests** on Python 3.10.19 / NumPy 1.26.4 without native Torch; independent uninstall ownership, checkpoint resume and fresh-process callable restoration checked.
- **151 frontend checks**, WASM-driver and release-policy checks, and both local Node 22.23.3 build profiles. The full catalog WASM gate also uses Node 22.23.3; iteration 68 used Node 25, so their aggregate gate times are not a matched performance comparison.
- Preserved **13,909** immutable cached assets; added **174** reachable assets; checked **177** live export/index identities.
- Baseline application commit: `44e5ccc362b64393e908074cf74063bcf5142052`. No CI workflow or deployment was requested by these local checks. Main adoption is recorded separately.

## Performance and tradeoffs

18 fresh processes (6 per runtime), all 6 source-load permutations,12 balanced
common batches of at least 20 ms per flow and an identical-source A/A control.
Lower B/A means faster. Ratios below are medians of per-process ratios; they
are not ratios of independently pooled absolute times. The new VJP reference
is an **unchecked manual 0.68 AD wrapper**, not an existing functional-VJP API.

| Runtime | Flow | A/A | B/A |
| --- | --- | --- | --- |
| cpu | mha_step | 0.9947 | 0.9958 |
| cpu | groupnorm_step | 1.0076 | 1.0275 |
| cpu | linear_deepcopy | 0.9861 | 1.0119 |
| cpu | transformer_deepcopy | 1.0090 | 1.0152 |
| cpu | transformer_pickle | 1.0039 | 1.0182 |
| cpu | export_numpy | 1.0040 | 1.2088 |
| cpu | export_asarray | 1.0115 | 0.9940 |
| cpu | vjp_scalar | 1.0011 | 1.1240 |
| cpu | vjp_linear | 0.9908 | 1.0911 |
| node | mha_step | 1.0121 | 1.0073 |
| node | groupnorm_step | 0.9996 | 1.0003 |
| node | linear_deepcopy | 1.0200 | 1.0009 |
| node | transformer_deepcopy | 0.9932 | 1.0033 |
| node | transformer_pickle | 0.9966 | 1.0257 |
| node | export_numpy | 0.9973 | 1.3383 |
| node | export_asarray | 1.0134 | 1.0439 |
| node | vjp_scalar | 0.9791 | 1.0608 |
| node | vjp_linear | 0.9989 | 1.0730 |
| chrome | mha_step | 1.0019 | 0.9796 |
| chrome | groupnorm_step | 0.9910 | 1.0067 |
| chrome | linear_deepcopy | 1.0052 | 1.0031 |
| chrome | transformer_deepcopy | 1.0015 | 1.0163 |
| chrome | transformer_pickle | 0.9938 | 1.0106 |
| chrome | export_numpy | 1.0071 | 1.3617 |
| chrome | export_asarray | 1.0056 | 1.0131 |
| chrome | vjp_scalar | 1.0017 | 1.0552 |
| chrome | vjp_linear | 1.0161 | 1.1114 |

The initial candidate regressed copy/pickle latency by12–29%. Moving registered
function conversion to a type-filtered state path reduced those medians to
roughly 0–3% versus0.68. Initial and intermediate results remain in the raw archive.
Default `Tensor.numpy()` still incurs approximately21%CPU /34%Node /36%Chrome
relative overhead; the difference between absolute per-runtime medians is about
25ns /110ns /101ns. `np.asarray` medians differ by about-0.6% /+4.4% /+1.3%.
Public functional VJP costs about6–12% more than the unchecked manual wrapper,
including validation and caller-state isolation. Training and A/A outliers are
retained. No universal speedup or native-Torch/GPU performance claim is made.

The original timing environment disappeared after 11 successful children, before
the next process launched. Those 11 results were preserved. The remaining 7 ran in
a new environment with the same Python 3.12.11 and identical NumPy 2.5.4 files;
the interruption and resume runner are archived. No timing child overlapped
our package or full application gates.

## Reproducible evidence

Machine report: `packages/stridetorch/benchmarks/results/iteration-69.json`.
Raw archive: `iteration-69-harness.tar.gz` (7,214,748 bytes), SHA256
`26cc3b5b213f5527c3091f36038c1979d1105899cc7fced531e1894d7aa19b7d`. All 2328 logical payloads were verified after
writing; identical-content hardlinks preserve permissions/ownership. File
timestamps are not comparison inputs. Rejected prototypes and their logs are
historical evidence. Scripts retain original absolute paths and require path
adjustment, Python/NumPy/nativeTorch, pinned Pyodide and browser dependencies.
The selected source is under `versioned-release/snapshot`; final timing is
`performance-selected`, while `performance` records the slower initial candidate.
The independently named library remains StrideTorch pending a naming decision.

## Main adoption

Version 0.69.0 was adopted on main at `fc08faec7363620d5d0f57674b13a0eff905a423`, including implementation `7a4e7117f9dd14ca2f61dad97898a3a52fbcb159` and the qualified current UI (153 frontend checks and both local build profiles). A guarded ordinary push passed with no active release chain. The remote ref and 956 Git blobs were checked; all 13,909 earlier immutable asset tree entries remain identical. The adopted SHA had zero Actions runs at verification. No workflow dispatch, tag, package publication or deployment was initiated. [Adoption receipt](../../packages/torchwasm/benchmarks/results/iteration-69-main-adoption.json).
