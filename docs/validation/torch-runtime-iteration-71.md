# Torch runtime iteration 71: matrix-free Hessian-vector products

Version 0.71 adds `torch.autograd.functional.hvp(func, inputs, v=None,
create_graph=False, strict=False)` for supported real float32/float64 operators.
It returns the detached scalar objective and its Hessian-vector product without
assembling a dense Hessian. Tensor/tuple input structure is preserved; the
callback executes once with distinct detached input leaves, including repeated
arguments. Original data, gradient flags, `.grad` buffers and tensor hooks remain
intact. As with native functional AD, callbacks may update their own buffers:
training BatchNorm updates its tracked running statistics once.

The implementation differentiates a forward directional result in reverse.
LayerNorm, training BatchNorm/InstanceNorm, linear solves, slogdet and logdet
retain the dependencies needed for curvature. Ordinary JVP keeps its existing
native-compatible saved-statistic/factor behavior outside the scoped HVP path.
Near-singular determinant curvature uses cofactor products without division by
singular-value gaps. The general determinant rule uses quadratic matrix storage;
the nonsingular 2×2 path uses a direct polynomial expression.

There is one deliberate numerical extension: native higher-order SVD tapes can
produce NaNs at repeated singular values even though determinant is polynomial.
This library returns the finite polynomial curvature, checked with an independent
minor/cofactor chain-rule reference. These cases are identified separately below;
they are not claimed as exact native-output matches.

`strict=True` rejects independent objectives or constant derivatives as qualified
against native Torch 2.14.1. Omitted `v` requires a single one-element input;
supplied vectors follow input structure and are detached and converted to the
corresponding input dtype. `create_graph=True`, complex inputs, nested forward AD
and unsupported forward rules reject explicitly. Deliberate `detach()`,
`no_grad()` and `numpy(force=True)` disconnect the graph; implicit NumPy export
during the recorded callback is rejected. Module HVP attributes survive copy,
deepcopy, pickle and fresh-process restoration in both distribution namespaces.
This version adds no GPU backend or `torch.func.hessian` API.

Scalar `clamp`, `clamp_min` and `clamp_max` preserve the production Torch 2.6
inclusive-boundary derivative of one. A full-application regression rejected the
earlier attempt to follow Torch 2.14 strict boundaries; all 32 piecewise fixtures
now replay an isolated native Torch 2.6.0 oracle. Generic
absolute-value JVP retains an explicit zero reverse derivative with respect to
its primal while preserving functional-HVP constant-derivative strictness.

## Verification

- Full current application plus standalone CPU suite: **41989 passed, 1 skipped, 34 warnings in 430.69s (0:07:10)**.
- Actual Pyodide application catalog: **86 modules / 386 mutants**, with no browser-budget exceedance.
- Exact selected source in Node and Chrome WASM: **8,350 SDK cases / 71 suites each**.
- Numerical and caller-state fixtures on CPU and independently replayed in Node and Chrome: **720 cases each**. Of these, **676** directly compare native results or expected errors, **4** use analytic/native-FP64 normalization precision controls, and **40** check the finite determinant polynomial extension described above. Groups cover 128 boundary cases, 128 caller-state cases, 208 model cases, 64 singular/empty determinant cases, 160 extended determinant cases and 32 piecewise cases.
- Models include linear layers, normalization, convolution, SDPA with the native math backend, captured MultiheadAttention, solve/inverse, Cholesky, determinant objectives, classifier, stateful BatchNorm and embedding. Shapes, dtypes, strided inputs, tuple structure, callback counts, existing gradients and hooks are checked where applicable. The other native fixture groups use Torch 2.14.1, git `5c4886908584029761b579af026dcfb627c84070`.
- Public wheel/sdist and private application wheel: **8,363 installed SDK cases** on Python 3.10.19 / NumPy 1.26.4 without native Torch. All 61 runtime/license resources and 37 application source identities match. Co-installation/uninstall ownership, explicit aliases, checkpoint resume and fresh-process activation/VJP/Jacobian/HVP restoration passed.
- **156 frontend commands**, WASM-driver and release-policy checks, two Node 22.23.3 local build profiles and exact generated Worker derivation from its tracked source graph.
- **14,257** older immutable assets retained; **174** new reachable assets and **177** live export/index identities verified.
- Application baseline: `4f9ffcdb545dda8c57b70ba1e0ea76ba502e7841`. Main adoption is recorded separately after the guarded push.

## Latency and allocation

The latency run uses 18 fresh processes: six each for CPU, Node WASM and Chrome
WASM, all six source-load orders, twelve balanced common batches of at least
20 ms per flow, and an identical-source A/A control. Each child runs 20 flows
after numerical, dtype, shape, alias and caller-state checks. Seventeen flows
compare existing public 0.70 and 0.71 APIs, including public VJP and Jacobian.
The three HVP flows use an analytically checked composition of released 0.70
public JVP/grad primitives as reference; 0.70 had no public functional HVP API.

Ratios are medians of per-process ratios, not ratios of pooled absolute medians.
Lower B/A means faster. Across the old APIs, measured medians remain close to
same-source control variation; this is a bounded observation, not a universal
no-regression guarantee.

| Runtime | Existing API B/A range | Same-source A/A range |
| --- | --- | --- |
| cpu | 0.9947–1.0082 | 0.9915–1.0088 |
| node | 0.9913–1.0224 | 0.9855–1.0245 |
| chrome | 0.9858–1.0168 | 0.9896–1.0097 |

The HVP entry point has measured runtime changes of 2.1% to 5.7% relative
to its composed reference across these three flows and runtimes. The measurements do not isolate the cause of that difference.
Absolute values below are pooled median microseconds, reported alongside the
separately computed paired ratios.

| Runtime | HVP flow | A/A | B/A | B/AA | Reference → candidate µs |
| --- | --- | --- | --- | --- | --- |
| cpu | hvp_linear | 1.0023 | 1.0337 | 1.0289 | 164.77 → 170.02 |
| cpu | hvp_scalar | 0.9977 | 1.0240 | 1.0337 | 124.44 → 126.81 |
| cpu | hvp_unused | 0.9999 | 1.0507 | 1.0453 | 57.38 → 59.30 |
| node | hvp_linear | 0.9849 | 1.0210 | 1.0366 | 323.63 → 332.56 |
| node | hvp_scalar | 1.0066 | 1.0243 | 1.0171 | 252.45 → 259.43 |
| node | hvp_unused | 1.0030 | 1.0494 | 1.0448 | 115.91 → 122.49 |
| chrome | hvp_linear | 1.0049 | 1.0306 | 1.0256 | 316.80 → 328.91 |
| chrome | hvp_scalar | 1.0058 | 1.0379 | 1.0358 | 243.95 → 252.93 |
| chrome | hvp_unused | 1.0010 | 1.0569 | 1.0525 | 116.31 → 122.95 |

A separate CPU `tracemalloc` experiment checks three float64 sin objectives,
six balanced call orders and a dense duplicate A/A control. Its dense reference
assembles the full Hessian from the **same selected HVP implementation's basis
calls** and then multiplies by the vector. Analytic `-sin(x) * v` and caller state
are checked. This tests the storage advantage of direct products, not a distinct
released Hessian API or an independent derivative implementation.

| Input shape | Dense Hessian bytes | Direct / dense peak allocation | Dense A/A peak |
| --- | --- | --- | --- |
| [8, 8] | 32,768 | 0.1328 | 1.0000 |
| [8, 16] | 131,072 | 0.0605 | 1.0000 |
| [16, 16] | 524,288 | 0.0282 | 1.0000 |

Direct products use 2.8%–13.3% of the dense reference's tracked peak allocation
in these cases. These are Python/NumPy CPU allocations from one process, not
RSS, WASM heap, GPU memory, latency or a universal gain. Our heavy package tests,
application gates and allocation experiment did not overlap the latency children.

## Evidence and remaining scope

Machine report: `packages/stridetorch/benchmarks/results/iteration-71.json`.
Raw archive: `iteration-71-harness.tar.xz` (4,208,420 bytes), SHA256
`6a8e3459ffc80390a6d21217ccb2b347f312008c9b88d138c294734f92c2ffa5`. Every one of 4276 logical payloads was verified
after writing. Content-identical hardlinks require matching mode and ownership;
timestamps are not comparison inputs. The archive preserves original failed
prototypes, native and analytic fixtures, exact source/test snapshots, package
artifacts, latency samples and current-application gate logs. Scripts retain
absolute paths and require dependency setup and path adjustment for replay.

Only `production-release/snapshot/packages/stridetorch` is the selected final
source. Earlier snapshots and failed native-WASM NaN-mask attempts retain their
own original identities. General nonfinite piecewise behavior remains a future
audit target. GPU suitability work remains separate; this version enables no
GPU dispatch. No CI workflow, tag, package publication or deployment was initiated
by these checks. The library remains named StrideTorch pending a naming decision.

The evidence container was losslessly recompressed from gzip to XZ for transfer. The TAR stream SHA256 is `8e8a50c6160ede856971ae5257aec83e65be051b110d303eadb9303e206d4f24`. All 4,276 logical payloads and their mode/ownership metadata were checked; runtime sources, tests and qualification results are unchanged. The original gzip SHA256 is `945aeba437c7c9a7f0abd31d770cc9cf60ccc279d00f7934f8f98882a5639c10`.
