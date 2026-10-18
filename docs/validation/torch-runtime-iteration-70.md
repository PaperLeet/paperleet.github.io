# Torch runtime iteration 70: functional Jacobians with bounded temporary storage

Version 0.70 adds `torch.autograd.functional.jacobian` for the supported real
first-order reverse-mode operators. Each returned matrix has shape
`output.shape + input.shape` and the input's dtype. Tensor/tuple nesting follows
the input and output structure. The callback executes once with distinct input
objects, including repeated arguments. Original input data, gradient flags,
existing `.grad` buffers and original tensor hooks remain intact. The callback
can still update its own model buffers, as native Torch does: training BatchNorm
updates its running statistics once.

The implementation allocates one final dense buffer per input and fills rows
directly. It avoids retaining every temporary gradient row before stacking.
Dense output size still scales with output elements times input elements.

`strict=True` checks each output against each input. Default mode returns zeros
for constant or unused paths. Implicit NumPy exports during the recorded
callback raise an error; `detach()`, `no_grad()` and `numpy(force=True)` preserve
deliberate disconnection. Native empty-input, empty-output and empty-tuple
boundaries are checked. Complex inputs, higher-order graphs, vectorized or
forward-mode Jacobians and nesting inside forward AD remain unsupported.
This change adds neither `torch.func.jacrev`/`jacfwd` nor a GPU backend.

Ordinary `from torch.autograd.functional import jacobian` works inside the
existing explicit alias context. Registered Jacobian attributes on modules
survive shallow/deep copy, both tested pickle protocols and fresh-process
restoration in the standalone and private distribution namespaces.

## Verification

- Current application plus standalone CPU suite: **41839 passed, 1 skipped, 34 warnings in 308.43s (0:05:08)**.
- Actual current Pyodide catalog: **86 modules / 386 mutants**, with no browser-budget exceedance.
- Exact final source in Node and Chrome WASM: **8,200 SDK cases / 68 suites each**, plus **168 independent native fixtures each** (72 boundary cases and 96 model cases).
- The same 168 fixtures replayed on final-version CPU source. Models cover linear, normalization, convolution, SDPA, captured MultiheadAttention, inverse, Cholesky, slogdet, classifier, stateful BatchNorm and embedding. Inputs cover float32/float64, contiguous/strided layout, strict mode, gradient-buffer identity, hooks, tuple structure and callback count. The model maximum observed absolute difference is `1.9073486328125e-06` against Torch `2.14.1`.
- Actual standalone wheel/sdist and application wheel: **8,213 installed SDK cases** on Python 3.10.19 / NumPy 1.26.4 without native Torch. All 61 runtime/license identities and 37 application identities match. Independent uninstall ownership, aliases, checkpoint resume and fresh-process VJP/Jacobian/activation restoration passed.
- **154 frontend commands**, WASM-driver and release-policy checks, both Node 22.23.3 local build profiles and exact worker-bundle regeneration from committed source inputs.
- **14,083** earlier immutable cached assets preserved; **174** new reachable assets; **177** live export/index identities checked.
- Application baseline: `6030cd46d8774f9e0e5c586b517f69dea92aa5b0`. Local gates do not imply main adoption; an adoption receipt is recorded separately.

## Latency and allocation tradeoffs

The initial comparison uses 18 fresh processes: six each for CPU, Node WASM
and Chrome WASM, all six source-load orders, twelve balanced common batches
of at least 20 ms per flow, and an identical-source A/A control. The nine legacy
flows compare public 0.69 and 0.70 APIs, including public VJP on both sides.
The three Jacobian flows compare a validated naive prototype with final 0.70;
they are **not** a comparison against a released 0.69 Jacobian API.

Values below are medians of per-process ratios; they are not ratios of pooled
absolute medians. Lower B/A means faster.

| Runtime | Flow | A/A | B/A | B/AA |
| --- | --- | --- | --- | --- |
| cpu | jacobian_linear | 0.9658 | 0.9534 | 0.9787 |
| cpu | jacobian_sin64 | 0.9954 | 0.9656 | 0.9777 |
| cpu | jacobian_unused | 0.9693 | 0.9121 | 0.9334 |
| cpu | export_asarray | 1.0114 | 0.9915 | 0.9836 |
| cpu | export_numpy | 0.9973 | 0.9873 | 0.9977 |
| cpu | groupnorm_step | 0.9922 | 0.9720 | 0.9826 |
| cpu | linear_deepcopy | 0.9962 | 1.0311 | 1.0126 |
| cpu | mha_step | 1.0291 | 1.0324 | 0.9981 |
| cpu | transformer_deepcopy | 1.0055 | 0.9888 | 0.9903 |
| cpu | transformer_pickle | 1.0213 | 1.0268 | 1.0186 |
| cpu | vjp_linear | 0.9922 | 1.0246 | 1.0407 |
| cpu | vjp_scalar | 1.0302 | 1.0017 | 0.9936 |
| node | jacobian_linear | 0.9860 | 0.9596 | 0.9757 |
| node | jacobian_sin64 | 1.0127 | 0.9314 | 0.9408 |
| node | jacobian_unused | 1.0103 | 0.8997 | 0.8800 |
| node | export_asarray | 1.0014 | 1.0177 | 1.0266 |
| node | export_numpy | 0.9945 | 0.9975 | 1.0000 |
| node | groupnorm_step | 0.9967 | 0.9960 | 0.9880 |
| node | linear_deepcopy | 0.9799 | 1.0012 | 1.0031 |
| node | mha_step | 0.9997 | 0.9606 | 0.9663 |
| node | transformer_deepcopy | 0.9843 | 0.9917 | 1.0150 |
| node | transformer_pickle | 0.9947 | 0.9886 | 1.0074 |
| node | vjp_linear | 1.0093 | 1.0242 | 0.9961 |
| node | vjp_scalar | 0.9646 | 1.0188 | 1.0433 |
| chrome | jacobian_linear | 1.0353 | 0.9740 | 0.9392 |
| chrome | jacobian_sin64 | 1.0102 | 0.9904 | 0.9631 |
| chrome | jacobian_unused | 0.9962 | 0.9155 | 0.9364 |
| chrome | export_asarray | 1.0122 | 1.0096 | 0.9965 |
| chrome | export_numpy | 1.0034 | 1.0082 | 0.9992 |
| chrome | groupnorm_step | 0.9762 | 1.0032 | 1.0375 |
| chrome | linear_deepcopy | 1.0007 | 0.9948 | 1.0073 |
| chrome | mha_step | 0.9725 | 1.0008 | 1.0188 |
| chrome | transformer_deepcopy | 0.9980 | 1.0066 | 1.0007 |
| chrome | transformer_pickle | 0.9987 | 1.0227 | 1.0156 |
| chrome | vjp_linear | 0.9907 | 1.0002 | 1.0040 |
| chrome | vjp_scalar | 1.0254 | 1.0236 | 0.9857 |

Five older-API flows received an additional 18 fresh processes with batches
of at least 100 ms. CPU controls use hash seed 0; WASM uses its runtime default.
These longer controls retain all source-load orders and balanced A/A/B samples.

| Runtime | Flow | A/A | B/A | B/AA |
| --- | --- | --- | --- | --- |
| cpu | groupnorm_step | 0.9969 | 0.9855 | 0.9944 |
| cpu | linear_deepcopy | 0.9886 | 1.0035 | 1.0311 |
| cpu | transformer_pickle | 1.0090 | 1.0151 | 1.0014 |
| cpu | vjp_linear | 1.0088 | 0.9944 | 0.9920 |
| cpu | vjp_scalar | 1.0057 | 1.0165 | 1.0005 |
| node | groupnorm_step | 1.0016 | 0.9969 | 1.0014 |
| node | linear_deepcopy | 0.9996 | 1.0067 | 1.0014 |
| node | transformer_pickle | 1.0174 | 1.0166 | 1.0104 |
| node | vjp_linear | 1.0086 | 1.0125 | 1.0063 |
| node | vjp_scalar | 1.0018 | 0.9892 | 0.9919 |
| chrome | groupnorm_step | 0.9864 | 0.9905 | 1.0003 |
| chrome | linear_deepcopy | 1.0002 | 0.9870 | 0.9889 |
| chrome | transformer_pickle | 1.0024 | 0.9926 | 0.9908 |
| chrome | vjp_linear | 0.9971 | 0.9976 | 0.9961 |
| chrome | vjp_scalar | 1.0039 | 0.9860 | 0.9890 |

The longer B/A medians range from approximately 0.986 to 1.017. The initially
larger VJP differences did not persist at their initial magnitude. Paired
same-source variance remains: CPU copy has B/AA 1.0311 despite B/A 1.0035.
Every original sample and both denominators are retained. No blanket
no-regression or universal speedup claim is made.

A separate six-process CPU counterexample restored only the legacy private
flag name. It showed no stable benefit and was rejected; the already qualified
generic AD source remained unchanged.

CPU `tracemalloc` measurements use three dense float64 sin-Jacobian sizes,
six balanced source orders and an identical-source duplicate. Analytic diagonal
values and caller state are checked on every result. The candidate uses about
36%–46% of the naive prototype's tracked peak allocation in these cases.

| Input shape | Final dense Jacobian bytes | Candidate / naive peak | A/A peak |
| --- | --- | --- | --- |
| [8, 8] | 32768 | 0.3587 | 1.0000 |
| [16, 16] | 524288 | 0.4283 | 1.0000 |
| [24, 24] | 2654208 | 0.4631 | 1.0000 |

These are CPU tracked allocations, not RSS, WASM heap or GPU memory. No latency
child overlapped our package or full application gates. Numerical kernels and
the existing cotangent lifetime improvements remain in use.

## Reproducible evidence

Machine report: `packages/stridetorch/benchmarks/results/iteration-70.json`.
Raw archive: `iteration-70-harness.tar.gz` (3,274,431 bytes), SHA256
`820ca3aa1fe482d7b166f4c7253158968d18cda9e07cd9335faac220d48e569b`. All 1544 logical payloads were verified after
writing; identical-content hardlinks preserve permissions and ownership. File
timestamps are not comparison inputs. The archive preserves the rejected flag
experiment, initial short samples, longer controls, source snapshots, model
fixtures and installed-package evidence. Scripts retain original absolute paths
and require dependency setup and path adjustment for replay.

The selected final source is `versioned-release/snapshot`; earlier prototype
sources retain their original 0.69 version literal and are not relabeled as
final-version runs. No CI workflow, tag, package publication or deployment was
initiated by these local checks. The independent library name remains
StrideTorch pending a naming decision.

## Main adoption

Version 0.70.0 was adopted on main at `9f2715fc21fa408230a3c49c047dd14af1c69c13`. The guarded ordinary push passed with no active release chain. The remote ref and 2300 tested Git blobs were verified, together with generated Worker derivatives and all 14,083 prior immutable asset tree entries. The adopted SHA had zero Actions runs at verification. No workflow dispatch, tag, package publication or deployment was initiated. [Adoption receipt](../../packages/stridetorch/benchmarks/results/iteration-70-main-adoption.json).
