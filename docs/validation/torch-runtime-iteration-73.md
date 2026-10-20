# Torch runtime iteration 73: functional JVP preprocessing

Version 0.73 preserves the functional-JVP API and reduces temporary preprocessing
objects. Validation helpers and real dtype descriptors are shared across calls;
single-Tensor validation retains all checks without creating a generator, and
non-strict calls do not allocate strict-mode tracking sets. Exact base-Tensor
leaves receive their final gradient metadata at construction. Tensor subclasses
retain their `detach`/`requires_grad_` dispatch.

An early prototype bypassed three observed subclass `detach` calls. Its source,
original passing narrower evidence and failing dispatch probe are preserved.
The corrected candidate adds 128 regression cases spanning Tensor/returned-
subclass forms, inputs/vectors, dtype, original gradient flags, strictness and
graph mode. These check dispatch, values, reverse queries and caller buffers.

The existing callback count, original flags/buffers/hooks, cast behavior,
zero/unused/empty strict semantics and returned reverse graphs retain their
qualified behavior. Supported functional-JVP differentiation remains real
float32/64 with ordinary reverse queries of returned graphs. General higher-order
reverse AD, complex/nested transforms, callback hook registration and Tensor
mutation inside forward AD remain outside this bounded API. No GPU backend is
enabled by this change.

## Exact-version verification

| Gate | Result |
| --- | --- |
| Minimum installed standalone wheel | 8,851 tests; Python 3.10.19, NumPy 1.26.4, no native Torch |
| Node / Chrome Pyodide SDK | 8,838 tests and 74 suites each; Pyodide 0.29.4, NumPy 2.2.5 |
| Package ownership | Wheel/sdist/private-wheel identities, aliases, separate uninstall, checkpoints and fresh-process JVP operator pickle |
| JVP CPU/Node/Chrome inventory | 2,304 cases per runtime; 2,184 native-qualified, 88 explicit math controls, 32 explicit unsupported hook cases |
| Legacy HVP CPU/Node/Chrome inventory | 720 cases each; 676 direct native, four precision controls, 40 polynomial extensions |
| Current application CPU | 42,477 passed |
| Full browser catalog | 86 modules, 386 mutants; no failed or over-budget modules |
| Frontend | 160 check commands, Worker derivation and two build profiles |
| Previous immutable assets | 14,605 preserved byte-for-byte |
| New export assets | 174 reachable content-addressed assets |

The JVP numeric inventory is paired to the previously qualified public 0.72
rows at their original tolerances. It retains the independent normalization and
singular-determinant graph controls described in the iteration-72 report; these
are not literal native graph matches or a newly executed native oracle.
The 2,184 native-qualified entries include numeric comparisons and matched error
boundaries.
Four extra CPU mutation probes remain explicitly unsupported and separate from
the 2,304-case inventory. Source hashes, imported/executed helper paths, original
prototype versions and all failed/intermediate snapshots are retained.

The integration base is `ac992064795d131a23e62d46ac2ec930008e9bfa`. Application/frontend inputs,
release policies and generated Worker derivation were checked against their
recorded identities. Local validation does not itself prove main adoption or
deployment; remote adoption is recorded separately. No CI, tag or release was
requested by these checks.

CPU and full-catalog WASM gates executed on `2ae9bd77b06895380915bfc39b9702221fb6bea1`.
The base subsequently advanced through 10 frontend/documentation files
to `ac992064795d131a23e62d46ac2ec930008e9bfa`. Recorded runtime, Python application/test/catalog,
package/dependency and Worker inputs are unchanged. The complete updated frontend
suite and both build profiles were rerun on the latter base. Original execution
records remain in `validated-on-2ae9bd77b068`; the CPU/WASM results are
source-equivalent evidence, not claims of re-execution on the later Git SHA.


## CPU and WASM latency

Both sides call the **same public APIs**: released 0.72 is A, its same-source
control is A′, selected 0.73 is B. There is no composed JVP baseline. Ten JVP
workflows include single-Tensor/one-Tensor-tuple input forms, `nn.Parameter`
arguments, graph results, linear and normalization callbacks. Twenty additional
public workflows cover established behavior.

Each runtime has six fresh processes and every source-load permutation, with
twelve balanced common batches of at least 20 ms per workflow. Numerical,
dtype/shape, alias and caller-state checks precede timing. Ratios below one mean
shorter duration. Interpret both A/A denominators and the explicit performance
review. These data do not establish universal gains or native-Torch/GPU/import-
cold-start performance.
Ratios are medians of six per-process ratios. The microsecond columns are
independently summarized medians and need not reproduce those ratios.

| Runtime | JVP workflow | A′/A | B/A | B/A′ | A → B μs |
| --- | --- | --- | --- | --- | --- |
| cpu | functional_jvp_batchnorm | 1.0001 | 0.9936 | 0.9919 | 119.11 → 118.15 |
| cpu | functional_jvp_graph_scalar | 0.9961 | 0.9859 | 0.9851 | 50.04 → 49.32 |
| cpu | functional_jvp_graph_scalar_tensor | 0.9942 | 0.9771 | 0.9869 | 50.09 → 49.14 |
| cpu | functional_jvp_instancenorm | 0.9994 | 0.9952 | 0.9936 | 119.95 → 119.57 |
| cpu | functional_jvp_layernorm | 1.0007 | 0.9879 | 0.9841 | 125.55 → 123.62 |
| cpu | functional_jvp_linear | 1.0034 | 0.9829 | 0.9814 | 53.70 → 52.36 |
| cpu | functional_jvp_parameter_linear | 1.0000 | 0.9910 | 0.9968 | 53.81 → 53.66 |
| cpu | functional_jvp_scalar | 0.9990 | 0.9769 | 0.9782 | 48.03 → 46.92 |
| cpu | functional_jvp_scalar_tensor | 0.9996 | 0.9787 | 0.9803 | 48.05 → 47.31 |
| cpu | functional_jvp_unused | 1.0055 | 0.9812 | 0.9733 | 51.68 → 50.43 |
| node | functional_jvp_batchnorm | 0.9977 | 0.9888 | 0.9868 | 241.53 → 237.60 |
| node | functional_jvp_graph_scalar | 1.0020 | 0.9901 | 0.9862 | 107.89 → 106.64 |
| node | functional_jvp_graph_scalar_tensor | 1.0011 | 0.9761 | 0.9747 | 107.34 → 105.53 |
| node | functional_jvp_instancenorm | 1.0000 | 0.9871 | 0.9933 | 242.56 → 240.38 |
| node | functional_jvp_layernorm | 0.9968 | 0.9816 | 0.9841 | 263.62 → 259.11 |
| node | functional_jvp_linear | 0.9990 | 0.9661 | 0.9694 | 118.83 → 114.40 |
| node | functional_jvp_parameter_linear | 0.9990 | 1.0001 | 0.9998 | 117.00 → 116.91 |
| node | functional_jvp_scalar | 0.9989 | 0.9723 | 0.9789 | 104.13 → 101.35 |
| node | functional_jvp_scalar_tensor | 0.9995 | 0.9759 | 0.9708 | 103.32 → 100.80 |
| node | functional_jvp_unused | 1.0021 | 0.9755 | 0.9748 | 112.68 → 109.87 |
| chrome | functional_jvp_batchnorm | 1.0024 | 0.9879 | 0.9855 | 242.97 → 240.43 |
| chrome | functional_jvp_graph_scalar | 1.0044 | 0.9837 | 0.9793 | 106.25 → 104.98 |
| chrome | functional_jvp_graph_scalar_tensor | 1.0009 | 0.9797 | 0.9786 | 106.74 → 104.88 |
| chrome | functional_jvp_instancenorm | 0.9992 | 0.9833 | 0.9872 | 246.48 → 242.97 |
| chrome | functional_jvp_layernorm | 1.0022 | 0.9790 | 0.9767 | 265.04 → 261.33 |
| chrome | functional_jvp_linear | 1.0017 | 0.9715 | 0.9682 | 116.41 → 113.09 |
| chrome | functional_jvp_parameter_linear | 1.0017 | 0.9967 | 0.9942 | 117.48 → 117.19 |
| chrome | functional_jvp_scalar | 1.0000 | 0.9771 | 0.9792 | 103.22 → 101.07 |
| chrome | functional_jvp_scalar_tensor | 1.0067 | 0.9725 | 0.9677 | 103.03 → 100.00 |
| chrome | functional_jvp_unused | 1.0009 | 0.9743 | 0.9750 | 110.64 → 108.69 |

| Runtime | Other 20 A′/A range | B/A range | B/A′ range |
| --- | --- | --- | --- |
| cpu | 0.9959–1.0163 | 0.9931–1.0086 | 0.9910–1.0069 |
| node | 0.9942–1.0090 | 0.9934–1.0094 | 0.9938–1.0057 |
| chrome | 0.9944–1.0056 | 0.9925–1.0109 | 0.9934–1.0075 |

## CPU allocation peaks

After all latency children terminated, six fresh CPU processes exercised thirteen
API workflows with all source-load permutations, equal prewarm/capture counts
and rotated workflow/call orders. The original one-process observations are
retained separately. These `tracemalloc` peaks include
retained output graphs where requested. They are not RSS, WASM/GPU memory,
latency, isolated-cause evidence or universal improvement claims.

The additional controls show 0.8–3.7% lower tracked peaks for these JVP fixtures.
The three existing HVP fixtures show 0.24%, 0.85% and 1.25% higher peaks; this
tradeoff is retained and is not described as a memory improvement.

| Workflow | A′/A peak | B/A peak |
| --- | --- | --- |
| functional_jvp_scalar | 1.0000 | 0.9740 |
| functional_jvp_graph_scalar | 1.0000 | 0.9757 |
| functional_jvp_unused | 1.0000 | 0.9740 |
| functional_jvp_linear | 1.0000 | 0.9631 |
| functional_jvp_layernorm | 1.0000 | 0.9820 |
| functional_jvp_batchnorm | 1.0000 | 0.9920 |
| functional_jvp_instancenorm | 1.0000 | 0.9919 |
| functional_jvp_scalar_tensor | 1.0000 | 0.9741 |
| functional_jvp_graph_scalar_tensor | 1.0000 | 0.9758 |
| functional_jvp_parameter_linear | 1.0000 | 0.9840 |
| hvp_scalar | 1.0000 | 1.0024 |
| hvp_linear | 1.0000 | 1.0085 |
| hvp_unused | 1.0000 | 1.0125 |

## Reproduction evidence

Machine report: `packages/stridetorch/benchmarks/results/iteration-73.json`.
Raw harness `iteration-73-harness.tar.xz`: 3,812,424 bytes,
SHA-256 `dd8fa1800c47c3df6b4f263cf4a01fb0c3cf7e10eda5e7c19ffbdf0e09c429a2`. It contains 2,458 logical files,
501 unique mode/ownership-compatible payloads and
1,957 deduplicated hardlinks. Every logical extracted payload
was verified. Scripts retain absolute paths and require dependency/path setup;
virtual environments are not bundled.

## Main adoption

Version 0.73.0 was adopted on main at `cc2e83868783ee6b00e86bc0db6c0374cf1ff55d`. The guarded ordinary push passed with no active release chain. The remote ref and 2327 tested Git blobs were verified, together with generated Worker derivatives and all 14,605 prior immutable asset tree entries. The adopted SHA had zero Actions runs at verification. No workflow dispatch, tag, package publication or deployment was initiated. [Adoption receipt](../../packages/torchwasm/benchmarks/results/iteration-73-main-adoption.json).
