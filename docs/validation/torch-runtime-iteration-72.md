# Torch runtime iteration 72: functional JVP and returned reverse graphs

Version 0.72 adds `torch.autograd.functional.jvp(func, inputs, v=None,
create_graph=False, strict=False)` for the supported real float32/float64
operators. Tensor/flat-tuple input and output structure is preserved. The callback
runs once; processed arguments isolate repeated inputs and captured originals.
Original gradient flags, existing gradient buffers and original argument hooks
remain intact during the API's internal work.

Default results are detached. `create_graph=True` enables ordinary reverse
queries of the returned value and directional derivative. This does not enable
general `autograd.grad(create_graph=True)`, arbitrary higher-order reverse AD,
complex differentiation, nested forward transforms or unvalidated forward rules.
Registering gradient hooks inside the callback explicitly rejects because such
hooks can change reverse-based JVP values.

Strict mode tracks Jacobian dependencies of every input and output, including
zero directions. Private vector leaves cannot make a constant Jacobian appear
naturally differentiable. Differentiable captured weights remain valid. Empty
sums and empty determinants retain their different strict and graph semantics.

Normalization/matrix-factor graphs remain differentiable where requested.
Default normalization JVP avoids building the additional higher-order graphs.
Singular determinant directions use cofactors; 1x1 determinant directions keep
their symbolic zero reverse edge. Ordinary `torch.func.jvp`, production Torch 2.6
inclusive clamp boundaries and existing functional HVP behavior remain qualified
by the old SDK/model fixtures and the full application checks below.

## Reference boundaries

The JVP inventory has **2,304 cases per CPU/Node/Chrome runtime**:

- 2,184 directly native-qualified cases, including expected errors.
- 48 fused-normalization graph cases use independent complex-step derivatives,
  validated by full-coordinate native FP64 two-step finite differences.
- 40 singular determinant graph cases use an independent permutation-polynomial
  reference. Five fixture families in both rounded-FP32/lifted-FP64 and FP64 are
  additionally validated against native FP64 JVP value differences for every
  matrix/vector coordinate of both scalar outputs. Native returned reverse
  graphs can give zero gradients or SVD-tape NaNs on these fixtures; the library
  deliberately provides their mathematical cofactor/curvature derivatives.
- 32 callback hook combinations explicitly raise `BrowserUnsupported`.

Four additional CPU boundary probes confirm that `Tensor.add_` on captured
vectors before or after constructing the callback output explicitly rejects in
both graph modes. These are separate unsupported-operation checks, not part of
the 2,304-case inventory or a native-compatible feature claim.

Raw native disagreements, rejected prototypes, source identities and original
tolerances are retained. These mathematical controls are not literal native
graph matches. The adopted 0.71 HVP inventory is also replayed on this source in
CPU/Node/Chrome: 720 cases each, comprising 676 direct native cases, four explicit
precision controls and 40 finite polynomial extensions to native SVD-tape NaNs.

## Final-version verification

| Gate | Result |
| --- | --- |
| Minimum installed standalone wheel | 8,723 tests; Python 3.10.19, NumPy 1.26.4, no native Torch |
| Node / Chrome Pyodide SDK | 8,710 tests and 73 suites each; Pyodide 0.29.4, NumPy 2.2.5 |
| Installed package ownership | Wheel/sdist identities, coinstall, separate uninstall, aliases, checkpoints and fresh-process JVP operator pickle |
| Current application CPU | 42,349 passed |
| Full browser catalog | 86 modules, 386 mutants; no failed cases or browser-budget overruns |
| Frontend | 158 check commands, Worker derivation and two build profiles |
| Existing immutable export assets | 14,431 preserved byte-for-byte |
| New export assets | 174 reachable content-addressed assets |

The integration base is `77be0d1be2af79a3a7d315c80991d55014c711c8`. All application/frontend inputs
and seven canonical release-policy files were checked against their recorded
identities. Generated Worker files are verified by exact esbuild derivation,
separately from tracked source files. No CI, tag, release or deployment is part
of this local validation. Remote main adoption is recorded separately.

The full CPU/WASM and installed-package executions were completed on
`096e3ec42c82de5e202bd0691aef7948f2bc7f7f`. Main then advanced with nine
frontend/documentation files. Every numeric/runtime/test/package/export input
and Worker source retained its exact content identity. The complete updated
frontend suite, driver/policy checks, both builds and Worker derivation were
rerun on `77be0d1be2af79a3a7d315c80991d55014c711c8`. The original execution records and
their original raw archive are preserved; they have not been relabelled as new
executions.

## CPU and WASM latency

Six fresh processes per runtime exercise all six source-load permutations,
twelve balanced common batches of at least 20 ms per workflow, and a same-source
duplicate. Every child runs numerical, dtype/shape, alias and caller-state checks
before timing. A is released 0.71, A′ is its same-source control, B is selected
0.72. Ratios below one mean shorter measured duration. Interpret both denominators
and the explicit performance review; these measurements do not establish
universal gains or native-Torch/GPU/cold-start performance.

Twenty existing public API workflows use the same public APIs on both sides:

| Runtime | B/A median range | A′/A median range | B/A′ median range |
| --- | --- | --- | --- |
| cpu | 0.9955–1.0140 | 0.9948–1.0072 | 0.9941–1.0082 |
| node | 0.9921–1.0078 | 0.9924–1.0076 | 0.9938–1.0084 |
| chrome | 0.9913–1.0164 | 0.9947–1.0079 | 0.9918–1.0132 |

The seven new functional-JVP workflows compare B with explicitly isolated
compositions of released 0.71 public AD primitives. They are independently
validated for these pure callbacks only; 0.71 has no functional-JVP API.

| Runtime | Workflow | A′/A | B/A | B/A′ | A → B μs |
| --- | --- | --- | --- | --- | --- |
| cpu | functional_jvp_batchnorm | 1.0003 | 1.0767 | 1.0764 | 110.88 → 119.53 |
| cpu | functional_jvp_graph_scalar | 0.9981 | 1.0870 | 1.0886 | 46.39 → 50.51 |
| cpu | functional_jvp_instancenorm | 0.9985 | 1.0718 | 1.0762 | 112.42 → 120.68 |
| cpu | functional_jvp_layernorm | 1.0050 | 1.0636 | 1.0583 | 118.68 → 126.57 |
| cpu | functional_jvp_linear | 1.0024 | 1.1313 | 1.1336 | 47.44 → 53.66 |
| cpu | functional_jvp_scalar | 1.0040 | 1.1017 | 1.0988 | 44.11 → 48.41 |
| cpu | functional_jvp_unused | 1.0090 | 1.1123 | 1.1119 | 46.27 → 51.80 |
| node | functional_jvp_batchnorm | 1.0005 | 1.0701 | 1.0673 | 225.41 → 241.10 |
| node | functional_jvp_graph_scalar | 0.9949 | 1.0740 | 1.0773 | 101.43 → 108.78 |
| node | functional_jvp_instancenorm | 1.0047 | 1.0706 | 1.0630 | 228.31 → 244.39 |
| node | functional_jvp_layernorm | 1.0007 | 1.0651 | 1.0638 | 246.88 → 263.15 |
| node | functional_jvp_linear | 1.0022 | 1.1310 | 1.1288 | 104.07 → 117.56 |
| node | functional_jvp_scalar | 0.9977 | 1.0913 | 1.0962 | 95.76 → 104.62 |
| node | functional_jvp_unused | 1.0003 | 1.1108 | 1.1107 | 100.75 → 111.51 |
| chrome | functional_jvp_batchnorm | 0.9983 | 1.0680 | 1.0696 | 232.03 → 247.27 |
| chrome | functional_jvp_graph_scalar | 0.9961 | 1.0812 | 1.0853 | 99.90 → 108.11 |
| chrome | functional_jvp_instancenorm | 1.0063 | 1.0771 | 1.0718 | 235.74 → 253.71 |
| chrome | functional_jvp_layernorm | 0.9977 | 1.0661 | 1.0643 | 253.52 → 269.14 |
| chrome | functional_jvp_linear | 1.0009 | 1.1357 | 1.1278 | 104.00 → 118.46 |
| chrome | functional_jvp_scalar | 0.9960 | 1.0953 | 1.0962 | 96.97 → 106.05 |
| chrome | functional_jvp_unused | 1.0025 | 1.1116 | 1.1057 | 101.46 → 112.01 |

## CPU allocation peaks

One CPU process uses six balanced call orders after every latency child has
terminated. These are `tracemalloc` peaks for ten particular API workflows,
including retained result graphs when requested. They are not RSS, WASM/GPU
memory, latency, isolated-cause evidence or universal improvement claims.

| Workflow | A′/A peak | B/A peak |
| --- | --- | --- |
| functional_jvp_scalar | 1.0003 | 1.0427 |
| functional_jvp_graph_scalar | 1.0006 | 1.0403 |
| functional_jvp_unused | 1.0005 | 1.0422 |
| functional_jvp_linear | 1.0018 | 1.0631 |
| functional_jvp_layernorm | 1.0000 | 1.0284 |
| functional_jvp_batchnorm | 1.0000 | 1.0243 |
| functional_jvp_instancenorm | 1.0000 | 1.0238 |
| hvp_scalar | 0.9844 | 0.9844 |
| hvp_linear | 1.0008 | 1.0008 |
| hvp_unused | 1.0023 | 1.0023 |

Earlier optimization timings remain attached to their original prototype source
freezes and do not substitute for these selected-source measurements.

## Reproduction evidence

Machine evidence: `packages/stridetorch/benchmarks/results/iteration-72.json`.
Raw harness: `iteration-72-harness.tar.xz`, 6,099,320 bytes,
SHA-256 `79f96453f953af8203ede218887084a320ec50e57a4e75e9202b9dfce9954e81`. It retains 3,110 logical files,
559 unique mode/ownership-compatible payloads and
2,551 deduplicated hardlinks. Every extracted logical payload
was verified against its original digest. Scripts retain absolute paths and
require dependency setup/path adjustment; no virtual environments are bundled.
