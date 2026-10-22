# Cumulative-product development

This is a historical development snapshot. The final version bump, full SDK,
application/package/browser qualification and restoration checks are recorded
in [TorchWasm 0.91 qualification](torch-runtime-0.91-qualification.md).

The baseline TorchWasm 0.90 `cumprod` delegates its forward scan to NumPy and performs a
Python loop over the scan dimension in reverse. Before replacing it, a read-only
audit ran the same 144 cases against independently executed PyTorch 2.6.0 and
2.14.1. It retained complete values, error observations and three independent
polynomial-Jacobian checks with zeros.

Both environments expose five forward-value differences: low-precision long
scans and complex infinities. Twelve cases incorrectly permit an integral output
while recording a floating-input graph. Nine direct JVP, functional JVP and HVP
cases are supported by native PyTorch but remain unsupported by TorchWasm.
The audit also retains nine older-native errors in zero-containing,
dtype-widening reverse cases; PyTorch 2.14.1 accepts those cases. Those native
version differences should not be counted as nine additional library defects.

## Forward candidate

The standalone [candidate](../../packages/torchwasm/benchmarks/fixtures/cumprod_scan_candidate.py)
converts each input block to the requested output dtype before using the wider
CPU-style accumulation type. It preserves the accumulator across block
boundaries and performs the initial multiply by one, which is observable for
complex infinities. Its 256 KiB workspace planning budget is not a bound on
total memory: the output, carry, input conversion and wide outer layouts also
occupy memory. The prototype still has small-call allocation/dispatch costs to
measure and improve.

The final 553-case corpus uses identical dyadic inputs on both native
environments. All candidate forward values match native PyTorch in each
environment. Native input and output hashes also agree between the two Torch
versions. Node/Pyodide and a real Chrome Worker each replay all 553 cases and
match the full native output hashes. Python is 3.13.2, NumPy 2.2.5 and Pyodide
0.29.4 in both WASM runs. The Worker is terminated after its result is captured.

Coverage includes float16/32/64, complex64/128 values with zero imaginary
components, requested dtype conversions, scalar/empty shapes, C/F/strided/
read-only/broadcast inputs, long scans crossing block boundaries, zeros,
infinities and NaNs. Comparisons canonicalize each NaN component's sign/payload;
finite values and finite zero signs remain exact. Six earlier special-value
cases differ only under the stricter NaN-sign comparison, and their original
observations are retained. Arbitrary complex-imaginary inputs, integral kernels
and very wide outer layouts still need additional coverage.

The [development record](../../packages/torchwasm/benchmarks/results/cumprod-development.json)
and archive retain the baseline audit, candidate, all harness sources, native
references and WASM results. Archive SHA-256:
`48ba0e00756ca0e836a5017e240888a86d953d52980ef24a1e990cded9c51de6`.
Integrity and unchanged-runtime checks pass with the public evidence verifier.
This is a standalone forward-kernel result. It does not qualify replacement of
the complete `torch.cumprod` API, derivatives, packaging or application runtime.
That standalone-candidate snapshot retains the qualified 0.90 runtime and SDK
identities. The later API development below changes the local runtime source;
the historical standalone record remains unchanged.

## Retained harness failures and next work

The first audit queried version metadata on the `torch` alias rather than its
owning package and was repaired in a fresh capture. A first cross-environment
fixture used floating random arithmetic: 75 input hashes differed across NumPy
versions, so its plan assertion stopped. Its attempted WASM runners had no plan
and exited before calculations. The repaired corpus uses exact dyadic
arithmetic and checks both native input/output identities before WASM replay.
Node's first loader received a file URL where Pyodide expects a filesystem path;
the Node-only correction retains the same kernel and fixtures. The independent
Chrome HTTP Worker needed no loader repair. All original logs are archived.

Reverse work must account for each prefix's product excluding the differentiated
factor, including single and multiple zeros. Division by the input alone is
insufficient. Direct JVP, reverse-derived functional JVP and composed HVP need
independent polynomial checks, native dtype-rounding comparisons and mutation
guards. Scalar result ownership, axis/dtype validation and complex-autograd
rejection also need API tests. Complete inference/gradient timings on CPU and
WASM, then SDK/application/package verification, are required before adoption.
No performance improvement or WebGPU support is claimed by this prototype.

The implementation direction is checked against the official
[cumprod dtype contract](https://docs.pytorch.org/docs/stable/generated/torch.cumprod.html),
[CPU scan kernel](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp)
and [autograd declarations](https://github.com/pytorch/pytorch/blob/v2.6.0/tools/autograd/derivatives.yaml).
No CI, tag, package publication or deployment was initiated for this work.

## API and derivative development

Commit `d7fe79acf9ffe375b19a11ad449c58a7a7577281` integrates cumulative
products into the checked reduction layer. It adds opmath accumulation,
owned scalar/empty results, axis/dtype validation, a linear masked reverse
scan, direct JVP, differentiable functional JVP and composed HVP. Forward
transforms retain the existing real float32/64 boundary; complex autograd,
output buffers and general backward `create_graph` remain unsupported.
The differentiable zero-input path enumerates omitted products and is
quadratic in scan length. Ordinary reverse does not use that path.

Direct and reverse-derived JVP preserve their different dtype-rounding order.
Scalar, empty and singleton scans have a constant functional Jacobian. An
internal prefix reduction preserves nonfinite product adjoints without
changing the public `prod` primitive's existing support boundary. Saved input
and output version guards remain active. A short-scan path avoids block
geometry; larger scans retain their carried accumulator.

The final 492-case regression module checks an independently enumerated
polynomial Jacobian and Hessian, composed square/sine/norm losses, gradients
of returned JVPs, zero factors, layouts, dtypes, singleton strictness, saved
versions and host NumPy error-policy independence. It passes on both native
environments, Node/Pyodide and a Chrome Worker, including isolated exported
namespace/JVP/HVP smokes. All 243 affected application/conversion tests pass.

The preceding development snapshot passed the full native SDK: 13,820 on
modern NumPy, and 13,817 plus three existing skips on the minimum environment.
Node and Chrome each passed the full 13,804 selected SDK cases. The final
followup changes only four cumprod helpers and its regression module; all
other helpers, public source files and 101 other SDK modules retain their
checked identities. The final 492-case module is rerun on all four hosts.
This is explicitly not a full 13,845/13,829-case rerun on the final source.

An additional 2,298-case public-API corpus covers nonfinite/zero reverse rules
and complex-imaginary forward inputs. All values and non-NaN signs match the
newer native reference exactly on CPU, Node and Chrome. The minimum native
version rejects 296 dtype-widening zero cases; those version differences are
retained separately. Forty-eight transform/returned-gradient controls match
the modern native results within recorded tolerances; two older-native dtype
errors remain observations. Both native versions agree on all 31 final
boundary-control acceptance decisions and returned values.

## Complete-flow latency observations

The comparisons use 36 finite all-one fixtures: three lengths, float32/64,
zero/one/two zero factors, and inference or square/sum/backward. Each environment
has four ABBA observations per case. Native environments run sequentially;
Node uses independent baseline/candidate Pyodide runtimes. Fixtures isolate
latency from the baseline's incorrect low-precision accumulation. These are
scoped latency measurements, without an allocation, RSS or WASM heap claim.

The table gives median candidate/baseline latency ratios across the six
dtype/zero cases in each row. Values below one are faster; the complete
case timings and ranges remain in the machine record.

| Length | Flow | Modern NumPy | Minimum NumPy | Node/WASM |
| --- | --- | ---: | ---: | ---: |
| 16 | Inference | 1.300 | 1.555 | 1.281 |
| 16 | Reverse | 0.995 | 1.055 | 0.874 |
| 4,096 | Inference | 1.200 | 1.538 | 2.049 |
| 4,096 | Reverse | 0.019 | 0.022 | 0.015 |
| 65,536 | Inference | 0.636 | 1.076 | 1.130 |
| 65,536 | Reverse | 0.008 | 0.008 | 0.005 |

The first block-only candidate made small inference about 2.6 times slower
on modern NumPy. The final short-scan path reduces that cost substantially,
but small and medium inference costs remain. Medium WASM inference is the
next performance target. Wider accumulation and API validation must remain
correct while reducing unnecessary allocation and block setup.

The [API development record](../../packages/torchwasm/benchmarks/results/cumprod-api-development.json)
binds the exact source, reused full-suite snapshot, final affected checks,
native references and all timings. Its 1,482,733-byte archive recovers 1,581
logical files from 321 unique blobs; SHA-256:
`04953c3e1538840b954b27e8fbbdbff84e8005b958df3ac18781c8ad4c39d4c3`.
Integrity and current-source checks pass with the public verifier.

Original failures remain archived: an incorrect offline HVP expected constant
(30 instead of 32), singleton strict-HVP and nonfinite-zero HVP gaps found by
the added boundary controls, and two WASM captures that failed to save results
under ENOSPC. The latter are not counted as successful checks. Verified hardlink
deduplication of 147 byte/mode-identical frozen archive copies preserved every
path and freed about 452 MB; fresh WASM captures then passed.

This is integrated development source, with package metadata still at 0.90.
Version selection, package/install qualification, full native application,
all-89-module WASM runs, fresh immutable exports, frontend/build/Worker
integration and current-main adoption remain pending. The prior failed
OpenSSL delivery grant stays consumed. No new delivery attempt, CI dispatch,
tag, package publication or deployment was initiated by this development.

The subsequent [scan-execution followup](torch-cumprod-execution.md) reduces
medium-scan setup/copy costs, adds long/layout and cross-threshold HVP controls,
and records both latency improvements and remaining large CPU inference costs.
Its comparison baseline is this API development, rather than the earlier
NumPy-delegating implementation used by the table above.
