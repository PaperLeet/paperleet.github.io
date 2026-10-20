# Owned-flow allocation, latency and current GPU comparison

The frozen candidate changes ordinary multiply/direct-divide pullback closures
to retain input shapes and only the operand values used by an active derivative.
The measured baseline and candidate both have the 0.82 version literal. Their
only source difference is the previously validated closure change. Numerical
helpers, storage guards and higher-order rules remain unchanged. Application
adoption of a future 0.83 package is a separate gate.

## Complete-call allocation

Twenty-four flows include forward, ordinary gradients or HVP, and independent
owned NumPy result copies. Inputs are copied inside the call. The flows cover
float32/float64 scalar chains, tensor operands, mixed dtypes, both-active
multiply/divide and addition/HVP controls. An independent native Torch 2.6
oracle validates all outputs before measurement; baseline/candidate output bytes
also match. Instrumentation runs after all latency processes finish.

Each runtime first verifies that tracing observes an independently allocated
1 MiB NumPy buffer. All four environments pass that capability check. Three
uninstrumented traced-allocation samples are separate from three instrumented
Tensor-backed-owner lifetime calls per variant and flow.

For the 256-by-256 float64 chain of 32 scalar multiplications, full-call traced
peak medians are:

| Runtime | Baseline bytes | Candidate bytes |
| --- | ---: | ---: |
| Python 3.12 / NumPy 2.5.4 | 18,935,096 | 3,171,256 |
| Python 3.10 / NumPy 1.26.4 | 18,945,704 | 3,183,156 |
| Node / Pyodide / NumPy 2.2.5 | 18,905,204 | 3,158,588 |
| Chrome Worker / Pyodide / NumPy 2.2.5 | 18,905,204 | 3,158,588 |

This is about an 83% reduction in the measured flow. Tensor-backed owner
high-water bytes separately fall from 18,350,088 to 2,097,160; all observed
owners release after the calls. Traced allocations, Tensor-backed owners,
WASM heap capacity and physical RSS are different metrics. RSS is unmeasured.

Ten new lifetime regression cases produce eight expected failures in the old
baseline and all ten passes in both native candidate environments. They check
unused non-leaf storage release before backward, broadcast gradients, and
retention of operands required by both-active derivatives until graph consumption.

## Prespecified timing controls and costs

Twelve fresh processes cover two native NumPy environments, Node WASM and
Chrome WASM, with three processes each. Every process measures all 24 flows,
giving 288 process/flow observations. Two independently loaded baseline copies
and the candidate use all six implementation orders and their reverse. Normal
GC remains enabled; an explicit collection precedes each batch outside the
clock. Warmup, calibration, batch floors and acceptance are frozen before the
first measurement.

The predefined acceptance uses each flow/runtime's median of three independent
process ratios, requiring both ratio-of-medians and median-of-blocked-ratios
estimators to satisfy A/A +/-5% and candidate regression <=10%. Every batch
must satisfy the 60 ms floor, and every individual candidate ratio must stay
below 1.25. All 96 aggregate contexts pass these gates, with no discarded or
replacement measurements.

An individual Node float32/addition control in process 2 is retained as flagged:
paired A/A is 1.0707 and the unpaired candidate ratio is 1.1771; its paired
candidate ratio is 1.0197. It is not relabeled as a passing individual control.
The predefined three-process aggregate passes. The largest aggregate cost is
about 7.9% on Python 3.12 / NumPy 2.5.4 for the 128-by-128 float32 scalar chain.
This is a memory/latency tradeoff, not a universal acceleration claim. The prior
inconclusive terminal-HVP campaigns and their original gates remain unchanged.

## GPU comparison against actual 0.82 CPU/WASM

The existing bounded matrix service is reused with unchanged shaders, protocol,
default quotas, cancellation/rollback mechanisms and the original fixture
deadline. Three fresh Chrome processes compare actual 0.82 CPU autograd with
explicit GPU forward/VJP programs, including three owned CPU readbacks. The
66 loaded CPU runtime resources include the qualified WASM row-packing kernels.
Both native Torch 2.6/2.14 numerical controls pass, input fingerprints remain
unchanged, and all 24 paired CPU A/A controls pass +/-5%.

Times are medians of the three process medians; ratios show each process's
median paired GPU/CPU ratio. Shapes are `(m,k,n)`.

| Shape | Inputs | CPU/WASM ms | GPU ms | GPU / CPU range |
| --- | --- | ---: | ---: | ---: |
| 37,53,29 | Cold | 0.186 | 1.350 | 7.147–7.531 |
| 37,53,29 | Resident | 0.162 | 1.400 | 7.444–8.879 |
| 64,64,64 | Cold | 0.676 | 1.675 | 2.340–2.625 |
| 64,64,64 | Resident | 0.626 | 1.431 | 2.037–2.439 |
| 256,256,256 | Cold | 20.167 | 4.794 | 0.229–0.242 |
| 256,256,256 | Resident | 19.700 | 3.587 | 0.170–0.193 |
| 512,512,512 | Cold | 148.400 | 21.575 | 0.137–0.162 |
| 512,512,512 | Resident | 147.200 | 16.867 | 0.113–0.115 |

Large finite float32 matrices still merit a GPU path; small matrices favor CPU.
These steady-state measurements exclude browser/Pyodide/device startup and
first shader compilation. Resident setup is measured separately. The original
eight-call GPU cap leaves some batches at roughly 9–34 ms; all short batches
are retained. No general hardware/shape dispatch threshold follows, and absolute
times from different historical measurement sessions are not a version speedup
comparison. GPU VJPs remain explicit, with no automatic Torch GPU storage or
autograd backend. Production WebGPU stays disabled.

## Evidence and continuing work

[Machine-readable results](../../packages/torchwasm/benchmarks/results/operand-lifetime-owned-flow.json)
and the [verified raw archive](../../packages/torchwasm/benchmarks/results/operand-lifetime-owned-flow-harness.tar.gz)
contain all inputs, independent references, 12 complete timing processes,
allocation measurements, regression results, GPU fixtures and original flags.
The archive contains 316 files, 183 unique payloads, SHA-256
`28bb3e649dc33f0279254a7f060af904df7b399815674f06a45603da3d730404`.

The next application candidate still needs final-version SDK, package,
PaperLeet catalog/export and browser integration checks before main adoption.
The explicit GPU service additionally needs automatic Torch operator/storage
semantics, suspension-time mutation/gradient-state checks, fallback policy and
broader numerical/device coverage. No CI, tag, publication or deployment occurs
in these experiments.
