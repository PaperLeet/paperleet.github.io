# Matrix RPC capability negotiation

The experimental GPU service now advertises its protocol and optional features
in the existing ready event. This prevents an updated Python wrapper from
uploading data to a cached service that cannot retain uploaded program inputs.
Discovery adds no RPC. The default CPU/WASM package remains version 0.83.0.

## Compatibility contract

`MatrixRpc.ready` negotiates `stridetorch-matrix-rpc` major 1, minor 0. The
optional features are `program-matmul-f32` and `program-retain-inputs`.
`client.protocol` is an immutable snapshot. `supports(feature)` reports only
known, advertised features of an open, ready connection; `requireFeatures`
reports `NotReady`, `Closed` or `UnsupportedFeature` as appropriate.

An unadvertised historical service keeps the original matrix operations and
computed-output programs. It does not grant retained-input capability. New
clients can use that base contract; old clients can use their original contract
with a new host. The three historical client/service/planner files are preserved
byte-for-byte in `benchmarks/webgpu_rpc/fixtures/protocol-v0` for these tests.

Retained-input Python programs check capabilities before ndarray conversion.
The default resident session checks them at construction, before taking GPU
ownership. The JS program method checks them before converting a PyProxy,
copying a typed array or consuming a request ID. These errors do not trigger an
automatic CPU retry. All public operation wrappers wait for startup. Raw
`send()` remains an explicit low-level protocol/testing interface.

An advertised incompatible major, wrong name or malformed feature declaration
rejects startup with `IncompatibleProtocol` and sends a close request. Startup
timeout also requests closure. Closing before startup completes abandons
negotiation and rejects pending public operations. Duplicate ready events cannot
upgrade a connection. Future minor versions may add features; only the known
intersection is enabled. The host's original lease remains the resource bound
if delivery fails. Shader, execution, planner and quotas are unchanged.

## Functional and resource checks

- NumPy 2.5.4 / native Torch 2.14.1 and NumPy 1.26.4 / native Torch 2.6 each
  pass 132 tests, including the original 123 and nine Python feature-preflight
  checks. The repository integration also passes 132 tests.
- The repository passes 154 portable cases: 39 negotiation, 63 program, nine
  lifecycle/protocol, 39 primitive and four cancellation/drain checks. These use
  real MessageChannels and an injected GPU, including historical client/service
  implementations. They check both directions of compatibility, unsupported
  retention before conversion, operation readiness, malformed advertisements,
  timeout disposal and close during startup.
- Real Chrome 155 / Pyodide 0.29.4 / NumPy 2.2.5 retains all 51 independent
  native numerical contexts and 31 state/ownership controls from the resident
  session qualification, with unchanged `rtol=1e-5, atol=1e-6`.
- Four real GPU hosts additionally exercise new/new, new/historical,
  historical/new and explicitly incompatible-major connections. Python runs
  actual matrix programs through both compatible mixed pairs and verifies
  their values and releases. Unsupported Python retention and resident-session
  construction consume no request IDs. The incompatible host performs no math
  and peaks at zero GPU allocation. All four hosts close with zero tracked
  allocations, handles and bytes.

The first draft's 30 negotiation cases passed but review found that primitive
upload could run before ready. The added regression test fails on that draft
with one input conversion instead of zero. The next draft exposed a second
close-during-startup boundary. Both failing records are retained; the final
candidate passes the same checks, including both fixes.

## Steady-state overhead

Six fresh Chrome processes cover sizes 256 and 512, three per size. Each tests
repeated single-node computation, repeated two-node computation and a right
operand updated by a fixed step after each forward/backward call. Two independent
previous-resident controls and the candidate use the same shader, planner,
autograd algorithm and explicit backend indirection, with identical new hosts.
The baseline retains the old JS and Python clients; the candidate uses the new
ones. Native references, owned outputs, live-cache accounting and final cleanup
are checked outside the timed batches. Per-call feature checks, readiness waits,
array copies, GPU transfers, eviction and numerical work remain inside timing.
Ordinary GC is enabled. Setup and final close are outside timing.

The frozen rule requires both aggregate A/A estimates within 5%, each individual
A/A estimate within 10%, all batches at least 30 ms and both aggregate candidate
latency ratios at most 1.05. All six contexts pass. These measurements qualify
bounded overhead; they do not establish a speedup.

| Size and flow | Unpaired candidate/baseline | Paired candidate/baseline |
| --- | ---: | ---: |
| 256, repeated single | 1.0116 | 1.0304 |
| 256, repeated chain | 1.0078 | 1.0152 |
| 256, changing right operand | 1.0172 | 1.0081 |
| 512, repeated single | 0.9937 | 0.9943 |
| 512, repeated chain | 0.9852 | 0.9885 |
| 512, changing right operand | 0.9527 | 0.9840 |

Each ratio is the median of three process estimates. The largest aggregate
paired overhead is 3.04%. One individual A/A context remains flagged: the 512
changing-operand flow in cycle 2 has estimates 0.9216 and 0.9314. Its aggregate
A/A estimates are 0.9504 and 0.9706, inside the preplanned band. No individual
process was favorably rerun or replaced. Minimum batches range from 39.2 to
55.2 ms across aggregate contexts. All raw batches and durable per-row ledgers
are retained.

## Scope and replay

The canonical 66 runtime resources, 93 SDK sources, 206 package inputs and 183
active export identities remain unchanged. Only developer GPU code, tests,
documentation and evidence change. This does not establish a stable packaged GPU
API, general task isolation, resident public Torch storage or automatic dispatch.
No CI, tag, publication or deployment is enabled.

The [machine-readable qualification](../../packages/torchwasm/benchmarks/results/protocol-negotiation.json)
links the [verified raw archive](../../packages/torchwasm/benchmarks/results/protocol-negotiation-harness.tar.gz):
357 files, 181 unique payloads, SHA-256
`46721a754cdf51f94e6795e97f7eade71bf0853880eda14b6c3aa6a577d853f4`.
It includes the failed drafts, historical fixtures, native results, real-browser
four-host gate, all six performance processes and replay dependencies. Pyodide
runtime assets and native environments remain external replay dependencies.
The runtime improvement goal remains active.

## Main adoption

Commit `460878308149351120063b9841e04bde3a1989ff` adopts the qualified
negotiation changes on main. Direct remote-tree verification covers all 15
feature paths, 66 runtime resources, 183 active exports, seven release policies
and 206 package inputs. The three targeted frontend checks also pass after
integrating the then-current main changes. The default runtime stays 0.83.0.

The feature and integration SHAs have no workflow runs. Pages still uses the
legacy `gh-pages` branch; no tag, publication or deployment was made.
The [main adoption receipt](../../packages/torchwasm/benchmarks/results/protocol-negotiation-main-adoption.json)
and [raw evidence](../../packages/torchwasm/benchmarks/results/protocol-negotiation-main-adoption-harness.tar.gz)
record the guarded push and exact remote verification.
