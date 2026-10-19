# Iteration 89: second Chrome observation and isolated vector norm prototype

The continuing efficiency/API goal remains active. The running 0.78 sources
and Git execution identity remain unchanged. No main adoption, CI dispatch,
tag, publication or deployment was initiated. BrowserTorch remains a naming
proposal; the production distribution is still StrideTorch.

## Completed Chrome evidence

An independent verifier checks the first two completed 124-flow Chrome results
against their flushed raw records and successful process receipts. It checks
source identities, equal iteration counts, each label's position counts, every
round/batch/process ratio, and the minimum timing chunks. It does not alter the
running benchmark or reinterpret its original schedule as a Williams design.

| Chrome process | Minimum chunk | Maximum A/A drift | Controls above 5% |
| --- | ---: | ---: | ---: |
| 0 | 39.9 ms | 4.03% | 0 |
| 1 | 38.4 ms | 3.12% | 0 |

| Batched Linear forward | Process 0 candidate / baseline | Process 1 candidate / baseline |
| --- | ---: | ---: |
| float32 | 0.99399 | 0.99676 |
| float64 | 0.99601 | 0.98671 |

Neither process has a flow above the automatic 10% slowdown alert. These are
partial Chrome observations: the third process is live. The repeatable CPU
float32 batched Linear slowdown of roughly 6.75%, one CPU control excess and
five Node control excesses remain unresolved. No inference about the cause or
full performance approval follows from the browser observations.

## API development isolated from the measurements

`torch.linalg.vector_norm` is missing from the production registry. A separate
self-contained prototype adds direct norm reductions and real analytical
reverse/forward rules. It covers finite positive/negative orders, zero order,
positive/negative infinity, selected dimension sets, scalar/empty geometry,
precision widening, complex inference and the existing explicit AD boundaries.
This is implementation scope, not a validated compatibility claim.

The prototype, registry overlay and independent tests live outside the checkout
under the task's `torch-iteration-89` directory. Its 64 resource files have no
symlinks to production. The existing 63 production resources retain their
original hashes. Test cases cover known geometric answers, finite-difference
derivatives, Euclidean second derivatives, tied extrema, zeros, empty reductions,
storage guards and final backward release. Only syntax checks have run so far.

A frozen CPU validation queue is waiting behind the known WASM factor supervisor,
which itself waits behind CPU factor timing and the original 124-flow campaign.
It will execute the same 780-case probe against native Torch 2.6, native Torch
2.14 and the prototype, comparing forward values, VJPs, direct JVPs (including
their reverse graphs), functional JVPs, HVPs and inference dtypes. It then runs
the independent tests. The comparison records all discrepancies for repair;
no expected-result suppression or compatibility claim is made before execution.
WASM qualification of this prototype is still pending.

The PyTorch 2.6 norm reverse/JVP definitions used as reference are preserved
with source URLs and hashes in the task evidence. See the
[public vector norm contract](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.vector_norm.html)
and [native derivative source](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/autograd/FunctionsManual.cpp).

## Remaining work

Finish the original campaign and inspect the balanced factor results before
choosing a production Linear change. Repair and qualify the new norm across
native and actual WASM runtimes before integrating it. Any changed production
source needs renewed library/package/application evidence and a fresh guarded
main adoption; the separate prototype checks cannot qualify that adoption.
CPU/WASM efficiency, core API coverage and backend work continue afterward.
