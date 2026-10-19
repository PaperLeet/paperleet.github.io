# Iteration 85: repeated Linear slowdown and an isolated follow-up

The unchanged 0.78 candidate has completed all three fresh CPU processes of the
124-flow campaign. Node/WASM and Chrome/WASM measurements continue sequentially.
The [application, packaging and numerical evidence](torch-runtime-iteration-83.md)
retains its original source bindings. No production source, main branch, CI,
tag, package publication or deployment changes occur in this iteration.

## Completed CPU observations

Each process has 124 complete raw flow records. Source hashes and final JSON
match those records and the successful subprocess receipts. Minimum chunk time
across the CPU campaign is 31.40 ms. One of 372 flow/process A/A controls exceeds
5%: float32 empty-mv backward in the third process is 0.94893, or 5.11% drift.
That observation remains visible and requires review; the whole campaign is
not represented as an unconditional performance pass.

The batched Linear signal from [iteration 84](torch-runtime-iteration-84.md)
repeats across all three float32 processes:

| Biased batched Linear forward | Process B/A0 ratios | Campaign median B/A0 | Paired B/A1 median |
| --- | --- | ---: | ---: |
| float32 | 1.07025, 1.06684, 1.06749 | 1.06749 | 1.06584 |
| float64 | 1.03947, 1.03984, 0.98187 | 1.03947 | 1.04529 |

B/A1 is independently computed within each raw round and then summarized by
batch, process and campaign; it is not obtained by dividing aggregate medians.
The float32 A/A controls are approximately 1.00107, 0.99892 and 0.99985. Its
roughly 6.75% slowdown is therefore an actionable repeated signal. The float64
case also needs review against both controls, especially the third process's
less stable A0 denominator. Neither signal is dismissed because it falls below
the driver's automatic 10% slowdown alert.

## Prepared factor experiment

Four private source variants separate the two changes in `functional.linear`:

| Variant | Bias addition | Inference tape handling |
| --- | --- | --- |
| A0/A1 control | New output allocation | Legacy result/edge construction |
| B, current candidate | In-place on private output | Fast return when no gradient is active |
| C | New output allocation | Fast return |
| D | In-place on private output | Legacy result/edge construction |

A0 and A1 are independent imports of the same variant. Every variant uses the
same 0.78 resources except `_nn.py`; the control's `_nn.py` is byte-identical to
the 0.77 version. This isolates the factors without modifying the measured
candidate or changing other runtime logic.

The prepared 18-flow fixture covers both dtypes, tiny/medium/batched/singleton
inputs, forward and selected backward calls, no bias, strided input and Fortran
weight layouts. All labels share physical operand and gradient-seed buffers.
Before timing, it checks values, gradients, shapes and dtypes. Three fresh
processes use three batches of five interleaved rotations: each of the five
labels occupies every timing position once per batch. Calibration targets
100 ms for the fastest label; the hard minimum recorded chunk is 40 ms.
Completed flow samples are flushed outside the measured intervals.

Only syntax, resource identity and exact patch boundaries have been checked so
far. These variants are not selected or numerically qualified runtime changes.
The dedicated runner waits for the known current measurement supervisor to
terminate, verifies completion receipts for all nine CPU/Node/Chrome timing
jobs, and then runs its numerical validation before timing. No owned heavy
jobs overlap. Any selected implementation still requires fresh qualification
before main adoption.

## Evidence and remaining work

The task's `torch-iteration-85` directory contains `cpu-summary.json`, the
two-process historical observation, the four private variants, frozen
`linear-factor-plan.json`, the numerical/timing program and queued runner.
The original three CPU results and all raw samples stay in `torch-iteration-82`.
No earlier qualified receipt or measured source is rewritten.

Finish and review the WASM measurements, execute the factor experiment, resolve
the Linear slowdown and empty-mv control observation, and qualify the selected
version. Then reconcile current main's UI, complete its build/worker/page
checks, verify the explicit evidence archive and use the existing guarded
ordinary main-push procedure. Further core-API and backend work remains active.
