# Iteration 96: completed factor campaign and zero-rule diagnosis

All nine iteration-92 factor timing processes are terminal, and iteration-94
qualification finished on CPU, Node/Pyodide and real Chrome. The measurements
are useful but do not qualify either prototype for production adoption. The
continuing goal remains active; main, CI, tags and release are unchanged.

## Complete factor observations

An independent verifier checks all twelve process receipts, immutable inputs,
current canonical source hashes, durable raw rows, equal iteration counts,
Williams positions/ordered pairs, calibration and every aggregated ratio. It
covers all 162 flow records across three processes per runtime. No new factor
A/A comparison exceeds 5%; the old campaign's six anomalies remain preserved.

| Runtime | Smallest measured chunk across three processes | Largest A/A drift |
| --- | ---: | ---: |
| CPU | 76.14 ms | 4.68% |
| Node/Pyodide | 83.62 ms | 1.49% |
| Chrome/Pyodide | 89.80 ms | 1.12% |

The paired in-place/out-of-place fast-path ratios for batched forward are
0.99164/0.99989 on CPU, 1.00005/0.99956 on Node and 1.00450/1.00000 on Chrome
for float32/float64 respectively. These forward observations do not support a
blanket inference-path change. Backward must be assessed separately: paired
in-place/out-of-place fast-path ratios for batched backward are 1.02583/1.01369
on CPU, 1.00424/0.99880 on Node and 1.00097/0.99761 on Chrome. CPU float32
backward therefore supplies evidence in favor of allocation rather than reuse,
with a peak-memory tradeoff that still needs evaluation. A forward-only table
cannot select the policy for all modes. These factors still hold every
non-Linear file at 0.78,
so they cannot resolve the original full-0.77-to-full-0.78 CPU regression.
Iteration 95 now measures that distinct complete-version comparison.

## Norm qualification completed with findings

The isolated declared-zero prototype passes all 11,982 existing SDK tests and
95 targeted tests on CPU. Node and real Chrome each pass the same combined
12,077 tests, with no skips. Its 780 primary norm cases match all 3,660 checked
stages per native reference version on all three runtimes.

Broader probes retain these discrepancies against each of native Torch 2.6
and 2.14, identically across CPU and WASM:

| Probe | Cases | Matches | Discrepancies |
| --- | ---: | ---: | ---: |
| Norm compositions, checked stages | 1,600 | 1,592 | 8 |
| Logical-zero chains | 288 | 264 | 24 |

The earlier composition probe had twenty mismatched stages per native version;
this declared-zero version reduces that to eight. The remaining findings are
HVP behavior at zero for squared inputs and normalization, plus a negative-order
norm of exponentials at nonfinite input. They remain unsupported work, not
discarded undefined-point tests.

The expanded zero-chain probe finds six upstream operations that currently
materialize the constant cotangent too early: reshape, scalar multiplication
by zero or two, addition, scalar division and sum. Each mismatch occurs for
both dtypes and for absent/identity hooks at nonfinite inputs. Replacement
hooks, undefined upstream gradients and numerical-zero rules remain checked.
Negation, mean and matrix kernels must not be declared zero-preserving merely
because a mathematical simplification appears possible; their native observed
boundaries differ.

The independent terminal observation checks all 246 frozen inputs, twelve
successful receipts/log hashes, counts, native probe identity and matching
runtime results. It binds the terminal outputs by fresh hashes. No claim of
complete library, package, application or performance qualification follows.

## Next isolated rule revision

The successor changes only `_tensor.py` and `_autograd.py` relative to the
iteration-94 prototype. It declares addition, scalar multiplication/division,
reshape and sum as zero-preserving. The view helper accepts an explicit flag;
its default and other view rules remain unchanged. Tensor-valued multiplicative
operands, subtraction, negation, mean and matrix kernels stay undeclared pending
their own evidence. Ordinary numerical zeros and `None` are not reclassified.

This revision has passed syntax/source checks only. Its diagnostic expands
from sixteen to thirty-two chains, adding views, broadcasting-related geometry,
tensor operands, reflected arithmetic, nonfinite scalar coefficients and clones:
576 cases across two dtypes, three input families and three hook modes. Fresh
native controls must also reproduce the original 288 rows exactly, so expansion
cannot silently change the earlier reference. All primary/composition probes,
95 targeted tests and 11,982 existing SDK tests are retained for the CPU run.

The frozen 79-file successor queue waits for the known iteration-95 timing
supervisor to terminate before doing numerical work. It also freezes 144
external inputs, including canonical resources/tests and prior native chain
results. No successor numerical results exist at this checkpoint. Its eight
known HVP discrepancies are still open; broader WASM/package/application checks
will follow source repair and selection.

Evidence is in `torch-iteration-95/complete-factor-observations.json` and
`torch-iteration-96/qualification-94-independent-observation.json`, together
with the successor source, probe, frozen plan and waiting process record.
