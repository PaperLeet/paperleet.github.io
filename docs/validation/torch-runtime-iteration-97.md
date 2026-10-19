# Iteration 97: isolate higher-derivative routes and retain parent controls

The previous turn completed real CPU/Node/Chrome factor evidence, qualified the
declared-zero prototype with retained findings and prepared the next numerical
queue. This turn prepares a separate HVP hypothesis and an unchanged parent
control while the complete-version CPU timings continue. No production source
has changed and no candidate is selected for main.

## Two explicit HVP hypotheses

The isolated successor differs from the iteration-96 source in `_vector_norm.py`
only. Its zero-norm identity guard now also protects the forward direction's
reverse route. That direction can depend on the original input when the norm's
input is a nonlinear expression, such as a square. Guarding only the saved
primal leaves that separate path exposed to zero-times-infinity arithmetic.

For orders below one, the proposed guard also requires finite source values
throughout the reduction. A negative-order norm can be zero while its input
contains both infinity and zero; checking only the resulting norm's finiteness
cannot distinguish this boundary from a finite-input zero subgradient.

These are source-derived hypotheses, not measured fixes. The old normalization
HVP discrepancy also remains open. The native reference is the preserved,
hash-bound Torch 2.6 `FunctionsManual.cpp`, particularly its norm pullback and
forward rules. The library currently implements HVP through reverse-over-forward
AD, while native functional HVP uses three reverse traversals. Matching regular
derivatives therefore does not establish identical singular-point behavior.

## Expanded observations and unchanged parent

The CPU diagnostic retains all 780 primary norm cases, 320 full composition
cases and 95 targeted tests. A new 960-case HVP matrix covers:

- float32 and float64;
- ordinary, zero, mixed-zero and three nonfinite input families;
- ten orders including negative and infinite orders;
- direct norms, nonlinear inputs/outputs and two normalization expressions.

Both native Torch versions generate fresh controls from the exact same probe
source. All 320 original native composition HVP observations must occur in this
matrix unchanged. Every new mismatch remains visible.

Iteration 98 keeps an exact 64-resource copy of the unchanged iteration-96
parent and the identical 960-case probe. After iteration 97 terminates, it runs
the parent against the same native controls, then compares candidate and parent
case by case. Its report distinguishes fixed cases, new regressions and changed
results where both remain wrong. Aggregate match counts cannot conceal a
regression through improvements elsewhere.

## Execution ordering and performance scope

The known dispatcher timing supervisor is still executing three fresh CPU
processes. Iteration 96 waits for that process to terminate before numerical
execution; iteration 97 waits for iteration 96, and the parent control waits
for iteration 97. Each queue verifies the expected PID/command and frozen inputs.
Observation delays do not restart or terminate a numerical process.

The first complete-version timing process does not reproduce the earlier
batched-forward regression. This is one process under a different, balanced
protocol and a focused twelve-flow workload. It is not a causal explanation or
permission to discard the original 124-flow findings. The remaining two fresh
processes, paired ratios and all control observations still require review.

At this checkpoint no HVP-successor numerical job has started. Its 78 files and
145 external inputs are frozen; the 67-file parent control freezes 147 external
inputs. Neither queue is a complete library/package/application qualification.
After repair and selection those broader checks and current-main guarded
integration remain necessary. GitHub CI, tags, release and deployment remain
untouched, and the continuing core/API/backend goal remains active.
