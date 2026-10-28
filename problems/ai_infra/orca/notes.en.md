# Orca Iteration-Level Dynamic Batching — Solution notes

## Derivation and implementation

Track remaining tokens. Admit ready pending requests before executing, decrement active requests afterward, and immediately release completed slots. Admission does not wait for the slowest member of the old batch.

$$
C_i=1+\max\{t:i\in B_t\},\qquad |B_t|\le K,\qquad \sum_t\mathbf{1}[i\in B_t]=L_i.
$$

## Boundary and gradient ownership

arrivals and lengths are equal-length integer lists; arrivals>=0, lengths>=1, capacity>=1. Break arrival ties by input index. Fill slots at each iteration boundary in (arrival,index) order without preempting active requests. Completion is the end of the final occupied slot. Return (batches, completion_times), preserving idle slots as empty tuples.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Replay six simultaneous requests of unequal length, compare completions with request-level fixed batches, and sweep capacities 1, 2 and 3 for latency and slot occupancy.

Discrete generation replay after prefill, with unit-duration iterations and one token per active request; selective-batching operators are outside this kernel.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| dynamic_mean_latency | 3.5 |
| static_mean_latency | 7 |
| tokens_executed | 11 |

## Common mistakes

- m1_batch_barrier: Holding admission until an old batch drains wastes released slots.
- m2_completion_start: Completion occurs at the end, not the start, of an iteration.

[Orca](https://www.usenix.org/conference/osdi22/presentation/yu) · Section 3, S1 and Figure 4: iteration-level scheduling; Section 4.2, Algorithm 1: FCFS admission.
