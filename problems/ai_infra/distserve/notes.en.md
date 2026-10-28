# DistServe Disaggregated Queues and Dual Latency SLOs — Solution notes

## Derivation and implementation

Maintain separate worker clocks. Decode waits for its own worker and the current request’s prefill plus transfer. Average TPOT includes all waiting between first and last tokens, not only raw decode service cost.

$$
F_i=\max(A_i,F_{i-1})+P_i,\quad E_i=\max(F_i+K_i,E_{i-1})+(n_i-1)D_i,\quad TTFT_i=F_i-A_i,\quad TPOT_i=\frac{E_i-F_i}{n_i-1}.
$$

## Boundary and gradient ownership

Equal-length numeric lists: nonnegative sorted arrivals, positive prefill/decode costs, nonnegative transfer costs, integer tokens>=2. Prefill completion F produces the first token; KV transfer affects later decoding. Each worker follows input FCFS order and transfer may overlap. Return (first,last). Goodput is jointly compliant requests divided by max(last)-min(arrivals), with inclusive thresholds.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Replay three simultaneous requests with prefill cost four, decode cost one and three output tokens, sweeping KV-transfer time. Report first/last-token times and jointly compliant goodput.

An FCFS discrete-event model with one prefill worker and one decode worker, including KV-transfer waits; request-serial decoding is a teaching simplification.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| first_token_last_request | 12 |
| zero_transfer_goodput | 0.21428571 |
| long_transfer_goodput | 0 |

## Common mistakes

- m1_colocated_clock: Coupling worker clocks removes prefill/decode overlap.
- m2_ignore_transfer: Dropping KV-transfer time underestimates decode latency.

[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) · Sections 2 and 3: TTFT/TPOT, disaggregated prefill and decoding, goodput.
