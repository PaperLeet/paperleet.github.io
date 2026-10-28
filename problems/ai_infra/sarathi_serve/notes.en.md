# Sarathi-Serve Decode Priority and Chunked Prefill — Solution notes

## Derivation and implementation

Subtract one token for every decode first, then process prefills in order. An unfinished long prompt retains the queue head. New prompts cannot displace ongoing decodes or force an unbounded prefill.

$$
B_{prefill}=B-|D|,\qquad c_i=\min(r_i,C,B_{remaining}),\qquad |D|+\sum_i c_i\le B.
$$

## Boundary and gradient ownership

decodes is a list of distinct nonnegative IDs; prefills contains positive remaining token counts in FIFO order. budget>=len(decodes), chunk_size>=1. Every decode runs once. Give the head prefill min(remaining,chunk_size,budget_left); if unfinished, stop admitting later prefills. At most one chunk per request per iteration. Return decodes, prefills as (index,count), and remaining with zeros for completed items.

## Independent checks

T0 checks the callable contract and invalid boundaries. T1 uses hand calculations or an independent scalar formulation. T2 checks invariances, state conservation, gradients when meaningful, or atomic state changes. T3 invokes the supplied implementation in the deterministic experiment; it does not compare two imports of the same reference.

## Behavioral experiment

Keep two decodes active with prefill lengths 13, 3 and 9, budget eight and chunk cap four. Consume the queue over multiple iterations and check uninterrupted decodes and conserved prefill work.

One hybrid-batch budget allocation and repeated replay; fixed token budgets and maximum chunks replace the paper performance model.

### Reference calibration

Deterministic CPU fixture; Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| prefill_tokens | 25 |
| decode_stalls | 0 |
| maximum_batch_tokens | 8 |
| maximum_chunk | 4 |

## Common mistakes

- m1_no_decode_reservation: Prefill consumes capacity already promised to active decodes.
- m2_unchunked_prefill: Removing the chunk cap violates the bounded prefill work contract.

[Sarathi-Serve](https://arxiv.org/abs/2403.02310) · Section 4.1–4.2; Algorithm 3, stall-free batching.
