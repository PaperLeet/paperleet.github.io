# Implementation notes

Shift within each sequence before reduction. Use logsumexp or log_softmax. The final logit position must receive exactly zero gradient.

Common mistakes:

- Use current-token rather than next-token targets.
- Reverse the prediction direction.
- Count the unused final position in the denominator.
- Detach the loss.

## Alignment and gradients

1. `logits[:, :-1]` holds predicting positions; `tokens[:, 1:]` holds their targets.
2. Compute `logsumexp(logits) - correct_logit` per prediction, then average over all targets.
3. At T=1 return a graph-connected zero rather than a mean over an empty tensor.

With N=B(T−1), each contributing row has gradient `(softmax(logits) - one_hot(target))/N`. Final logits have zero gradient. Flattening before shifting incorrectly connects the end of one sequence to the start of another.

Time O(BTV); a full one-hot label tensor is unnecessary.

Center logits before logsumexp and target subtraction. Otherwise a shared offset such as 1e20 can erase log(V), even for equal logits, and corrupt gradients. The reference uses float64 intermediates and restores the input dtype.
