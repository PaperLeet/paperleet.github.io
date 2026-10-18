# Pitfalls and checks

- 80/10/10 is conditional on selection, not a selection rate over all tokens.
- Exactly .8 selects a random token; exactly .9 keeps the original. A random replacement may equal the original.
- Targets are original tokens without a next-token shift, even when inputs change.
- Divide by selected targets, not B×L. Preserve connected zero for empty supervision.
- Use stable log-softmax and preserve the logit graph.
