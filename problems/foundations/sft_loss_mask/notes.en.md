# Implementation notes

Drop final logits and first targets/mask entries. Sum valid losses and divide by the total valid-token count, not a mean of per-sequence means. An empty selection must retain a zero-gradient graph.

Common mistakes:

- Apply response masks at predicting positions.
- Ignore padded target positions.
- Average over all positions rather than valid targets.
- Detach the response loss.

## Masking and reduction

For response mask r and attention mask a, prediction t uses the target-position weight `(r & a)[t+1]`. The first answer token is often predicted from the last prompt position, so a predicting position being in the prompt does not itself exclude the target.

If two sequences contain 1 and 3 valid response tokens, average across all 4 tokens. Equally averaging their per-sequence means changes the objective. A fully masked sequence contributes neither numerator nor denominator.

Masks are supplied by the caller; chat-template parsing and cross-document packing boundaries belong upstream. Time O(BTV).

Center logits before logsumexp and target subtraction. Otherwise a shared offset such as 1e20 can erase log(V), even for equal logits, and corrupt gradients. The reference uses float64 intermediates and restores the input dtype.
