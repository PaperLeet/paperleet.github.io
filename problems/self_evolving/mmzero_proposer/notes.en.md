# Implementation notes

Average over every render attempt: failures contribute zero and remain in the denominator. With one success in two attempts, easy=1 and c=0.5 yield base reward 0.5. Invalid proposal format overrides everything with -1.

Easy-question contribution caps at 0.5, as does difficulty. Diversity uses caption/easy/hard weights 0.45/0.20/0.35. M is the supplied valid proposal population, not necessarily this tensor’s row count. The other two adjustments are supplied.

The reference behavior is defined by the starter, including deterministic tie handling and caller-supplied upstream observations. The toy policy is a controlled test, not evidence that model-generated labels are correct. Compare a held-out gold check separately from agreement with synthetic labels.

https://arxiv.org/abs/2603.09206
