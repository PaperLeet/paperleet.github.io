# Implementation notes

Normalize each completed domain's importance by its own squared displacement plus xi before summing domains. Squaring the sum of displacements introduces cross-domain cancellation. Averaging histories weakens the constraint as history grows and violates Eq. (3).

The regularizer has derivative `2*strength*factor*(prompt-anchor)`. EWC's conventional one-half is absent. The historical factor and anchor are fixed records. In prompt addition, shared canvas gradients sum over images; image gradients remain separate. Do not clamp a prompted image to a display range.

The synthetic objective is `0.5*||p-[1,-1]||² + strength*4*p0²`, so stationary coordinates are `p0=1/(1+8*strength)` and `p1=-1`. This independent solution checks the learner's actual SGD path, rather than comparing two copies of the reference formula. At strength 1, the final data loss is approximately 0.3950617284; without regularization it is below 1e-8.

The source has inconsistent prompt symbols around Eqs. (4)–(6). We follow the prose's role assignment: the domain-agnostic prompt is constrained. This module does not resolve full training-loop gradient routing or implement both original consistency objectives.

Sources: [Method, Eqs. (1), (3), (4)](https://arxiv.org/abs/2212.04145), [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/25922).
