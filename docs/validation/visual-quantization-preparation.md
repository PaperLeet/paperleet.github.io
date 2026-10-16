# Visual quantization preparation

The interview handbook adds a three-step route: Rotation Trick, residual
multi-scale tokenization, then next-scale teacher inputs and attention masks.
These are three maintained exercises, with nine new bilingual explain,
implement and verify prompts. The handbook now has 60 guides / 180 prompts and
13 routes. Existing guides and routes remain intact. Company filters organize
the reading context; this change supplies no company-frequency claims or
synthetic interview reports.

Rotation Trick is a quantizer-training surrogate-gradient method. It is not
presented as a component of the original VAR training recipe. The guide freezes
the selected code, rotation directions and norm ratio, while leaving the encoder
input differentiable. For e=(2,0), q=(0,3) and upstream gradient (1,0), the
reference produces encoder gradient (0,-1.5) and no codebook gradient through
this surrogate. The verify prompt checks the analytical surrogate Jacobian,
rather than finite differences of discrete nearest-neighbor selection. Zero
and opposite-vector inputs follow the exercise's explicit rejection contract.

The tokenizer guide follows this exercise's shared squared-Euclidean codebook,
area downsampling, full-resolution cumulative residual reconstruction and fixed
channel projection. The fixed C-by-C projection is a teaching simplification of
VAR's learned spatial residual transformation. Snapshots must be independent and
detached; decoding sums all scale contributions. It does not promise monotonically
decreasing intermediate MSE with an arbitrary frozen codebook: for a 2-by-2 field
of ones, codebook [-1,+1], scales [1,2] and identity projection, coarse codes
reconstruct ones, then a zero-residual tie selects -1 and the final reconstruction
is zero. The reference gives MSE [0,1].

The next-scale guide distinguishes target labels from teacher inputs. Each
scale's inputs use only the sum of all earlier scales, accumulated at full
resolution before area downsampling. Same-scale positions may attend to each
other because their inputs do not contain same-scale targets. For scales [1,2],
the 5-by-5 visibility mask includes a separate start position, while the returned
teacher sequence has four positions. Changing the last target has no effect on
those inputs; changing the earlier coarse code does. A single scale returns an
empty teacher sequence. The exercise uses True for visible mask entries.

Sources reviewed against the maintained exercise READMEs, solutions and tests:

- [Rotation Trick paper, section 4](https://arxiv.org/html/2410.06424v1).
- [Pinned author implementation](https://github.com/cfifty/rotation_trick/blob/99104924a2a61d393be30567ee8a47b350050622/src/models/vq_vae.py).
- [VAR paper, section 3.2 and Algorithms 1–2](https://arxiv.org/html/2404.02905v2).
- [Pinned VAR quantizer](https://github.com/FoundationVision/VAR/blob/78b95394fc5896192e3a003e4b295f8ea743c48f/models/quant.py)
  and [scale mask](https://github.com/FoundationVision/VAR/blob/78b95394fc5896192e3a003e4b295f8ea743c48f/models/var.py).

Local native float64 checks exercise the maintained references on the hand
examples above, independent cumulative snapshots, decoding, coarse-prefix
dependence and the single-scale boundary. These small examples do not reproduce
full model training or paper performance.

The mounted React handbook check initially failed with Invalid interview
filters: the new catalog route was absent from the frontend routing allowlist.
The routing allowlist now accepts the new route without importing deferred
catalog metadata into startup. The existing strict parser still rejects malformed
routes. The new interaction check covers all nine bilingual prompts, native
answer disclosures, deferred loading, three curriculum-ordered practice links,
company/role/stage/saved/search/page filters and exact practice/sign-in returns.
Server continuation validation reads its route IDs from the pinned catalog;
the paired private change exercises both OAuth providers on the new hub and
nested practice destinations. No browser visual review is claimed.
