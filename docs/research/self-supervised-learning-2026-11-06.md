# Self-supervised learning collection

Verified on 2026-11-06. Practice order: CPC / InfoNCE → SimCLR → MoCo →
SimSiam → Barlow Twins → VICReg → I-JEPA. The chronological roadmap is a
separate view: MoCo's first preprint predates SimCLR. Existing published
module and paper numbers remain unchanged.

## Source and exercise scope

| Paper | Primary source | Numerical core |
| --- | --- | --- |
| CPC / InfoNCE | [CPC §2.3, Eq. 4](https://arxiv.org/abs/1807.03748) | Stable positive classification from supplied logits, arbitrary target positions; the denominator includes the positive. No implicit normalization or temperature. |
| SimCLR | [ICML paper, Eq. 1 / Algorithm 1](https://proceedings.mlr.press/v119/chen20j.html) | Reuse module 84: all 2B anchors, self excluded, same-view negatives retained, both directions averaged. |
| MoCo | [CVPR paper, Eqs. 1–2 / Algorithm 1](https://arxiv.org/abs/1911.05722) | Reuse module 85: query-only contrastive gradient, momentum encoder and circular dictionary. |
| SimSiam | [CVPR paper, §3 / Algorithm 1](https://arxiv.org/abs/2011.10566), [official implementation](https://github.com/facebookresearch/simsiam) | Symmetric cross-view negative cosine; only targets stop gradients. |
| Barlow Twins | [ICML paper](https://proceedings.mlr.press/v139/zbontar21a.html), [official implementation](https://github.com/facebookresearch/barlowtwins/blob/main/main.py) | Batchwise feature normalization and cross-correlation. Fix population variance plus epsilon to match official BatchNorm; sum both diagonal and full off-diagonal penalties. |
| VICReg | [ICLR paper](https://arxiv.org/abs/2105.04906) | Reuse variance, invariance and covariance exercise. |
| I-JEPA | [CVPR paper](https://arxiv.org/abs/2301.08243) | Reuse masked target representation prediction and momentum update exercise. |

## Curation and roadmap evidence

The original papers and their released implementations establish numerical
contracts. Independent curriculum evidence is [Stanford CS231n's 2022
self-supervised learning lecture](https://cs231n.stanford.edu/slides/2022/lecture_14_jiajun.pdf).
The research comparison [Exploring the Equivalence of Siamese Self-Supervised
Learning via A Unified Gradient Framework](https://arxiv.org/abs/2112.05141)
groups MoCo/SimCLR, SimSiam and Barlow Twins/VICReg into the contrastive,
asymmetric and feature-decorrelation families. [Meta's I-JEPA research
announcement](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
supplies an additional author-maintained account of predictive representation
learning. Inclusion confidence is high for these canonical mechanism examples;
this is a compact learning sequence, not an exhaustive survey or a popularity
ranking.

The roadmap uses first arXiv announcement years and the actual publication venue.
Its four arrows represent explicitly labelled mechanism relationships:
CPC→MoCo and CPC→SimCLR reuse contrastive candidate classification;
SimCLR→SimSiam compares two-view Siamese objectives with changed gradient
ownership; Barlow Twins→VICReg connects redundancy reduction with separately
controlled variance and covariance. The SimCLR→SimSiam label is a conceptual
comparison, not a claim that SimCLR alone originated SimSiam. I-JEPA is shown
without an invented direct inheritance edge.

Five added cover thumbnails are first-page renders of the linked arXiv PDFs,
168 pixels wide, WebP quality 72, with metadata stripped. Existing thumbnails
are reused for the other two papers.

Each new package supplies bilingual statements and notes, independent numerical
or gradient oracles, deliberate faulty implementations, and a bounded synthetic
CPU experiment. These experiments demonstrate the exercised mechanism; they
do not claim full image pretraining or paper benchmark reproduction.
