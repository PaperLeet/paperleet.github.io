# Self-supervised learning collection

Verified on 2026-11-06. The collection contains 17 papers with executable numerical
exercises. InfoNCE and SimCLR remain the entry points; the expansion covers
instance discrimination, transformation prediction, clustering, momentum targets,
self-distillation, redundancy reduction and masked prediction. Existing public
paper and module numbers are unchanged; the ten new exercises are 179–188.

## Primary sources and exercise scope

| Paper | Primary source / released implementation | Exercised numerical mechanism |
| --- | --- | --- |
| CPC / InfoNCE | [CPC §2.3, Eq. 4](https://arxiv.org/abs/1807.03748) | Positive identification from supplied logits and arbitrary positive positions. |
| Instance Discrimination | [CVPR paper](https://arxiv.org/abs/1805.01978), [official NCE criterion](https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/NCECriterion.py) | Original sampled **binary NCE** with a fixed partition-function estimate and K/N noise prior; indexed memory-bank EMA. This is distinct from candidate softmax InfoNCE. |
| SimCLR | [ICML paper](https://proceedings.mlr.press/v119/chen20j.html) | Two-view NT-Xent with all 2B anchors, self exclusion and same-view negatives. |
| MoCo | [CVPR paper](https://arxiv.org/abs/1911.05722) | Query-only contrastive gradients, a momentum encoder and a circular dictionary. |
| RotNet | [ICLR paper](https://arxiv.org/abs/1803.07728), [official code](https://github.com/gidariss/FeatureLearningRotNet) | Four exact counterclockwise rotations and a four-class prediction objective. Unrelated to the existing VQ Rotation Trick exercise. |
| DeepCluster | [ECCV paper](https://arxiv.org/abs/1807.05520), [official clustering code](https://github.com/facebookresearch/deepcluster/blob/main/clustering.py) | L2 normalization of already PCA-whitened features, Euclidean nearest-center assignments and pseudo-label cross-entropy. |
| SwAV | [NeurIPS paper](https://arxiv.org/abs/2006.09882), [official code](https://github.com/facebookresearch/swav) | Single-process finite-iteration Sinkhorn assignments and stopped-target swapped prediction. |
| BYOL | [NeurIPS paper](https://arxiv.org/abs/2006.07733), [official code](https://github.com/google-deepmind/deepmind-research/tree/master/byol) | Normalized squared regression in both directions, summed before the batch mean, plus detached target-parameter EMA. |
| SimSiam | [CVPR paper](https://arxiv.org/abs/2011.10566), [official code](https://github.com/facebookresearch/simsiam) | Cross-view negative cosine with stopped targets and a two-direction average. |
| Barlow Twins | [ICML paper](https://proceedings.mlr.press/v139/zbontar21a.html), [official code](https://github.com/facebookresearch/barlowtwins/blob/main/main.py) | Population-variance batch normalization and full cross-correlation redundancy reduction. |
| VICReg | [ICLR paper](https://arxiv.org/abs/2105.04906) | Separate variance, invariance and covariance terms. |
| DINO | [ICCV paper](https://arxiv.org/abs/2104.14294), [official code](https://github.com/facebookresearch/dino/blob/main/main_dino.py) | Centered/sharpened teacher targets, multi-view cross-entropy excluding matching global views, and raw-logit center EMA. |
| BEiT | [ICLR paper](https://arxiv.org/abs/2106.08254), [official code](https://github.com/microsoft/unilm/tree/master/beit) | Discrete-token cross-entropy averaged across all selected masked positions. |
| MAE | [CVPR paper](https://arxiv.org/abs/2111.06377), [official loss](https://github.com/facebookresearch/mae/blob/main/models_mae.py) | Per-patch sample-variance target normalization and global masked-patch pixel MSE. |
| iBOT | [ICLR paper](https://arxiv.org/abs/2111.07832), [official code](https://github.com/bytedance/ibot/blob/main/main_ibot.py) | Same-view masked-patch soft-target distillation, reduced within each image before the batch mean. The separate class-token branch is covered conceptually by DINO. |
| DINOv2 | [paper](https://arxiv.org/abs/2304.07193), [official KoLeo component](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py) | Non-self nearest neighbors and the KoLeo regularizer adopted by DINOv2. KoLeo alone is not the full DINOv2 recipe; DINO/iBOT exercises cover its distillation components. |
| I-JEPA | [CVPR paper](https://arxiv.org/abs/2301.08243) | Masked target-representation prediction and momentum updates. |

The source boundary is explicit in each exercise. Inputs may be already encoded
features, scores, token IDs or patch predictions. Bounded synthetic CPU experiments
optimize these components; they do not claim full image pretraining or benchmark
reproduction. In particular, DeepCluster does not reimplement PCA/Faiss, BEiT does
not train a visual tokenizer, and DINOv2's data curation and large-scale training
are described separately from its KoLeo exercise. Parameter conventions and
intentional boundary extensions are stated alongside their original sources.

## Curation and reading order

The default paper and practice lists share a learning order, rather than a
publication-date sort. The contrastive foundations come first, followed by early
pretext/clustering tasks, non-contrastive methods, redundancy reduction and
teacher/masked prediction. Explicit user-selected sorting still works.

The original papers and official code establish the numerical contracts.
Independent curricular cross-checks are Stanford's [2022 SSL lecture](https://cs231n.stanford.edu/slides/2022/lecture_14_jiajun.pdf),
[2025 SSL lecture](https://cs231n.stanford.edu/slides/2025/lecture_12.pdf),
and [2025 lecture 13 DINO/DINOv2 recap](https://cs231n.stanford.edu/slides/2025/lecture_13.pdf).
The [unified gradient analysis](https://arxiv.org/abs/2112.05141) also provides a
research comparison of contrastive, asymmetric and feature-decorrelation methods.
This selection covers established mechanism families, with high inclusion
confidence; it is not an exhaustive ranking or a claim about current popularity.

## Roadmap

The representative roadmap retains the existing 14-node readability limit.
RotNet, Instance Discrimination and DeepCluster remain fully available in the
17-paper collection and practice list. The map uses first arXiv dates and actual
publication venues, including DINOv2's TMLR 2024 publication after its 2023 preprint.

Arrows describe the labelled mechanism relationships: contrastive classification
into MoCo/SimCLR; SimCLR's two-view setup into BYOL; BYOL's stopped-target design
into SimSiam and momentum-teacher design into DINO; Barlow Twins into VICReg; DINO and BEiT into iBOT; iBOT into DINOv2. The
MAE–I-JEPA arrow explicitly compares pixel and latent masked targets rather than
asserting exclusive historical descent. No crossing, node pass-through or
transitively redundant arrow is accepted by the existing layout checks.

Six new thumbnail files are first-page renders from the linked original PDFs,
168 pixels wide, WebP quality 72, with metadata stripped. Existing BYOL and prior
collection thumbnails are reused.

The ten added exercises retain their full starter sources and provided validation
helpers. Catalog bootstrap allocations move from 96/64 KiB to 88/72 KiB for
metadata/immutable starters, respectively, within the unchanged 160 KiB total
compressed budget. The 256 KiB wire-response bounds, exact source hashes,
bounded decoder, compatibility assets, and initial JavaScript budgets remain
unchanged. This avoids removing teaching content to satisfy a per-asset split.
