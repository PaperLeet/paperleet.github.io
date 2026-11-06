# DINOv2: KoLeo Feature-Spreading Regularization


## 1. Problem


Implement `nearest_neighbors` and `koleo_loss`, the feature-spreading regularizer used in DINOv2. KoLeo rewards larger nearest-neighbor distances on normalized representations. DINOv2 combines DINO/iBOT-style learning with data curation and training improvements; this exercise isolates one adopted component. KoLeo predates DINOv2 and is not claimed as its invention.


## 2. Core equations


First normalize $z_i=x_i/\max(\|x_i\|_2,\epsilon)$, then select $n_i=\arg\max_{j\ne i}z_i^Tz_j$. Match the official implementation: PairwiseDistance uses a fixed coordinate epsilon $\epsilon_p=10^{-8}$; the function's `eps` controls normalization and the outer logarithm:
$$d_i=\sqrt{\sum_a(z_{ia}-z_{n_i a}+\epsilon_p)^2},\qquad L=-\frac1B\sum_i\log(d_i+\epsilon).$$
The coordinate-wise epsilon inside the distance is always `1e-8`, matching the separately constructed PyTorch `PairwiseDistance` object. Changing `eps` does not change it; it is not an epsilon added after an ordinary norm. Both gathered endpoints contribute gradients. The index selection itself is discrete.


## 3. Equations to code


Normalize rows, compute the detached pairwise cosine matrix, exclude diagonal entries, and select each row’s maximum. Gather neighbors from the differentiable normalized tensor, compute distances with a fixed `1e-8` coordinate shift and average negative log distances. Detaching the gathered vectors loses valid neighbor gradients. The negative sign makes minimizing the loss spread close points.


## 4. Exercise conventions


Use finite float32/float64 `[B,D]`, B≥2 and D≥1; default `eps=1e-8` must be finite and positive. `nearest_neighbors` accepts already normalized rows and returns int64 IDs. Strict self-exclusion and lowest-index ties are explicit boundary conventions; the official code masks the diagonal with -1, which can admit self in an exact antipodal tie. This exercise masks it with negative infinity. Preserve input data, floating dtype/device and strided gradient paths. Zero rows normalize to zero. At neighbor-switch boundaries the argmax is nondifferentiable; gradient tests use unique neighbors.


Nonzero vectors shorter than `eps` divide by that caller-supplied norm floor. Independent scalar and finite-difference checks cover this linear normalization region. The fixed-inner-epsilon convention matches the official code for nondefault `eps` as well.

## 5. Interactive plots and CPU lab


Six three-dimensional vectors start close together and undergo 100 SGD updates at learning rate `.06` using the submitted loss. There are 18 trainable entries. The plot records actual loss and an independently measured mean nearest-neighbor Euclidean distance. T3 requires final distance above 1 and more than five times its initial value, with final loss below zero. This demonstrates KoLeo alone; it neither trains a DINOv2 image encoder nor establishes semantic quality or collapse prevention for an arbitrary network.


## 6. Running the exercise


Click **Run tests** for CPU/WASM or use the commands below. The suite checks explicit scalar distances, both epsilon placements, non-self/tie behavior, finite-difference gradients for both endpoints, strided VJPs, scaling, permutations, collisions and actual optimization.


```bash
paperleet test representation/dinov2_koleo --impl problems/representation/dinov2_koleo/solution.py
paperleet mutate representation/dinov2_koleo
paperleet vizdata representation/dinov2_koleo
```



## 7. Extensions and sources


Complete DINO and iBOT to study the teacher/student objectives used by the broader recipe. Then compare covariance regularizers such as VICReg and Barlow Twins with KoLeo’s nearest-neighbor geometry. The DINOv2 official loss credits Sablayrolles et al., “Spreading vectors for similarity search” (2018), for the entropy regularizer. Data curation, large ViT training and distillation are separate parts of the DINOv2 result.


[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) · [Official implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py)
