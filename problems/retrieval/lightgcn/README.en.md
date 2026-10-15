# LightGCN: Normalized Bipartite Graphs and Collaborative Propagation

A user-item history defines a bipartite graph. Users who observed the same item can exchange signals through two graph steps. Implement the graph and propagation core from [LightGCN](https://arxiv.org/abs/2002.02126), then inspect a small synthetic training run.

## 1. Symmetric graph normalization

`normalized_bipartite_graph(interactions, dtype=None)` takes Boolean `[U,I]` histories and returns a floating `[N,N]` graph, `N=U+I`. Users precede items; IDs start at zero with no padding. The default is float32; explicit float64 is also supported. Preserve the input device.

$$
A=\begin{bmatrix}0&R\\R^\top&0\end{bmatrix},\qquad
\widehat A=D^{-1/2}AD^{-1/2},\qquad D_{vv}=\sum_w A_{vw}.
$$

An observed edge has weight `1/sqrt(degree_user*degree_item)` in both directions. Do not add self loops. Isolated rows and columns stay zero; clamping degrees to at least one avoids division by zero without creating edges. Support empty user/item axes, all-empty graphs and strided histories. Preserve input values.

## 2. Linear propagation and layer aggregation

`lightgcn_embeddings(initial, normalized_graph, layers=2)` takes `[N,D]` initial embeddings and an `[N,N]` graph, with `D>0` and matching float32/float64 dtype and device. Depth is a nonnegative Python integer; Boolean depth is rejected.

$$
E^{(0)}=E_0,\qquad E^{(k+1)}=\widehat A E^{(k)},\qquad
E=\frac{1}{K+1}\sum_{k=0}^K E^{(k)}.
$$

Preserve output shape, dtype and device. Depth zero returns the initial values; zero nodes are valid. Each step is only a graph multiplication. There are no feature transforms, activations or implicit self loops. The paper permits general layer weights; this exercise uses its experimental uniform weights.

The initial layer is included, so an isolated node retains `E0/(K+1)`. Keep gradients through both differentiable tensor inputs and support strided layouts without mutation. Validation accepts any finite square operator of the matching dtype/device; tests also use general operators to check derivatives. The supplied training loop holds the graph fixed and learns **only E0**.

## 3. Two-hop collaborative influence

A user's signal travels through a shared item to another user. A popular shared item contributes less. Hand-computed two-hop coefficients, independent scalar loops, finite differences, linear superposition and node relabeling test distinct properties of this operation.

## 4. Actual training experiment

Four synthetic users observe `{0,1}`, `{1,2}`, `{3,4}` and the empty set. Item 5 is unobserved. Users 0 and 1 share item 1; user 3 and item 5 are isolated. A `10×3` initial table supplies 30 trainable entries. Propagate twice and average layers 0, 1 and 2.

The provided loop enumerates all 24 positive/unobserved triples and performs 160 actual Adam updates:

$$
L=\frac{1}{24}\sum_{(u,i,j)}\operatorname{softplus}\left(-e_u^\top(e_i-e_j)\right)
+0.001\lVert E_0\rVert_F^2.
$$

Scores use the aggregated vectors; regularization applies once to the complete initial table. Equation 15 sums its data term, whereas this exercise averages it; `l2=lambda_paper/24` converts the coefficient to this convention. The stable BPR objective and training loop are supplied.

Charts show the measured loss trajectory and three active users' item scores. Pair accuracy and Recall@2 are training metrics; the empty-history user is excluded from Recall. Counting conflicting item-pair directions gives any single global item-score ordering a strict pair-accuracy upper bound of `17/24`. This is a counting bound, not a measured baseline model.

This is a **PaperLeet teaching adaptation** with a tiny dense graph, fixed synthetic histories, full-batch updates and one initial-table prior. It does not reproduce sparse large-scale training, real datasets or held-out paper Recall/NDCG. The CPU/WASM browser judge needs no GPU, account, weights or external API. Unsupported custom PyTorch APIs retain the native judge path.

## 5. Sources and next steps

The [paper](https://arxiv.org/pdf/2002.02126), Sections 3.1.1–3.1.3, Equations 3–8 specify propagation and aggregation; Section 3.2.3, Equations 13–14 analyze two-hop influence; Section 3.3, Equation 15 defines the objective. See the [authors' implementation](https://github.com/kuandeng/LightGCN) for sparse graph training. Compare BPR's pairwise objective, SASRec's sequence model and TIGER's generated retrieval IDs.
