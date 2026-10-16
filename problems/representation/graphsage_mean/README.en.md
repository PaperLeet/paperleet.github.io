# GraphSAGE: Sampled Neighborhood Means and Inductive Encoding

## 1. Problem

Implement the mean aggregator and synchronous encoder from [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), Hamilton, Ying and Leskovec, NIPS 2017. Follow Algorithm 1, lines 4–7, with the mean operator of Section 3.3. The model learns shared transformations of features and neighborhoods, so those weights can encode nodes absent during training.

### Return requirements

`masked_neighbor_mean(features, neighbors, valid)` returns `[N,D]` neighbor means. `graphsage_encode(features, neighbors, masks, weights, eps=1e-12)` returns `[N,D_last]` embeddings, or the input for zero hops. Preserve dtype, device, all input values, and gradients to features and every weight. Validation helpers are supplied; retain their calls and implement the two TODO bodies.

## 2. Core formula

For sampled slots $S_k(v)$ and previous-hop features $h^{k-1}$:

$$
m_v^k=\frac{1}{|S_k(v)|}\sum_{u\in S_k(v)}h_u^{k-1},\quad
z_v^k=\operatorname{ReLU}\left([h_v^{k-1}\Vert m_v^k]W_k\right),\quad
h_v^k=\frac{z_v^k}{\max(\|z_v^k\|_2,\varepsilon)}.
$$

Here weights use the row-vector layout `[2*Din,Dout]`. Concatenate **own features first**, neighbor means second. Normalize over output features independently for each node, after every hop. All nodes at hop $k$ read hop $k-1$; updating nodes in place would introduce dependence on node order.

## 3. Formula to code

Each hop supplies its own index table `[N,S]` and Boolean mask of the same shape. Sanitize masked IDs **before** gathering: a masked slot may contain `99` or `-99`. Sum only valid slots and divide by their count, clamped to at least one. Duplicate IDs count repeatedly because slots describe samples, not a set. An empty neighborhood returns a zero context, while the own-feature branch still contributes.

For features `[[1,2],[-3,4],[5,-2]]`, sampled rows `[1,1,pad]`, `[0,2,pad]`, and `[pad,pad,pad]` produce means `[[-3,4],[3,0],[0,0]]`. Do not implicitly add a self-loop or remove duplicates. A node contributes to its neighbor branch only when its own ID is explicitly sampled.

## 4. Exercise contract

| Input | Contract |
| --- | --- |
| Features | Finite float32/float64 `[N,D]`, `N>=0`, `D>0` |
| Sampling | A list/tuple of integer int32/int64 `[N,S_k]` tables; valid IDs are in `[0,N)` |
| Masks | One Boolean `[N,S_k]` table per hop; masked IDs are ignored |
| Weights | One finite `[2*Din,Dout]` matrix per hop, `Dout>0`, matching feature dtype/device |
| Depth | Equal lengths for sampling, masks and weights; zero hops is the identity |
| Epsilon | Finite positive scalar, excluding bool; denominator is `max(norm,eps)` |

Strided tensors, zero nodes, zero sampled slots and all-masked rows are supported. Keep autograd connected through every aggregation and normalization. Inputs must remain unchanged.

**Scope:** this is Algorithm 1 with a mean aggregator and ReLU at each hop. Equation (2) is a different GCN-style variant that averages the node together with neighbors. The authors' TensorFlow implementation also has configurable projection/concatenation and final-layer activation choices; matching that entire implementation is outside this exercise. Fixed sampled tables, padding, empty-neighborhood behavior and epsilon stabilization are explicit teaching extensions.

## 5. Experiment

Train two shared encoder layers and a linear classifier: 80 parameters, 100 real Adam updates. All supervised query nodes have identical zero features; their balanced labels can be inferred from their anchor neighborhoods. A feature-only classifier can achieve at most 50% on these queries.

Freeze the learned weights and evaluate a distinct graph with new anchor features, a new node permutation, and 18 nodes instead of the training graph's 14. Charts show actual training loss, query accuracy, and frozen-model accuracy on both graphs. Submitted-run charts call your encoder; reference previews use the supplied reference.

This is a small synthetic teaching experiment. It uses no learned node-ID lookup, original benchmark data, pretrained model, GPU, or PyG. Its accuracy is not a paper benchmark result.

## 6. Run

Complete the web editor and run all tests, or use:

```bash
paperleet test representation/graphsage_mean --impl problems/representation/graphsage_mean/solution.py
paperleet mutate representation/graphsage_mean
```

## 7. Further study

T0 checks contracts and empty inputs. T1 is hand-computed. T2 compares independent scalar calculations, gradients, sampling/node permutations, and per-hop normalization. T3 trains and evaluates your encoder on two different synthetic graphs.

Why does neighbor order not matter while sampling multiplicity does? How does a detached neighbor mean change gradients to earlier hops? Explain why learning a separate embedding for each training node would prevent the same inductive transfer.
