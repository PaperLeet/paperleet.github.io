# DeepCluster: Learning from Cluster Pseudo-Labels


## 1. Problem


Implement `normalize_features`, `assign_clusters`, and `pseudo_label_loss`. DeepCluster repeatedly groups current image features and trains an encoder to predict those cluster IDs. The labels come from the representation, not annotated categories. This module starts after the paper's PCA/whitening preprocessing; it isolates the L2, nearest-center and pseudo-label training steps.


## 2. Core equations


For supplied whitened rows $x_i$, set $z_i=x_i/\max(\|x_i\|_2,\epsilon)$. Given ordinary K-means centers $c_k$,
$$y_i=\arg\min_k\|z_i-c_k\|_2^2,\qquad L=\frac1B\sum_i[\operatorname{logsumexp}_k(s_{ik})-s_{i,y_i}].$$
Centers are arithmetic means and need not have unit length. Do not turn this Euclidean assignment into a cosine classifier. Ties choose the lowest center index. Uniform logits produce `log(K)`.


## 3. Equations to code


Normalize feature rows with a norm floor, broadcast feature-center differences and reduce over the feature axis, then take `argmin` over centers. Cluster assignment is a detached, discrete target. Classification gradients flow through all logits; normalization remains differentiable when used in the encoder path. Use stable log-sum-exp for mean cross entropy.


## 4. Exercise conventions


All floating matrices are finite float32/float64, nonempty, with matching feature widths, dtype and device. Labels are int64 and in range. Accept strided inputs and leave them intact. Zero rows normalize to zero; the norm floor is `eps=1e-12`. `assign_clusters` accepts features already normalized by the caller and does not renormalize centers. Loss is a mean over supplied examples: the original training sampler balances clusters; this function does not silently insert class weights. Provided validation remains in the starter.


## 5. Interactive plots and CPU lab


The lab alternates cluster refresh and real learning on nine deterministic 2D points. A 2×2 encoder and 2×3 head have ten trainable entries. It performs 90 SGD updates at learning rate `.25`, refreshing assignments and mean centers at updates 0, 30, 60 and 90. Empty toy centers retain their previous values. The supplied experiment calls all three submitted functions. T3 requires all three clusters occupied, encoder motion above `.01`, pseudo-label accuracy 1 and final loss below `.1`. These are synthetic training metrics, without fitting PCA or training an image encoder.


## 6. Running the exercise


Click **Run tests** for CPU/WASM execution, or run the commands below. The suite checks scalar geometry, normalization VJPs including strided storage, hard-label semantics, stable classification gradients, relabeling and the complete training loop.


```bash
paperleet test representation/deepcluster --impl problems/representation/deepcluster/solution.py
paperleet mutate representation/deepcluster
paperleet vizdata representation/deepcluster
```



## 7. Extensions and sources


The original system refits PCA/whitening and K-means over extracted image features and retrains with cluster-balanced sampling. Study those outer-loop decisions after completing this core. Compare SwAV, which uses balanced soft assignments and swapped-view prediction. Paper and official implementation are linked below. The toy data, boundary choices, tests and small optimizer are original teaching material.


[Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520) · [Official implementation](https://github.com/facebookresearch/deepcluster/blob/main/clustering.py)
