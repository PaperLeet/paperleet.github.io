# MoCo: momentum encoder and negative dictionary

Implement `moco_loss`, `momentum_update` and `enqueue`, based on [MoCo](https://arxiv.org/html/1911.05722v3), Equations 1–2 and Algorithm 1.

## 1. One positive and K dictionary negatives

`query` and `key` have matching `[B,D]` shape; `queue` is `[K,D]`, with B,K,D>=1. Inputs share finite float32/float64 dtype/device, including strided views. Normalize each row by `max(L2 norm,1e-12)`; zero rows remain zero.

For query i, key i is its only positive; all K dictionary rows are negatives. Other current-batch keys are not implicitly negatives. Divide normalized dot products by temperature and return the scalar batch mean:

$$L=\frac1B\sum_i\left[\operatorname{logsumexp}(s_i^+,s_{i,0}^-,\ldots,s_{i,K-1}^-)-s_i^+\right].$$

The denominator includes the positive. Preserve dtype/device and inputs. Temperature must be a finite positive Python int/float, excluding bool and Tensor; supplied helpers validate the contract. Keep query gradients through normalization, but detach both key and queue, whose gradients must be None. Use stable logsumexp.

With q=k=[1,0] and queue=[[-1,0],[0,1]], loss is `log(1+exp(-2/T)+exp(-1/T))`. Identical nonzero representations give `log(K+1)`.

## 2. Momentum encoder

`momentum_update(query,key,momentum=.999)` operates on encoder parameters:

$$\theta_k^{new}=m\theta_k+(1-m)\theta_q.$$

Accept matching finite float32/float64 tensors of arbitrary shape, including scalars and empty tensors, and a finite Python coefficient in [0,1], excluding bool/Tensor. Return independent detached storage; never mutate or alias inputs. m=0 copies query and m=1 freezes key. The frozen endpoint is an explicit exercise extension; the paper uses m<1 during training.

## 3. Circular dictionary

`enqueue(queue,keys,pointer)` accepts `[K,D]` old dictionary and `[B,D]` new keys, K,D>=1 and B>=0, including B>K. Require matching finite floating dtype/device and an exact Python int pointer in [0,K), excluding bool.

Normalize and detach incoming rows, sequentially overwrite from pointer with wraparound, and return `(independent detached dictionary,(pointer+B)%K)`. Leave both inputs unchanged. An empty batch returns a copy with the same pointer. Multiple wraps must preserve the last K sequential writes.

Row-major `[K,D]` layout and arbitrary-batch wraparound are teaching interface choices, rather than a verbatim copy of the original project's queue representation.

## 4. Comparison with SimCLR

SimCLR builds candidates from both current views and differentiates both. MoCo uses a historical dictionary and differentiates only queries. EMA slows key-encoder changes while FIFO expires older representations. Cache this step's keys before updating either encoder; enqueue those keys after EMA, without recomputing them.

## 5. Actual CPU experiment

The supplied experiment uses six fixed paired 4D features, a 4×3 query linear encoder (12 trainable scalars), an initially equal key encoder, and five dictionary entries. Twelve actual SGD steps use batch=2, lr=.05, m=.8 and T=.3: encode q/k → detached-key loss → query SGD → key EMA → enqueue cached keys.

An independent Python scalar objective and central finite differences drive a separate complete parameter/EMA/dictionary trajectory. Compare every state at every step. Charts show actual batch loss, encoder parameter distance and queue pointer; no monotonic-loss or fabricated accuracy threshold is imposed. Further tests cover query finite differences, both detached branches, hand denominators, invariances, empty/oversized batches, multiple wraps and storage independence.

This original PaperLeet synthetic teaching adaptation has no images, ResNet, augmentation pipeline, distributed shuffling BN or downstream evaluation, and makes no paper-performance reproduction claim. No GPU, account, data download or pretrained weights are required.

## 6. Run

```bash
paperleet init representation/moco
paperleet test representation/moco
paperleet test representation/moco --impl problems/representation/moco/solution.py
paperleet vizdata representation/moco
```

CPU/WASM browser judging runs all T0–T3 tests. PyTorch usage outside the supported browser subset retains native server judging.
