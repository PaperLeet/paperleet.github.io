# Instance Discrimination: Binary NCE and Memory Bank


## 1. Problem


Implement `instance_nce_loss` and `update_memory` for the original non-parametric instance-discrimination method. Every training image has a persistent bank row. The current query is contrasted against a positive instance and sampled noise entries. This paper uses binary noise-contrastive estimation; candidate-softmax InfoNCE is a different objective.


## 2. Core equations


For query $q_b$, candidate IDs $j_{bt}$, fixed partition constant $Z>0$, temperature $T>0$, and $K$ noise draws from $N$ bank rows,
$$a_{bt}=q_b^T v_{j_{bt}}/T-\log Z-\log(K/N),$$
$$L=\frac1B\sum_b\left[\operatorname{softplus}(-a_{b0})+\sum_{t=1}^{K}\operatorname{softplus}(a_{bt})\right].$$
The positive is column zero. Each binary classifier distinguishes the model likelihood from uniform noise. When all $a=0$, loss is $(K+1)\log2$. Update a selected row by $v_i\leftarrow\operatorname{normalize}(m v_i+(1-m)q_i)$.


## 3. Equations to code


Gather bank rows with `memory.detach()`, compute dot products, and work in the log domain to avoid exponential overflow. Sum the K negative terms and average only over B. Do not drop `log_z` or replace `K/N` with `1/N`. Unlike softmax, shifting all model log-likelihoods changes binary NCE. Return an independent detached bank copy after indexed EMA and normalization.


## 4. Exercise conventions


Queries `[B,D]`, bank `[N,D]`, candidate IDs `[B,K+1]`, K≥1. Floats are finite float32/64; IDs are int64 and in range on the same device. Features normally come from a normalized encoder, but this function consumes their supplied values without additional normalization. `log_z` is a fixed finite Python scalar; the paper/official code estimate Z once, which is outside this exercise. Uniform noise draws may repeat and may hit the positive ID. Memory gradients stop; query gradients remain. Memory updates require distinct IDs, momentum in `[0,1]`, and norm floor `eps=1e-12`; untouched rows stay exactly unchanged and zero mixtures remain zero.


At momentum one, selected rows are still normalized; the return always has independent storage, even when no numerical update is needed.

## 5. Interactive plots and CPU lab


Train six normalized six-dimensional instance vectors for 80 SGD steps at learning rate `.35`, using T=`.2`, fixed Z=`exp(5)+5` and bank momentum `.95`. All five other toy IDs are deterministic negative candidates; this schedule illustrates the objective and is not the original stochastic sampler. The lab invokes both submitted functions. T3 requires loss below `.8` and 40% of its initial value, own-bank cosine above `.99`, and unit, detached bank rows. No image encoder or downstream benchmark is reproduced.


## 6. Running the exercise


Run the full CPU/WASM suite on the website or the commands below. Tests independently evaluate binary probabilities, query finite differences and strided VJPs, frozen memory, repeated noise, EMA endpoints, instance relabeling and extreme scores.


```bash
paperleet test representation/instance_discrimination --impl problems/representation/instance_discrimination/solution.py
paperleet mutate representation/instance_discrimination
paperleet vizdata representation/instance_discrimination
```



## 7. Extensions and sources


Compare InfoNCE candidate classification with this calibrated binary data-versus-noise classifier. Compare MoCo: MoCo uses a momentum-updated encoder and FIFO queue, whereas this method updates persistent rows addressed by dataset instance ID. The stable log-domain formula here is algebraically the binary NCE objective; it omits the official probability-space denominator’s tiny implementation epsilon to avoid changing calibrated probabilities. The bank update is explicit after the optimizer step rather than hidden in a custom backward operation.


[Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://arxiv.org/abs/1805.01978) · [Official implementation](https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/NCECriterion.py)
