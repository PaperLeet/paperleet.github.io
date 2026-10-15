# BPR: Implicit Feedback and Personalized Pairwise Ranking

Learn a ranking for each user from observed-versus-unobserved preferences. This exercise selects the training triples and matrix-factorization objective from [BPR Sections 3.2, 4.1 and 4.3.1](https://proceedings.mlr.press/r7/rendle09a.html).

## 1. Per-user negative candidates

`eligible_negatives(interactions, user_ids)` takes a Boolean `[U,I]` history and int64 `[B]` user IDs. Return Boolean `[B,I]`:

$$M_{b,j}=\neg S_{u_b,j}.$$

A `True` entry is unobserved by **this user**, making it an eligible comparison item rather than a known negative label. Exclude every item in that user's history. Preserve order and repeated users; another user's observed item remains eligible here. IDs are zero-based with **no padding**. Support empty batches, empty item axes, fully observed and entirely unseen users.

## 2. Pairwise matrix-factorization loss

`bpr_mf_loss(user_factors, item_factors, triples, l2=0.)` takes matching float32/float64 `[U,D]` and `[I,D]` factor tables, plus int64 `[B,3]` rows `(user, positive, unobserved)`, with `D > 0`. Triples are already constructed. Gather their factor rows and return the scalar:

$$x_{uij}=w_u^\top(h_i-h_j),$$
$$L=\frac{1}{\max(1,B)}\sum_{(u,i,j)}\operatorname{softplus}(-x_{uij})
+\lambda\left(\sum_{u,d}W_{u,d}^2+\sum_{i,d}H_{i,d}^2\right).$$

Use stable log-sigmoid/softplus rather than sigmoid followed by a direct log. Sum latent coordinates, average triples and retain all three selected-row gradient contributions. Repeated user/item IDs accumulate gradients. Preserve dtype, device, input values and support noncontiguous tables. Keep the supplied validation calls.

## 3. Global prior convention

`l2` is finite and nonnegative. Penalize both **complete tables**, including rows not selected in this batch, exactly once with no additional factor of one half. Penalizing gathered vectors changes the objective when IDs repeat or are absent.

The paper's BPR-Opt sums log likelihoods. This exercise normalizes the data term and exposes the prior coefficient at that scale: for a nonempty batch, `l2 = lambda_paper / B` corresponds to dividing the summed objective by `B`. An empty batch contributes zero likelihood while retaining the complete prior and both gradient graphs.

The pairwise score difference is essential. Two independent binary terms, as used in the SASRec exercise, define a different objective.

## 4. Verification

T0 checks shape, dtype, index validation, empty axes and unchanged inputs. T1 is hand-computable. T2 uses independent scalar likelihoods and central differences through both factor tables, repeated indices, gradients of unselected rows, batch replication, common item translation and extreme margins. Finite values and genuine gradients are required.

## 5. Actual training experiment

Three synthetic users like `{0,1}`, `{2,3}` and `{4,5}` respectively. Your mask supplies candidates for 24 training triples. Your objective trains all 27 factor entries for **120 actual Adam updates**. The charts show its loss and personalized item scores.

Each cross-group item pair has opposing preferences from two users. A single global item-score order can satisfy at most half of these strict comparisons. The displayed `1/2` bound is counted from directions, not measured from a baseline model. Personalized factors must exceed it and recover all training preferences.

This is a **PaperLeet teaching adaptation** using full batches and one global prior coefficient. It does not reproduce LearnBPR's bootstrap SGD, separate role-specific regularizers, public datasets or the paper benchmarks. Accuracy and Recall@2 are training metrics, without held-out generalization claims.

## 6. Run in the browser

Implement both functions and run all tests. CPU/WASM needs no GPU, account, model, dataset download or external API. Unsupported custom PyTorch operations retain the server path.

## 7. Sources and next steps

Read the [original paper](https://auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf), [publication record](https://proceedings.mlr.press/r7/rendle09a.html) or [arXiv archive](https://arxiv.org/abs/1205.2618). Compare with SASRec's sampled binary objective, ColBERT's paired softmax and TIGER's generated semantic IDs.
