# 111. VisPlay Image-Grouped Rewards

## 1. Problem

Generate question groups for unlabeled images and penalize repetition within each image. Different images may reuse local cluster IDs.

Implement `image_group_rewards`.

## 2. Core equations

$$
r_i=v_i\max(0,1-|2p_i-1|-\lambda |C_i^{(I)}|/G_I).
$$

## 3. From equations to code

Group by image IDs before comparing cluster IDs. Two duplicates among four questions for one image with p=0.5 earn 0.5. Adding questions for another image must not change that group’s rewards.

## 4. Exercise conventions

The denominator is the image-local question count, not the global batch. Reused cluster IDs across images do not imply duplication. VisPlay still requires unlabeled images; this exercise implements reward aggregation over observed feedback.

Finite CPU inputs, no mutation; float32/float64 scores, long IDs and bool masks. The APIs below define all shape, tie and gradient conventions.

~~~python
def image_group_rewards(consistency: Tensor, image_ids: Tensor, clusters: Tensor, valid: Tensor, strength: float = 1.) -> Tensor:
    """Float consistency [B] in [0,1]; long image/cluster IDs [B]; bool valid [B].
    Cluster IDs are LOCAL to each image. For each row count all proposals for its
    image (including invalid ones), and the same-image, same-cluster proposals.
    Return detached same-dtype [B]: valid*relu(1-abs(2*p-1)-strength*cluster/group).
    B>=1; IDs need not be contiguous; strength>=0. No mutation."""
~~~

## 5. Interactive figures and experiment

Open Explanation to step through the paper’s co-evolution loop: active model, frozen evaluator, feedback and filtering. Controls recompute the displayed teaching scores. The Lab trains a small categorical policy for 60 gradient steps at learning rate 0.5, starting from uniform logits, using your functions. It uses fixed feedback records and CPU float64; target probability must exceed 0.72 and improve by more than 0.3. This checks the local mechanism, not full LLM/VLM self-evolution or benchmark gains.

## 6. Running the exercise

Complete the starter and use Run tests. T0 checks interfaces; T1 uses independent scalar/counting oracles; T2 checks boundaries, invariances and gradient ownership; T3 runs your implementation in the toy learning loop.

~~~bash
paperleet test self_evolving/visplay_rewards --impl problems/self_evolving/visplay_rewards/solution.py
paperleet mutate self_evolving/visplay_rewards
~~~

## 7. Further questions

Why can global deduplication incorrectly penalize valid image-specific questions?

Source: [VisPlay: Self-Evolving Vision-Language Models from Images](https://arxiv.org/abs/2511.15661), Section 2.3, Eqs. 5–8: per-image diversity and uncertainty.
