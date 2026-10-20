# Implementation notes

Group by image IDs before comparing cluster IDs. Two duplicates among four questions for one image with p=0.5 earn 0.5. Adding questions for another image must not change that group’s rewards.

The denominator is the image-local question count, not the global batch. Reused cluster IDs across images do not imply duplication. VisPlay still requires unlabeled images; this exercise implements reward aggregation over observed feedback.

The reference behavior is defined by the starter, including deterministic tie handling and caller-supplied upstream observations. The toy policy is a controlled test, not evidence that model-generated labels are correct. Compare a held-out gold check separately from agreement with synthetic labels.

https://arxiv.org/abs/2511.15661
