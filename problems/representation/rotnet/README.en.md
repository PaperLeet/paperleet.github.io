# RotNet: Predicting Four Image Rotations


## 1. Problem


Implement `rotation_batch(images)` and `rotation_loss(logits, labels)`. RotNet learns by predicting which of four image rotations was applied. Transformation choices supply the labels automatically. This exercise preserves the exact image geometry and trains a small rotation classifier.


## 2. Core equations


For counterclockwise rotations $r\in\{0,1,2,3\}$ with angles $90r$ degrees,
$$\widetilde x_{rB+b}=R_{90r}(x_b),\quad y_{rB+b}=r,$$
$$L=\frac1M\sum_{i=1}^M[\operatorname{logsumexp}_{k=0}^3(s_{ik})-s_{i,y_i}].$$
Views are in four rotation-major blocks. A 90-degree counterclockwise rotation maps `[[1,2],[3,4]]` to `[[2,4],[1,3]]`. Equal logits yield `log(4)`.


## 3. Equations to code


Use transpose and flips on the two spatial axes, concatenate the 0°, 90°, 180°, 270° blocks, and repeat each label B times. Preserve the image autograd path: rotation is a pixel permutation. For the loss, gather the supplied class column, subtract it from stable row log-sum-exp, and average examples.


## 4. Exercise conventions


Images are nonempty finite float32/float64 `[B,C,H,H]`; square shapes allow all views to share a tensor. Output is `[4B,C,H,H]` plus int64 `[4B]` labels on the same device. Rotation-major batching is this exercise’s convention; the official data loader first creates views per sample. There is no interpolation, crop, color transform or input mutation. Loss accepts any nonempty `[M,4]` logits and int64 `[M]` labels in `[0,3]`, even when M is not a multiple of four. Strided images work. A 1×1 image still gets all four labels although those views are indistinguishable.


## 5. Interactive plots and CPU lab


Eight deterministic asymmetric 4×4 glyphs generate 32 rotated examples. A zero-initialized 16×4 linear head learns for 80 full-batch SGD updates at learning rate `.6`. Both submitted functions are used. T3 requires rotation accuracy 1 and loss below `.06` and 5% of its initial value. Symmetric images cannot always reveal their rotation; the synthetic glyphs are chosen to make this pretext task learnable. Rotation accuracy alone is not evidence of transfer representation quality.


## 6. Running the exercise


Use **Run tests** for CPU/WASM or the commands below. Tests cover exact pixel locations and labels, four-turn identity, pixel multisets, strided image VJPs, independent cross entropy, stable gradients and actual classifier training.


```bash
paperleet test representation/rotnet --impl problems/representation/rotnet/solution.py
paperleet mutate representation/rotnet
paperleet vizdata representation/rotnet
```



## 7. Extensions and sources


The paper studies transfer from rotation prediction to classification, detection and segmentation. Those evaluations require a trained image encoder and separate downstream data. Related early pretext tasks predict spatial context or solve jigsaws. This module is distinct from the existing VQ “Rotation Trick” exercise, which concerns a gradient estimator rather than image-rotation supervision.


[Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728) · [Official implementation](https://github.com/gidariss/FeatureLearningRotNet/blob/master/dataloader.py)
