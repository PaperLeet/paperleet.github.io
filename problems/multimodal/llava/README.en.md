# LLaVA visual-token insertion and label alignment

## 1. Problem

Implement `pack_visual_tokens(text: Tensor, visual: Tensor, labels: Tensor, image_index: int) -> tuple[Tensor, Tensor]`.

Pack one image into one sample and align supervision labels; features are already projected, excluding the vision encoder, language model, and data construction.

## 2. Core equations

$$
E=[E_{<i};V;E_{>i}],\qquad y=[y_{<i};\underbrace{-100,\ldots,-100}_{P};y_{>i}].
$$

## 3. From equations to code

The placeholder is a packing instruction, not an extra retained token. Concatenate its left and right slices around the visual block, using identical label boundaries. Visual embeddings provide context but have no language targets, so their labels are -100. Concatenation routes gradients back to retained sources; the removed placeholder cannot affect the output.

## 4. Conventions

One sequence: text is [T,D] with one image placeholder, visual is projected [P,D], and labels is int64 [T]. Replace the token at image_index with all P visual tokens; fill their labels with -100 and retain all other labels. Return length T+P-1 embeddings and labels. Gradients reach retained text and visual features; the removed placeholder receives zero gradient.

CPU float32/float64; no input mutation. Finite inputs only. Tests use scalar oracles, boundary cases, appropriate gradient checks, and a submitted-function experiment.

## 5. Interactive figure

The deterministic CPU experiment calls your submitted function. The chart displays measured optimization, retrieval, or grouping behavior; inspect its metrics alongside the trace. Synthetic experiments do not reproduce paper benchmark scores.

## 6. Run

```bash
paperleet test multimodal/llava --impl problems/multimodal/llava/solution.py
paperleet mutate multimodal/llava
paperleet vizdata multimodal/llava
```

## 7. Extensions

Read the cited mechanism, then extend the isolated kernel to the paper’s complete architecture and data pipeline. The exercise fixes a narrow, testable contract; its teaching simplifications are stated above.

Section 4.1 architecture; official llava_arch.py prepare_inputs_labels_for_multimodal

- [https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)
- [https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py)
