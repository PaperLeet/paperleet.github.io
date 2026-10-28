# LLaVA visual-token insertion and label alignment — Solution

The placeholder is a packing instruction, not an extra retained token. Concatenate its left and right slices around the visual block, using identical label boundaries. Visual embeddings provide context but have no language targets, so their labels are -100. Concatenation routes gradients back to retained sources; the removed placeholder cannot affect the output.

$$
E=[E_{<i};V;E_{>i}],\qquad y=[y_{<i};\underbrace{-100,\ldots,-100}_{P};y_{>i}].
$$

The reference implementation follows the declared layout and reduction:

```python
def pack_visual_tokens(text: Tensor, visual: Tensor, labels: Tensor, image_index: int) -> tuple[Tensor, Tensor]:
    _floating(text,visual)
    if text.ndim!=2 or visual.ndim!=2 or text.numel()==0 or visual.numel()==0 or text.shape[1]!=visual.shape[1] or labels.shape!=(text.shape[0],) or labels.dtype!=torch.long or not 0<=image_index<text.shape[0]:
        raise ValueError("expected nonempty compatible features, long labels and valid image index")
    packed=torch.cat((text[:image_index],visual,text[image_index+1:]),dim=0)
    ignored=torch.full((visual.shape[0],),-100,dtype=labels.dtype,device=labels.device)
    packed_labels=torch.cat((labels[:image_index],ignored,labels[image_index+1:]),dim=0)
    return packed,packed_labels
```

Common errors caught by mutation tests:

- Keeps the image placeholder in the embedding sequence
- Assigns language targets to image features

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Pack one image into one sample and align supervision labels; features are already projected, excluding the vision encoder, language model, and data construction.

- [https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)
- [https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py)
