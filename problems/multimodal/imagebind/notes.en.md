# ImageBind image-anchored multimodal objective — Solution

Images form the center of the pairing graph. Construct [B,B] similarities for each image-modality edge, compute a bidirectional loss, then average over M. Shared image identities organize representations without direct text-audio pairs. The experiment fixes image anchors, trains two synthetic modalities, and measures retrieval between them.

$$
L=\frac1M\sum_{m=1}^M\frac{L_{I\to m}+L_{m\to I}}2,\quad S^{(m)}_{ij}=\hat I_i^T\hat X^{(m)}_j/\tau.
$$

The reference implementation follows the declared layout and reduction:

```python
def image_anchor_loss(images: Tensor, modalities: Tensor, temperature: Tensor) -> Tensor:
    _floating(images,modalities,temperature)
    if images.ndim!=2 or images.numel()==0 or modalities.ndim!=3 or modalities.numel()==0 or modalities.shape[1:]!=images.shape or temperature.ndim!=0 or temperature.item()<=0:
        raise ValueError("expected (B,D), (M,B,D), positive scalar temperature")
    images=F.normalize(images,dim=-1,eps=1e-12);modalities=F.normalize(modalities,dim=-1,eps=1e-12)
    losses=[]
    for m in range(modalities.shape[0]):
        s=images@modalities[m].T/temperature
        losses.append(-(F.log_softmax(s,1).diag().mean()+F.log_softmax(s,0).diag().mean())/2)
    return torch.stack(losses).mean()
```

Common errors caught by mutation tests:

- Uses a sum instead of averaging modality losses
- Trains only the first modality

Validation: T0 checks the contract; T1 uses an independent oracle; T2 covers boundaries, invariants, and gradients where defined; T3 executes the supplied function on deterministic synthetic inputs. See experiment.py for the complete inputs, update rule, step count and metrics. The gate thresholds are asserted in the T3 test.

Implement the image-anchored pairing graph and reduction across modalities; align synthetic modalities without direct nonimage pairs, excluding the six encoders.

- [https://arxiv.org/abs/2305.05665](https://arxiv.org/abs/2305.05665)
