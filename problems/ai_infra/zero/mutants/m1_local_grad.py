# Bug: 使用每rank自己的梯度而未做跨rank归约
import torch
from torch import Tensor

def zero2_adam_step(params: Tensor, gradients: Tensor, first: Tensor, second: Tensor, step: int, lr: float, beta1: float = .9, beta2: float = .999, eps: float = 1e-8) -> tuple[Tensor,Tensor,Tensor]:
    if params.ndim!=2 or min(params.shape)<=0 or first.shape!=params.shape or second.shape!=params.shape or gradients.shape!=(params.shape[0],params.numel()):
        raise ValueError('incompatible owner shards and gradient replicas')
    if type(step) is not int or step<1 or lr<=0 or eps<=0 or not 0<=beta1<1 or not 0<=beta2<1:
        raise ValueError('invalid Adam hyperparameters')
    if any(t.dtype!=params.dtype or t.device!=params.device for t in (gradients,first,second)) or bool((second<0).any()):
        raise ValueError('matching tensors and nonnegative variance required')
    g=gradients[0].reshape(params.shape)
    m=beta1*first+(1-beta1)*g
    v=beta2*second+(1-beta2)*g*g
    updated=params-lr*(m/(1-beta1**step))/(torch.sqrt(v/(1-beta2**step))+eps)
    return updated,m,v

def state_bytes(parameters: int, world_size: int, stage: int) -> int:
    if type(parameters) is not int or type(world_size) is not int or parameters<=0 or world_size<=0 or parameters%world_size or type(stage) is not int or stage not in (0,1,2,3):
        raise ValueError('equal nonempty shards and stage 0..3 required')
    weights=2*parameters//(world_size if stage>=3 else 1)
    gradients=2*parameters//(world_size if stage>=2 else 1)
    optimizer=12*parameters//(world_size if stage>=1 else 1)
    return weights+gradients+optimizer
