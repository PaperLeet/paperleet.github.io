import math
import torch
from torch import Tensor


def _float(x, ndim=None):
    if not isinstance(x, Tensor) or not x.is_floating_point() or x.numel() == 0 or (ndim is not None and x.ndim != ndim) or not torch.isfinite(x).all():
        raise ValueError("expected a nonempty finite floating tensor with the declared rank")


def _same(a, b):
    _float(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("tensor shape, dtype and device must match")

def imitation_reward(pose_angles: Tensor, velocity_errors: Tensor, end_errors: Tensor, com_errors: Tensor) -> Tensor:
    """DeepMimic imitation reward for precomputed quaternion-angle and vector errors."""
    _float(pose_angles,2);_float(velocity_errors,3);_float(end_errors,3);_float(com_errors,2)
    if velocity_errors.shape!=(*pose_angles.shape,3) or end_errors.shape[0]!=pose_angles.shape[0] or end_errors.shape[-1]!=3 or com_errors.shape!=(pose_angles.shape[0],3) or ((pose_angles<0)|(pose_angles>math.pi)).any() or any(x.dtype!=pose_angles.dtype or x.device!=pose_angles.device for x in (velocity_errors,end_errors,com_errors)): raise ValueError("invalid motion error components")
    pose=torch.exp(-2*pose_angles.square().sum(dim=-1))
    velocity=torch.exp(-.1*velocity_errors.square().sum(dim=-1).sum(dim=-1))
    end=torch.exp(-40*end_errors.square().sum(dim=-1).sum(dim=-1))
    com=torch.exp(-10*com_errors.square().sum(dim=-1))
    return .65*pose+.10*velocity+.15*end+.10*com
