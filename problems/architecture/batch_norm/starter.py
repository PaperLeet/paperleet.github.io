import math
import torch
from torch import Tensor


def batch_norm(x: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor,
               running_var: Tensor, training: bool = True, momentum: float = .1,
               eps: float = 1e-5) -> tuple[Tensor, Tensor, Tensor]:
    """Return (y,next_mean,next_var), with y shaped like x [N,C,*spatial]
    and estimates [C]. weight, bias and buffers are [C], with matching float32/64
    dtype/device. Reduce all axes except C. In training require m>=2, normalize
    with population variance and update detached estimates by (1-momentum)*old
    + momentum*batch; running variance uses the unbiased m/(m-1) correction.
    In inference use detached supplied moments and return unchanged estimates;
    N=0 is allowed. Keep statistics differentiable only on the training output
    path. Never mutate inputs. Finite eps>0 goes inside sqrt; momentum in [0,1].
    Invalid shape, training sample count, epsilon or momentum raises ValueError."""
    raise NotImplementedError
