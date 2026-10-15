import math
import torch
from torch import Tensor


def _validate(e: Tensor, q: Tensor, eps: float):
    """Provided input validation.

    Returns:
        Provided helper: return (e_norm, q_norm), two floating Tensors of shape [...,1],
        reducing only the final D features of e/q [...,D] and retaining the dimension. Preserve
        dtype/device and their norm gradient paths; do not mutate inputs. Raise ValueError for
        invalid shape/epsilon, near-zero norms or antiparallel pairs. This returns norms, not
        None.
    """
    if e.shape != q.shape or e.ndim < 1 or e.shape[-1] < 2 or e.numel() == 0:
        raise ValueError("e and q must have equal nonempty [..., D] shapes with D >= 2")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    en, qn = e.norm(dim=-1, keepdim=True), q.norm(dim=-1, keepdim=True)
    if (en <= eps).any() or (qn <= eps).any():
        raise ValueError("zero or near-zero vectors are outside this exercise")
    if ((e/en + q/qn).norm(dim=-1) <= eps).any():
        raise ValueError("antiparallel pairs do not define this rotation")
    return en, qn


def rotation_quantize(e: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """Forward q; backward a frozen norm-scaled rotation applied to e.

    Returns:
        Return one floating Tensor with the same [...,D] shape, dtype and device as e and q
        (D>=2); its forward values equal q. Keep the surrogate gradient to e through the frozen
        scaled rotation, but no gradient to q or the rotation/scale construction. Do not detach
        the output, return Jacobians or mutate either input.
    """
    en, qn = _validate(e, q, eps)
    raise NotImplementedError("Freeze the rotation and norm ratio, then transform e")
