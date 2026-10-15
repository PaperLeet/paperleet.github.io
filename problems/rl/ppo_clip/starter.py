import torch
from torch import Tensor


# Required: implement the core functions below.
def clipped_surrogate(new_logp: Tensor, old_logp: Tensor, advantages: Tensor,
                      clip_eps: float = 0.2) -> Tensor:
    """Minimized PPO policy loss for [B] log-probs and already prepared advantages.

    Freeze the old policy and advantages; keep gradients through new_logp. Return one
    scalar. Advantage normalization is supplied by ppo_loss.

    Returns:
        Return one floating scalar Tensor loss, shape (): the negative batch mean of the
        clipped policy surrogate for [B] inputs. Preserve input floating dtype/device.
        Retain gradients through new_logp; freeze old_logp and advantages. Return the Tensor
        itself, without .item() or .detach().
    """
    raise NotImplementedError()


# Provided: keep these helpers and call them as needed.
def ppo_loss(new_logp: Tensor, old_logp: Tensor, advantages: Tensor, new_values: Tensor,
             old_values: Tensor, returns: Tensor, entropy: Tensor | None = None,
             clip_eps: float = 0.2, vf_clip: float | None = 0.2, vf_coef: float = 0.5,
             ent_coef: float = 0.0, norm_adv: bool = True) -> dict[str, Tensor]:
    """Provided training adapter: normalization, value/entropy terms, diagnostics.

    Returns:
        dict[str, Tensor] with scalar () values: loss (total objective), pg_loss, v_loss
        (half mean value error), entropy (mean or zero), approx_kl and clipfrac. Only
        approx_kl/clipfrac are detached diagnostics. Current policy/value loss paths stay
        live; old policy/values and advantage/return targets are frozen.
    """
    old_values = old_values.detach()
    advantages, returns = advantages.detach(), returns.detach()
    if norm_adv:
        std = advantages.std(unbiased=True) if advantages.numel() > 1 else advantages.new_zeros(())
        advantages = (advantages - advantages.mean()) / (std + 1e-8)
    pg_loss = clipped_surrogate(new_logp, old_logp, advantages, clip_eps)
    value_error = (new_values - returns).square()
    if vf_clip is not None:
        v_clipped = old_values + (new_values - old_values).clamp(-vf_clip, vf_clip)
        value_error = torch.maximum(value_error, (v_clipped - returns).square())
    v_loss = 0.5 * value_error.mean()
    ent = new_logp.new_zeros(()) if entropy is None else entropy.mean()
    with torch.no_grad():
        logratio = new_logp - old_logp.detach()
        ratio = logratio.exp()
        approx_kl = (ratio - 1 - logratio).mean()
        clipfrac = ((ratio - 1).abs() > clip_eps).to(new_logp.dtype).mean()
    return dict(loss=pg_loss + vf_coef * v_loss - ent_coef * ent, pg_loss=pg_loss,
                v_loss=v_loss, entropy=ent, approx_kl=approx_kl, clipfrac=clipfrac)
