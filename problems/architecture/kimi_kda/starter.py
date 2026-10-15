import torch
from torch import Tensor


def bounded_decay(logits: Tensor, log_scale: Tensor, lower_bound: float = -5.) -> Tensor:
    """Required: K3 Eq.5 retention exp(lower_bound * sigmoid(exp(A) * z)).
    logits [B,T,H,K], log_scale [H], lower_bound < 0; preserve autograd/dtype/device.

    Returns:
        Return one floating retention Tensor `[B,T,H,K]`, matching logits: B batch size, T
        sequence length, H head count, K key channels. Values lie between exp(lower_bound)
        and 1; return retention factors, not their logarithms. Preserve logits' dtype/device
        and gradients into logits and log_scale; do not change inputs.
    """
    raise NotImplementedError()


def recurrent_kda(q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor,
                  initial_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Required: decay state, correct its key prediction, then read the updated state.
    q/k are prepared vectors [B,T,H,K], v [B,T,H,V], alpha [B,T,H,K], beta [B,T,H].
    initial_state [B,H,K,V], default zeros. Return output [B,T,H,V], final [B,H,K,V].
    alpha and beta are retention/write factors, not logits; endpoints 0/1 are supported.
    No projections, q scaling, q/k normalization, output gate or RMSNorm here.
    Support T=0, K!=V, noncontiguous inputs, gradients and continuation from initial_state.
    Preserve all input tensors; the tiny experiments and projection scaffolding are supplied.

    Returns:
        Return `(output, final_state)` as floating tensors `[B,T,H,V]` and `[B,H,K,V]`. B is
        batch size, T token count, H head count, K key channels and V value channels (K and
        V may differ). Each output reads the state after that token's update; final_state
        can resume the scan. With T=0, output is `[B,0,H,V]` and final_state equals the
        supplied initial state, or zeros when it is None. Preserve shared dtype/device and
        gradient paths through nonempty scans to q/k/v/alpha/beta/initial_state; do not
        detach the returned state or mutate inputs.
    """
    raise NotImplementedError()
