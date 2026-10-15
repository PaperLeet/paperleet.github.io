import torch


def topk_tail_reverse_kl(student_logits, teacher_logits, k):
    """Coarsen the vocabulary into student top-k tokens and one complementary bin.

    Returns:
        Return one floating Tensor divergence, shape student_logits.shape[:-1]: remove only
        the vocabulary axis from [..., V]. Thus [B, L, V] returns [B, L], and [V] returns a
        scalar (). Each value is reverse KL for student top-k tokens plus the complementary
        tail bin (k=V gives full KL). Preserve student dtype/device and gradients; freeze
        teacher logits. Do not apply batch/token mean reduction.
    """
    raise NotImplementedError("Implement this function")
