import torch


def align_rollout_logits(student_logits, teacher_logits, student_prompt_lengths,
                         teacher_prompt_lengths, response_lengths):
    """Select next-token predictions on the SAME rollout under two contexts.

    Returns:
        Return the Python tuple `(student, teacher, mask)` in that order. The floating
        student/teacher tensors are `[B,T,V]`, with batch size B, vocabulary size V and
        T=max(response_lengths); each preserves its source logits' dtype/device. mask is a
        boolean Tensor `[B,T]` on the same device, True for valid response positions. Padded
        logits are exactly zero; all-empty responses yield `[B,0,V]`, `[B,0,V]`, `[B,0]`.
        Keep gradients only to selected student positions, detach teacher values, and do not
        mutate inputs.
    """
    raise NotImplementedError("Implement this function")
