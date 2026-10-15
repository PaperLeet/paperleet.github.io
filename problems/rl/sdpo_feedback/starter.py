import torch


def select_feedback(question_ids, success, rich_feedback):
    """Return solution index (-1 if absent) and usable-feedback availability.

    Returns:
        (solution_indices, feedback_available), both [N] on question_ids.device.
        solution_indices: long rows, own successful row else first same-question success,
        else -1. feedback_available: bool, solution exists OR rich feedback. No gradients.
        Available rich feedback may coexist with solution index -1.
    """
    raise NotImplementedError("Implement select_feedback")


def feedback_reverse_kl(student_logits, teacher_logits, token_mask, feedback_available):
    """Average reverse KL over tokens of attempts with usable teacher feedback.

    Returns:
        Return one floating scalar Tensor loss, shape (), matching student dtype/device. For
        [N, L, V] logits, sum reverse KL over vocabulary V and average tokens where
        token_mask & feedback_available[:, None] is true. Keep student gradients, freeze
        teacher logits. If no feedback tokens are eligible, return differentiable zero with
        zero student gradient.
    """
    raise NotImplementedError("Implement feedback_reverse_kl")
