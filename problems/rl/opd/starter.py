import torch
from torch import Tensor

def sample_student(transition_logits: Tensor, batch_size: int, length: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Fixed-length Markov language model: row V is BOS, rows 0..V-1 are tokens.

    Returns:
        Return (tokens, prefixes), a tuple of two torch.long Tensors of shape [batch_size,
        length], on transition_logits.device, with no gradients. tokens contains sampled IDs
        in [0, V-1]; prefixes[:, 0] is the BOS ID V and prefixes[:, 1:] == tokens[:, :-1].
        Sampling uses the supplied generator. Do not return probabilities or
        log-probabilities.
    """
    raise NotImplementedError('Implement sample_student')

def distillation_loss(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor, direction: str='reverse') -> Tensor:
    """Full-vocabulary KL at student-generated prefixes; valid-token average.

    Returns:
        Return one floating scalar Tensor loss, shape (), for logits [B, L, V]. Sum KL
        across the vocabulary V, then average only valid tokens selected by [B, L] mask.
        reverse means KL(student || teacher); forward means KL(teacher || student). Keep
        student gradients, freeze teacher logits; empty masks produce graph-connected zero.
        Preserve the input floating dtype/device.
    """
    raise NotImplementedError('Implement distillation_loss')
