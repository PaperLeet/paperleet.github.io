"""Dense teaching implementation of LightGCN's normalized graph and uniform layer aggregation."""
import torch
from torch import Tensor



def normalized_bipartite_graph(interactions: Tensor, dtype=None) -> Tensor:
    """Return symmetric D^-1/2 A D^-1/2, users first, without self loops.

    Returns:
        Return one floating Tensor [N,N], N=U+I for interaction shape [U,I], with users before
        items on both axes. Use float32 when dtype is None, otherwise the requested
        float32/float64, on the interaction device. This is the symmetric normalized adjacency,
        not embeddings: no self-loops, isolated rows/columns zero, including empty axes. No
        input mutation or gradients from Boolean history.
    """
    dtype = torch.float32 if dtype is None else dtype
    _check_graph(interactions, dtype)
    raise NotImplementedError('Build the symmetric normalized user-item graph')


def lightgcn_embeddings(initial: Tensor, normalized_graph: Tensor, layers: int = 2) -> Tensor:
    """Average E0 through EK, using only repeated graph multiplication.

    Returns:
        Return one floating Tensor [N,D], preserving initial embedding dtype/device, equal to
        the uniform mean of layers 0 through K=layers (K+1 terms). Depth zero returns initial
        values; N=0 is valid. Keep gradients through both initial embeddings and a
        differentiable graph, not just the final layer. No input mutation or detached output.
    """
    _check_propagation(initial, normalized_graph, layers)
    raise NotImplementedError('Propagate linearly and include the initial embeddings in the average')



# Provided validation: implement only the two functions above.
def _check_graph(interactions, dtype):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when interactions is Boolean [U,I] and dtype is
        explicitly float32 or float64 (the caller resolves None to float32). Raise ValueError
        when a checked condition fails; do not mutate inputs or return transformed tensors.
    """
    if interactions.ndim != 2 or interactions.dtype != torch.bool:
        raise ValueError('interactions must be Boolean [U,I]')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('The graph must use float32 or float64')


def _check_propagation(initial, normalized_graph, layers):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when initial [N,D], D>0, and graph [N,N] are
        finite matching float32/float64 tensors and layers is a nonnegative non-Boolean Python
        integer. Raise ValueError when a checked condition fails; do not mutate inputs or return
        transformed tensors.
    """
    if initial.ndim != 2 or initial.shape[1] < 1:
        raise ValueError('initial must have shape [N,D] with D > 0')
    if normalized_graph.ndim != 2 or normalized_graph.shape != (len(initial), len(initial)):
        raise ValueError('normalized_graph must have shape [N,N]')
    if initial.dtype not in (torch.float32, torch.float64) or normalized_graph.dtype != initial.dtype:
        raise ValueError('Embeddings and graph must share float32/float64 dtype')
    if normalized_graph.device != initial.device:
        raise ValueError('Embeddings and graph must share a device')
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 0:
        raise ValueError('layers must be a nonnegative Python integer')
    if not bool(torch.isfinite(initial).all()) or not bool(torch.isfinite(normalized_graph).all()):
        raise ValueError('Embeddings and graph must be finite')
