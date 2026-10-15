"""Dense teaching implementation of LightGCN's normalized graph and uniform layer aggregation."""
import torch
from torch import Tensor



def normalized_bipartite_graph(interactions: Tensor, dtype=None) -> Tensor:
    """Return symmetric D^-1/2 A D^-1/2, users first, without self loops."""
    dtype = torch.float32 if dtype is None else dtype
    _check_graph(interactions, dtype)
    raise NotImplementedError('Build the symmetric normalized user-item graph')


def lightgcn_embeddings(initial: Tensor, normalized_graph: Tensor, layers: int = 2) -> Tensor:
    """Average E0 through EK, using only repeated graph multiplication."""
    _check_propagation(initial, normalized_graph, layers)
    raise NotImplementedError('Propagate linearly and include the initial embeddings in the average')



# Provided validation: implement only the two functions above.
def _check_graph(interactions, dtype):
    if interactions.ndim != 2 or interactions.dtype != torch.bool:
        raise ValueError('interactions must be Boolean [U,I]')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('The graph must use float32 or float64')


def _check_propagation(initial, normalized_graph, layers):
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
