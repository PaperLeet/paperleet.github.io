# Bug: 把样本均值替换为微批次数归一化
import torch
from torch import Tensor

def pipeline_schedule(stages: int, microbatches: int) -> list[tuple]:
    if any(type(v) is not int or v <= 0 for v in (stages, microbatches)):
        raise ValueError('positive integer stages and microbatches required')
    events = []
    for m in range(microbatches):
        for s in range(stages):
            events.append(('F', s, m, s + m))
            events.append(('B', s, m, stages + microbatches - 1 + stages - 1 - s + microbatches - 1 - m))
    return sorted(events, key=lambda e: (e[3], e[1], e[0]))

def microbatch_mse(x: Tensor, y: Tensor, weight: Tensor, chunk_size: int) -> Tensor:
    if type(chunk_size) is not int or chunk_size <= 0 or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError('nonempty matrix and positive chunk size required')
    if y.shape != (x.shape[0],) or weight.shape != (x.shape[1],) or any(t.dtype != x.dtype or t.device != x.device for t in (y, weight)):
        raise ValueError('incompatible regression tensors')
    total = x.new_zeros(())
    for start in range(0, x.shape[0], chunk_size):
        error = x[start:start+chunk_size] @ weight - y[start:start+chunk_size]
        total = total + (error * error).sum()
    return total / ((x.shape[0] + chunk_size - 1) // chunk_size)
