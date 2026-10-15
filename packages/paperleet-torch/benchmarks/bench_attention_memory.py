"""CPython retained/peak SDPA allocations, measured separately from timings.

NumPy allocations registered with tracemalloc are included; this is not RSS or
a browser-memory measurement. Construct operands before tracing each case.
"""
import gc
import json
import tracemalloc

import numpy as np
from paperleet_torch import torch


def measure_attention_memory(dtype):
    operands = [torch.tensor(np.ones((2, 4, 64, 32), dtype=dtype), requires_grad=True)
                for _ in range(3)]
    mask = torch.tensor(np.zeros((64, 64), dtype=dtype), requires_grad=True)
    gc.collect()
    tracemalloc.start()
    try:
        output = torch.nn.functional.scaled_dot_product_attention(*operands, attn_mask=mask)
        retained, forward_peak = tracemalloc.get_traced_memory()
        output.sum().backward()
        _, backward_peak = tracemalloc.get_traced_memory()
        return dict(output_dtype=str(output.dtype), forward_retained_bytes=retained,
                    forward_peak_bytes=forward_peak, backward_peak_bytes=backward_peak)
    finally:
        tracemalloc.stop()


if __name__ == '__main__':
    print(json.dumps({str(np.dtype(dtype)): measure_attention_memory(dtype)
                      for dtype in (np.float32, np.float64)}, indent=2))
