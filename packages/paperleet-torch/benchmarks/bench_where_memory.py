"""Retained-loss allocation evidence for mask-only where pullbacks, not RSS."""
import argparse
import gc
import json
import platform
import sys
import tracemalloc

import numpy as np
from paperleet_torch import torch, __version__


def run_benchmarks(dtype='float32', depth=16, shape=(64, 128)):
    if dtype not in ('float32', 'float64') or depth < 1 or min(shape) < 1:
        raise ValueError('Use float32/float64 and positive dimensions')
    kind = np.dtype(dtype)
    raw = np.full(shape, .25, dtype=kind)
    raw_mask = np.arange(raw.size).reshape(shape) % 3 != 0
    expected = np.where(raw_mask, raw + np.array(depth * .125, dtype=kind), raw * (-1)**depth)
    derivative = np.where(raw_mask, 1, (-1)**depth).astype(kind)
    expected_loss = np.mean(expected**2)
    expected_gradient = expected * derivative * np.asarray(2 / raw.size, dtype=kind)
    results = []
    for retain in [False, True]:
        root = torch.tensor(raw, requires_grad=True); mask = torch.tensor(raw_mask)
        gc.collect(); tracemalloc.start()
        try:
            value = root
            for _ in range(depth):
                left, right = value + .125, -value
                value = torch.where(mask, left, right)
            loss = value.square().mean()
            del value, left, right; gc.collect()
            forward, forward_peak = tracemalloc.get_traced_memory()
            loss.backward(retain_graph=retain); gc.collect()
            after, backward_peak = tracemalloc.get_traced_memory()
            final = None
            if retain:
                loss.backward(); gc.collect(); final = tracemalloc.get_traced_memory()[0]
            results.append(dict(retain_graph=retain, loss=loss.item(), gradient_max=float(root.grad.numpy().max()),
                                forward_retained_bytes=forward, forward_peak_bytes=forward_peak,
                                after_backward_retained_bytes=after, backward_peak_bytes=backward_peak,
                                after_final_release_bytes=final))
        finally:
            tracemalloc.stop()
        np.testing.assert_allclose(loss.item(), expected_loss, rtol=1e-6)
        np.testing.assert_allclose(root.grad.numpy(), expected_gradient * (2 if retain else 1), rtol=1e-6)
        del root, mask, loss; gc.collect()
    return dict(library_version=__version__, python=platform.python_version(), numpy=np.__version__,
                implementation=sys.modules[torch.Tensor.__module__].__file__, dtype=dtype,
                depth=depth, shape=list(shape), inputs_and_condition_allocated_before_trace=True, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--output')
    options = parser.parse_args()
    raw = json.dumps(run_benchmarks(options.dtype, options.depth), indent=2) + '\n'
    if options.output:
        with open(options.output, 'w', encoding='utf8') as stream: stream.write(raw)
    print(raw)
