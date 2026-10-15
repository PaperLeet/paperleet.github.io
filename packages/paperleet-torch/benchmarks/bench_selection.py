"""Same-dtype selection/join costs with independent value/gradient checks."""
import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np
from paperleet_torch import torch, __version__


def run_benchmarks(repeat=11, iterations=120, warmup=3, dtype='float32', cases=None):
    kind = np.dtype(dtype)
    if dtype not in ('float32', 'float64') or min(repeat, iterations) < 1 or warmup < 0:
        raise ValueError('Use float32/float64 and positive sample counts')
    raw_a = np.full((64, 128), .25, dtype=kind)
    raw_b = np.full(raw_a.shape, .75, dtype=kind)
    raw_mask = np.arange(raw_a.size).reshape(raw_a.shape) % 3 != 0
    mask = torch.tensor(raw_mask)
    a, b = torch.tensor(raw_a), torch.tensor(raw_b)
    left, right = torch.tensor(raw_a, requires_grad=True), torch.tensor(raw_b, requires_grad=True)
    root = torch.tensor(raw_a, requires_grad=True)

    def where_backward():
        left.grad = right.grad = None
        loss = torch.where(mask, left, right).square().mean()
        loss.backward()
        return loss

    def join_backward(stack=False):
        left.grad = right.grad = None
        value = torch.stack([left, right], dim=0) if stack else torch.cat([left, right], dim=1)
        loss = value.square().mean(); loss.backward()
        return loss

    def where_chain_backward():
        root.grad = None
        value = root
        for _ in range(16): value = torch.where(mask, value + .125, -value)
        loss = value.square().mean(); loss.backward()
        return loss

    available = {
        'where_forward': lambda: torch.where(mask, a, b),
        'where_float_scalar': lambda: torch.where(mask, a, .125),
        'cat_forward': lambda: torch.cat([a, b], dim=1),
        'stack_forward': lambda: torch.stack([a, b], dim=0),
        'where_backward': where_backward,
        'cat_backward': join_backward,
        'stack_backward': lambda: join_backward(True),
        'where_chain_backward': where_chain_backward,
    }
    selected = list(available) if cases is None else list(cases)
    if not selected or len(set(selected)) != len(selected) or any(name not in available for name in selected):
        raise ValueError('Choose unique selection/join cases')
    expected = {
        'where_forward': np.where(raw_mask, raw_a, raw_b),
        'where_float_scalar': np.where(raw_mask, raw_a, np.array(.125, dtype=kind)),
        'cat_forward': np.concatenate([raw_a, raw_b], axis=1),
        'stack_forward': np.stack([raw_a, raw_b], axis=0),
    }
    results, contracts = {}, {}
    for name in selected:
        run = available[name]
        value = run()
        assert value.dtype == kind, (name, value.dtype)
        contracts[name] = dict(dtype=str(value.dtype), shape=list(value.shape))
        if name in expected:
            np.testing.assert_array_equal(value.numpy(), expected[name])
        elif name == 'where_backward':
            scale = np.asarray(2 / raw_a.size, dtype=kind)
            np.testing.assert_allclose(left.grad.numpy(), np.where(raw_mask, scale * raw_a, 0), rtol=1e-6)
            np.testing.assert_allclose(right.grad.numpy(), np.where(raw_mask, 0, scale * raw_b), rtol=1e-6)
        elif name == 'where_chain_backward':
            final = np.where(raw_mask, raw_a + np.array(2., dtype=kind), raw_a)
            np.testing.assert_allclose(root.grad.numpy(), final * np.asarray(2 / raw_a.size, dtype=kind), rtol=1e-6)
        else:
            np.testing.assert_allclose(left.grad.numpy(), raw_a / raw_a.size, rtol=1e-6)
            np.testing.assert_allclose(right.grad.numpy(), raw_b / raw_a.size, rtol=1e-6)
        del value
        inner = max(1, iterations // 4) if name == 'where_chain_backward' else iterations
        for _ in range(warmup): run()
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(inner): run()
            samples.append((time.perf_counter() - start) * 1000 / inner)
        results[name] = dict(median_ms=statistics.median(samples), samples_ms=samples, iterations=inner)
    return dict(library_version=__version__, python=platform.python_version(), numpy=np.__version__,
                platform=platform.platform(), implementation=sys.modules[torch.Tensor.__module__].__file__,
                dtype=dtype, shape=[64, 128], chain_depth=16, repeat=repeat, warmup=warmup,
                iterations=iterations, contracts=contracts, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=11)
    parser.add_argument('--iterations', type=int, default=120)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--case', action='append', dest='cases')
    parser.add_argument('--output')
    options = parser.parse_args()
    raw = json.dumps(run_benchmarks(options.repeat, options.iterations, dtype=options.dtype, cases=options.cases), indent=2) + '\n'
    if options.output:
        with open(options.output, 'w', encoding='utf8') as stream: stream.write(raw)
    print(raw)
