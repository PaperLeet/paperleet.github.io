"""Complete Tensor construction/import costs, including allocation and cleanup."""
import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np
import paperleet_torch
from paperleet_torch import torch


def run_benchmarks(repeat=11, iterations=1200, warmup=3, dtype='float32', cases=None):
    if dtype not in ('float32', 'float64') or min(repeat, iterations) < 1 or warmup < 0:
        raise ValueError('Use float32/float64 and positive sample counts')
    kind = np.dtype(dtype)
    source = np.full((64, 128), 1.25, dtype=kind)
    transposed = source.T
    other = np.full(source.shape, 1.25, dtype=np.float64 if dtype == 'float32' else np.float32)
    existing = torch.tensor(source)
    scalar = kind.type(1.25)
    values = [1.25, 2.5, 3.75]
    available = {
        'builtin_float': lambda: torch.tensor(1.25),
        'builtin_int': lambda: torch.tensor(7),
        'builtin_bool': lambda: torch.tensor(True),
        'python_list': lambda: torch.tensor(values),
        'numpy_scalar': lambda: torch.tensor(scalar),
        'numpy_copy': lambda: torch.tensor(source),
        'numpy_share': lambda: torch.from_numpy(source),
        'numpy_transposed_share': lambda: torch.as_tensor(transposed),
        'numpy_cast': lambda: torch.as_tensor(other, dtype=kind),
        'tensor_copy': lambda: torch.tensor(existing),
        'new_tensor': lambda: existing.new_tensor(source),
        'existing_identity': lambda: torch.as_tensor(existing),
        'numpy_export': lambda: existing.numpy(),
        'numpy_array_protocol': lambda: np.asarray(existing),
    }
    selected = list(available) if cases is None else list(cases)
    if not selected or len(set(selected)) != len(selected) or any(name not in available for name in selected):
        raise ValueError('Select unique available cases: ' + ', '.join(available))
    assert torch.from_numpy(source).data_ptr() == source.__array_interface__['data'][0]
    assert torch.as_tensor(existing) is existing
    assert torch.tensor(source).data_ptr() != source.__array_interface__['data'][0]
    results = {}
    for name in selected:
        run = available[name]
        multiplier = 100 if name in ('numpy_export', 'numpy_array_protocol') else 10 if name in ('builtin_float', 'builtin_int', 'builtin_bool', 'python_list', 'numpy_scalar', 'existing_identity') else 1
        inner = iterations * multiplier
        for _ in range(warmup): run()
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(inner): run()
            samples.append((time.perf_counter() - start) * 1000 / inner)
        results[name] = dict(median_ms=statistics.median(samples), samples_ms=samples, iterations=inner)
    return dict(library_version=paperleet_torch.__version__, python=platform.python_version(),
                numpy=np.__version__, platform=platform.platform(), dtype=dtype,
                implementation=sys.modules[torch.Tensor.__module__].__file__,
                default_scalar_dtype='float32/int64/bool regardless of array fixture dtype',
                repeat=repeat, iterations=iterations, warmup=warmup, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=11)
    parser.add_argument('--iterations', type=int, default=1200)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--case', action='append', dest='cases')
    parser.add_argument('--output')
    options = parser.parse_args()
    raw = json.dumps(run_benchmarks(options.repeat, options.iterations, dtype=options.dtype, cases=options.cases), indent=2) + '\n'
    if options.output:
        with open(options.output, 'w', encoding='utf8') as stream: stream.write(raw)
    print(raw)
