"""Repeatable CPU/WASM workloads; each timed call includes forward and backward.

Run with ``python benchmarks/bench_runtime.py --output results.json`` from this
subproject. ``run_benchmarks`` is also callable inside Pyodide. Timings exclude
imports, fixture construction and result serialization, and never run in CI as
a speed assertion. Compare identical environments and workload configuration.
"""
import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np
from stridetorch import __version__, torch


def workloads(dtype='float32'):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Benchmark dtype must be float32 or float64')
    rng = np.random.default_rng(412)
    np.random.seed(412)  # Module initialization uses the library's global RNG.

    def tensor(shape, requires_grad=True):
        return torch.tensor(rng.normal(size=shape).astype(dtype), requires_grad=requires_grad)

    x = tensor((64, 128))
    weight = tensor((128, 128))

    def linear_backward():
        x.grad = weight.grad = None
        torch.nn.functional.linear(x, weight).square().mean().backward()

    query, key, value = [tensor((2, 4, 64, 32)) for _ in range(3)]
    mask = tensor((64, 64))

    def attention_backward():
        for item in (query, key, value, mask):
            item.grad = None
        torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask).square().mean().backward()

    sliced = tensor((256, 256))

    def slice_backward():
        sliced.grad = None
        sliced[::2, 16:240].square().sum().backward()

    indexed = tensor((256, 64))
    indices = torch.tensor(rng.integers(0, 256, size=256), dtype=torch.int64)

    def indexed_backward():
        indexed.grad = None
        indexed[indices].square().sum().backward()

    parameter = torch.nn.Parameter(tensor((256, 256), requires_grad=False))
    parameter.grad = tensor(parameter.shape, requires_grad=False)
    optimizer = torch.optim.Adam([parameter], lr=.001)

    def adam_step():
        optimizer.step()

    sgd_parameter = torch.nn.Parameter(parameter.detach().clone())
    adamw_parameter = torch.nn.Parameter(parameter.detach().clone())
    sgd_parameter.grad = parameter.grad.clone()
    adamw_parameter.grad = parameter.grad.clone()
    sgd_optimizer = torch.optim.SGD([sgd_parameter], lr=.001, momentum=.9)
    adamw_optimizer = torch.optim.AdamW([adamw_parameter], lr=.001)
    inactive = [torch.nn.Parameter(torch.ones(64, 64, dtype=dtype)) for _ in range(8)]

    def adam_construct():
        return torch.optim.Adam(inactive)

    def sgd_step():
        sgd_optimizer.step()

    def adamw_step():
        adamw_optimizer.step()

    def adamw_zero_grad():
        adamw_optimizer.zero_grad(set_to_none=False)

    def same_dtype_to():
        return x.to(dtype=dtype)

    def cpu_noop():
        return x.cpu()

    def contiguous_noop():
        return x.contiguous()

    transposed = x.T

    def contiguous_copy():
        return transposed.contiguous()

    def model_construct():
        return torch.nn.Sequential(torch.nn.Linear(16, 32, dtype=dtype), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 8, dtype=dtype))

    model = torch.nn.Sequential(torch.nn.Linear(16, 32, dtype=dtype), torch.nn.ReLU(),
                                torch.nn.Linear(32, 8, dtype=dtype))
    model_input = tensor((16, 16), requires_grad=False)
    model_optimizer = torch.optim.SGD(model.parameters(), lr=.001)

    def model_training_step():
        model_optimizer.zero_grad()
        model(model_input).square().mean().backward()
        model_optimizer.step()

    def model_parameter_walk():
        return list(model.parameters())

    cases = dict(linear_backward=linear_backward, attention_backward=attention_backward,
                 slice_backward=slice_backward, indexed_backward=indexed_backward,
                 adam_step=adam_step, sgd_step=sgd_step, adamw_step=adamw_step,
                 adamw_zero_grad=adamw_zero_grad, adam_construct=adam_construct,
                 same_dtype_to=same_dtype_to, cpu_noop=cpu_noop,
                 contiguous_noop=contiguous_noop, contiguous_copy=contiguous_copy,
                 model_construct=model_construct,
                 model_training_step=model_training_step, model_parameter_walk=model_parameter_walk)
    for m, k, n in [(32, 32, 32), (64, 128, 128), (128, 128, 128),
                    (256, 256, 256), (512, 512, 512), (37, 53, 29)]:
        a = torch.tensor(((np.arange(m*k) % 17 - 8) / 9).astype(dtype).reshape(m, k))
        b = torch.tensor(((np.arange(k*n) % 13 - 6) / 7).astype(dtype).reshape(k, n))
        cases[f'matmul_{m}_{k}_{n}'] = lambda a=a, b=b: torch.matmul(a, b)
    return cases


def run_benchmarks(*, repeat=7, iterations=10, warmup=3, dtype='float32', cases=None):
    if repeat < 1 or iterations < 1 or warmup < 0:
        raise ValueError('repeat/iterations must be positive; warmup must be nonnegative')
    available = workloads(dtype)
    names = list(available) if cases is None else list(cases)
    if not names or len(names) != len(set(names)) or any(name not in available for name in names):
        raise ValueError('Select nonempty, unique benchmark cases from: ' + ', '.join(available))
    results = {}
    for name in names:
        operation = available[name]
        for _ in range(warmup):
            operation()
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(iterations):
                operation()
            samples.append((time.perf_counter() - start) * 1000 / iterations)
        results[name] = dict(median_ms=statistics.median(samples), samples_ms=samples)
    return dict(python=platform.python_version(), platform=platform.platform(),
                library_version=__version__,
                implementation=sys.modules[torch.Tensor.__module__].__file__,
                numpy=np.__version__, dtype=np.dtype(dtype).name, seed=412, repeat=repeat,
                iterations=iterations, warmup=warmup, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--case', action='append', dest='cases', help='select a case; may be repeated')
    options = parser.parse_args()
    report = json.dumps(run_benchmarks(repeat=options.repeat, iterations=options.iterations,
                                     dtype=options.dtype, cases=options.cases), indent=2)
    if options.output:
        with open(options.output, 'w', encoding='utf-8') as stream:
            stream.write(report + '\n')
    print(report)
