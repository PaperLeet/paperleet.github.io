"""Public operator dispatch with identical numerical work and direct controls."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torchwasm
from torchwasm import torch


def run_benchmarks(*, dtype='float32', repeat=11, iterations=2000, cases=None):
    if cases is not None: raise ValueError('Dispatch controls use all twelve fixed cases')
    dt = np.dtype(dtype)
    raw = np.linspace(-1, 1, 16, dtype=dt)
    x = torch.from_numpy(raw.copy())
    matrix = torch.from_numpy(raw.reshape(4, 4).copy())
    leaf = torch.from_numpy(raw.copy()).requires_grad_()
    def reverse():
        leaf.grad = None
        torch.sum(torch.exp(leaf)).backward()
    def method_reverse():
        leaf.grad = None
        leaf.exp().sum().backward()
    workloads = {
        'exp_positional': lambda: torch.exp(x),
        'exp_keyword': lambda: torch.exp(input=x),
        'exp_optional_none': lambda: torch.exp(x, out=None),
        'sum_keywords': lambda: torch.sum(x, dim=0, keepdim=True),
        'var_mean_keywords': lambda: torch.var_mean(x, dim=-1, correction=1),
        'add_positional': lambda: torch.add(x, 2),
        'matmul_positional': lambda: torch.matmul(matrix, matrix),
        'cat_nested': lambda: torch.cat([x, x], dim=0),
        'functional_reverse': reverse,
        'method_exp_control': lambda: x.exp(),
        'method_sum_control': lambda: x.sum(0, True),
        'method_reverse_control': method_reverse,
    }
    expected = np.exp(raw)
    for name in ('exp_positional', 'exp_keyword', 'exp_optional_none', 'method_exp_control'):
        np.testing.assert_allclose(workloads[name]().numpy(), expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(workloads['sum_keywords']().numpy(), raw.sum(keepdims=True), atol=1e-6)
    variance, mean = workloads['var_mean_keywords']()
    np.testing.assert_allclose(variance.numpy(), raw.var(ddof=1), rtol=2e-6)
    np.testing.assert_allclose(mean.numpy(), raw.mean(), atol=1e-6)
    np.testing.assert_array_equal(workloads['add_positional']().numpy(), raw + 2)
    np.testing.assert_allclose(workloads['matmul_positional']().numpy(), raw.reshape(4, 4) @ raw.reshape(4, 4), atol=1e-6)
    np.testing.assert_array_equal(workloads['cat_nested']().numpy(), np.concatenate([raw, raw]))
    for fn in (reverse, method_reverse):
        fn(); np.testing.assert_allclose(leaf.grad.numpy(), expected, rtol=2e-6)
    results = {}
    for name, function in workloads.items():
        count = max(1, iterations // 5) if 'reverse' in name else iterations
        for _ in range(3): function()
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(count): function()
            samples.append((time.perf_counter() - start) * 1000 / count)
        results[name] = dict(iterations=count, median_ms=statistics.median(samples), samples_ms=samples)
    return dict(library=torchwasm.__version__, runtime_module=str(Path(torchwasm.__file__).with_name('_registry.py')),
                python=platform.python_version(), numpy=np.__version__, dtype=dtype, repeat=repeat,
                iterations=iterations, warmup=3, scope='Twelve fixed steady-state public-call and direct-method workloads; identical numerical work, independent NumPy values/gradients. Excludes cold-start, native Torch and application speed claims.', results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--repeat', type=int, default=11)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--output')
    options = parser.parse_args()
    raw = json.dumps(run_benchmarks(dtype=options.dtype, repeat=options.repeat, iterations=options.iterations), indent=2) + '\n'
    if options.output: Path(options.output).write_text(raw)
    print(raw, end='')
