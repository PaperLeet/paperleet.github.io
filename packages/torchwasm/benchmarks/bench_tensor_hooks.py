"""Tensor callback dispatch, retention and registration costs on CPU/Pyodide."""
import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np
import torchwasm
from torchwasm import torch


def run_benchmarks(repeat=11, iterations=1200, warmup=3, dtype='float32', cases=None):
    if dtype not in ('float32', 'float64') or min(repeat, iterations) < 1 or warmup < 0:
        raise ValueError('Use float32/float64 and positive sample counts')
    kind = getattr(torch, dtype)
    def leaf_case(hook=None, post=None):
        value = torch.ones(16, dtype=kind, requires_grad=True)
        seed = torch.ones_like(value)
        if hook is not None: value.register_hook(hook)
        if post is not None: value.register_post_accumulate_grad_hook(post)
        def run():
            value.grad = None
            value.backward(seed)
        return run
    def graph_case(retain=False, hook=None):
        x = torch.ones(16, 32, dtype=kind, requires_grad=True)
        def run():
            x.grad = None
            middle = x.square()
            if retain: middle.retain_grad()
            if hook is not None: middle.register_hook(hook)
            middle.square().mean().backward()
        return run
    available = {
        'leaf_backward': leaf_case(),
        'graph_backward': graph_case(),
        'retained_backward': graph_case(retain=True),
    }
    if hasattr(torch.Tensor, 'register_hook'):
        unchanged = lambda gradient: None
        available.update({
            'leaf_observer': leaf_case(hook=unchanged),
            'leaf_transform': leaf_case(hook=lambda gradient: gradient * .5),
            'leaf_post_accumulate': leaf_case(post=lambda parameter: None),
            'leaf_pre_post': leaf_case(hook=unchanged, post=lambda parameter: None),
            'graph_intermediate_observer': graph_case(hook=unchanged),
            'graph_retained_observer': graph_case(retain=True, hook=unchanged),
        })
        temporary = torch.tensor(2., dtype=kind, requires_grad=True)
        def register_remove(): temporary.register_hook(unchanged).remove()
        available['register_remove'] = register_remove
    if hasattr(torch.Tensor, 'copy_'):
        source = torch.ones(64, 128, dtype=kind)
        destination = torch.zeros_like(source)
        strided = torch.zeros(128, 64, dtype=kind).T
        available['copy_contiguous'] = lambda: destination.copy_(source)
        available['copy_strided'] = lambda: strided.copy_(source)
    selected = list(available) if cases is None else list(cases)
    if not selected or len(set(selected)) != len(selected) or any(name not in available for name in selected):
        raise ValueError('Select unique available cases: ' + ', '.join(available))
    results = {}
    for name in selected:
        run = available[name]
        inner = iterations * (10 if name == 'register_remove' else 1)
        for _ in range(warmup): run()
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(inner): run()
            samples.append((time.perf_counter() - start) * 1000 / inner)
        results[name] = dict(median_ms=statistics.median(samples), samples_ms=samples, iterations=inner)
    return dict(library_version=torchwasm.__version__, python=platform.python_version(),
                numpy=np.__version__, platform=platform.platform(), dtype=dtype,
                implementation=sys.modules[torch.Tensor.__module__].__file__,
                repeat=repeat, iterations=iterations, warmup=warmup, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=11)
    parser.add_argument('--iterations', type=int, default=1200)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--case', action='append', dest='cases')
    parser.add_argument('--output')
    options = parser.parse_args()
    report = json.dumps(run_benchmarks(options.repeat, options.iterations, dtype=options.dtype, cases=options.cases), indent=2) + '\n'
    if options.output:
        with open(options.output, 'w', encoding='utf8') as stream: stream.write(report)
    print(report)
