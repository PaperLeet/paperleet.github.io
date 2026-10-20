"""Compare direct 1D pooling with the equivalent 2D composition, on CPU or Pyodide.

Pass two standalone source directories (containing __init__.py), not installed
distribution names. The baseline needs only the existing 2D API. Every flow is
checked before timing; source hashes and all timing samples travel with results.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


def load(name, directory):
    directory = Path(directory)
    spec = importlib.util.spec_from_file_location(name, directory / '__init__.py',
                                                 submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
              if p.suffix == '.py' or p.name in ('LICENSE', 'NOTICE')}
    return module, hashes


def pool(api, kind, direct):
    if kind == 'fixed':
        if direct: return lambda x: api.nn.functional.avg_pool1d(x, 3, 2, 1, True, False)
        return lambda x: api.nn.functional.avg_pool2d(x.unsqueeze(-2), (1, 3), (1, 2), (0, 1), True, False).squeeze(-2)
    count = {'irregular': 7, 'upsample': 67, 'global': 1}[kind]
    if direct: return lambda x: api.nn.functional.adaptive_avg_pool1d(x, count)
    return lambda x: api.nn.functional.adaptive_avg_pool2d(x.unsqueeze(-2), (1, count)).squeeze(-2)


def make_flow(api, kind, mode, direct, dtype):
    rng = np.random.default_rng(4121)
    a = rng.normal(size=(2, 4, 31)).astype(dtype)
    # Keep genuinely strided NumPy storage rather than copying through tensor().
    if mode == 'strided': a = np.repeat(a, 2, axis=-1)[..., ::2]
    x = api.from_numpy(a); function = pool(api, kind, direct)
    if mode == 'backward':
        x.requires_grad_()
        def run():
            x.grad = None; y = function(x); y.square().sum().backward()
            return y, x.grad
    elif mode == 'jvp':
        direction = api.from_numpy(rng.normal(size=a.shape).astype(dtype))
        def run(): return api.func.jvp(function, (x,), (direction,))
    else:
        def run(): return (function(x),)
    return run


def make_training(api, direct, dtype):
    rng = np.random.default_rng(4122)
    class Fixed(api.nn.Module):
        def forward(self, x):
            return api.nn.functional.avg_pool2d(x.unsqueeze(-2), (1, 3), (1, 2), (0, 1), True, False).squeeze(-2)
    class Adaptive(api.nn.Module):
        def forward(self, x): return api.nn.functional.adaptive_avg_pool2d(x.unsqueeze(-2), (1, 3)).squeeze(-2)
    fixed = api.nn.AvgPool1d(3, 2, 1, True, False) if direct else Fixed()
    adaptive = api.nn.AdaptiveAvgPool1d(3) if direct else Adaptive()
    model = api.nn.Sequential(api.nn.Conv1d(2, 4, 3, padding=1), api.nn.SiLU(),
                              fixed, adaptive, api.nn.Flatten(), api.nn.Linear(12, 2))
    model.to(dtype=getattr(api, np.dtype(dtype).name))
    state = {name: api.from_numpy(rng.normal(scale=.1, size=value.shape).astype(dtype))
             for name, value in model.state_dict().items()}
    model.load_state_dict(state)
    x = api.from_numpy(rng.normal(size=(4, 2, 31)).astype(dtype))
    target = api.from_numpy(rng.normal(size=(4, 2)).astype(dtype))
    opt = api.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    def run():
        opt.zero_grad(); output = model(x); loss = (output - target).square().mean()
        loss.backward(); opt.step()
        return loss, *tuple(model.parameters())
    return run


def arrays(values): return [value.detach().numpy().copy() for value in values]


def measure(function, iterations, samples):
    for _ in range(3): function()
    measurements = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(iterations): function()
        measurements.append((time.perf_counter() - start) * 1e6 / iterations)
    return dict(median_us=statistics.median(measurements), samples_us=measurements,
                iterations=iterations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', required=True)
    parser.add_argument('--candidate-source', required=True)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert args.samples > 0 and args.iterations > 0
    baseline, old_hashes = load('pooling_baseline', args.baseline_source)
    candidate, new_hashes = load('pooling_candidate', args.candidate_source)
    variants = [('baseline_composition', baseline.torch, False),
                ('candidate_composition', candidate.torch, False), ('candidate_direct', candidate.torch, True)]
    if args.reverse: variants.reverse()
    results = []
    for dtype in (np.float32, np.float64):
        cases = [(kind, mode) for kind in ('fixed', 'irregular', 'upsample', 'global')
                 for mode in ('forward', 'strided', 'backward', 'jvp')] + [('cnn', 'training')]
        for kind, mode in cases:
            functions = {name: make_training(api, direct, dtype) if mode == 'training'
                         else make_flow(api, kind, mode, direct, dtype) for name, api, direct in variants}
            checked = {name: arrays(function()) for name, function in functions.items()}
            expected = checked['baseline_composition']
            for name, values in checked.items():
                assert len(values) == len(expected)
                for aa, bb in zip(values, expected): np.testing.assert_allclose(aa, bb, rtol=3e-5, atol=3e-6)
            iterations = max(4, args.iterations // 8) if mode == 'training' else args.iterations
            timings = {name: measure(function, iterations, args.samples) for name, function in functions.items()}
            results.append(dict(dtype=np.dtype(dtype).name, kind=kind, mode=mode,
                                verified_arrays=len(expected), timings=timings))
    report = dict(status='success',python=sys.version,numpy=np.__version__,
                  baseline_version=baseline.__version__,candidate_version=candidate.__version__,
                  source_hashes=dict(baseline=old_hashes,candidate=new_hashes),
                  order=[name for name, _, _ in variants],results=results,
                  scope='Complete calls; existing 2D composition vs direct 1D API, including VJP/JVP and Conv1d-SiLU-pools-linear-loss-SGD. No native Torch or GPU speed claim.')
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'],flows=len(results),output=args.output)))


if __name__ == '__main__': main()
