"""Actual-library SDPA comparison, including repeated-K/V and training controls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np


def load(name, path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path / '__init__.py', submodule_search_locations=[str(path)])
    package = importlib.util.module_from_spec(spec); sys.modules[name] = package; spec.loader.exec_module(package)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.suffix == '.py' or p.name in ('LICENSE', 'NOTICE')}
    return package, hashes


def run(baseline_source, candidate_source, samples=5, calls=8, reverse_order=False):
    baseline, bh = load('_attention_baseline', baseline_source)
    candidate, ch = load('_attention_candidate', candidate_source)
    variants = dict(baseline=baseline.torch, candidate=candidate.torch, repeated_control=candidate.torch)
    rng = np.random.default_rng(5015); results = []
    cases = [('small', 2, 4, 4, 4, 4, 6, 8), ('self', 2, 4, 4, 4, 64, 64, 16),
             ('grouped', 2, 8, 2, 1, 64, 64, 16), ('cross_grouped', 2, 6, 2, 3, 8, 48, 16),
             ('long_grouped', 1, 8, 2, 2, 128, 128, 16)]
    def attention(t, name, q, k, v, causal=False):
        grouped = q.shape[-3] != k.shape[-3] or q.shape[-3] != v.shape[-3]
        if grouped and name != 'candidate':
            # The old library already broadcasts singleton heads. Preserve
            # that efficient path in the baseline; retain the fully repeated
            # formulation separately as a same-source control.
            if k.shape[-3] != q.shape[-3] and (k.shape[-3] != 1 or name == 'repeated_control'):
                k = k.repeat_interleave(q.shape[-3] // k.shape[-3], -3)
            if v.shape[-3] != q.shape[-3] and (v.shape[-3] != 1 or name == 'repeated_control'):
                v = v.repeat_interleave(q.shape[-3] // v.shape[-3], -3)
        return t.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=grouped and name == 'candidate')
    def measure(actions):
        for action in actions.values(): action()
        times = {name: [] for name in actions}
        for sample in range(samples):
            names = list(actions)
            if (sample + int(reverse_order)) % 2: names.reverse()
            for name in names:
                start = time.perf_counter()
                for _ in range(calls): actions[name]()
                times[name].append((time.perf_counter() - start) / calls)
        return dict(samples_s=times, median_s={name: statistics.median(values) for name, values in times.items()})
    for kind, b, hq, hk, hv, length, source, width in cases:
        for dtype in ('float32', 'float64'):
            arrays = [rng.normal(size=s).astype(dtype) for s in [(b,hq,length,width),(b,hk,source,width),(b,hv,source,width)]]
            seed = rng.normal(size=(b,hq,length,width)).astype(dtype); expected = None
            for name, t in variants.items():
                values = [t.tensor(a, requires_grad=True) for a in arrays]
                y = attention(t, name, *values, causal=kind != 'cross_grouped'); y.backward(t.tensor(seed))
                actual = [x.numpy() for x in (y, *[v.grad for v in values])]
                if expected is None: expected = [a.copy() for a in actual]
                else:
                    for a, e in zip(actual, expected): np.testing.assert_allclose(a,e,rtol=3e-5 if dtype=='float32' else 1e-11,atol=1e-5 if dtype=='float32' else 5e-12)
            for mode in ('forward', 'reverse'):
                def action(name):
                    t = variants[name]
                    if mode == 'forward':
                        with t.no_grad(): return attention(t, name, *[t.tensor(a) for a in arrays], causal=kind != 'cross_grouped')
                    values = [t.tensor(a, requires_grad=True) for a in arrays]
                    y = attention(t, name, *values, causal=kind != 'cross_grouped'); y.backward(t.tensor(seed)); return y
                results.append(dict(kind=kind, dtype=dtype, mode=mode, shapes=[list(a.shape) for a in arrays], **measure({name: lambda name=name: action(name) for name in variants})))
    for dtype in ('float32', 'float64'):
        x = rng.normal(size=(2,16,8)).astype(dtype)
        arrays = [rng.normal(size=s).astype(dtype)*.1 for s in [(64,8),(16,8),(8,8),(8,64)]]
        target = rng.normal(size=x.shape).astype(dtype)
        def make(name):
            t = variants[name]; params = [t.nn.Parameter(t.tensor(a)) for a in arrays]
            optimizer = t.optim.SGD(params, lr=.01, momentum=.7); sample = t.tensor(x, requires_grad=True); goal = t.tensor(target)
            def step():
                optimizer.zero_grad(); sample.grad=None
                q,k,v=[t.nn.functional.linear(sample,p).reshape(2,16,h,8).transpose(1,2) for p,h in zip(params,[8,2,1])]
                y=attention(t,name,q,k,v,causal=True).transpose(1,2).reshape(2,16,64)
                y=t.nn.functional.gelu(t.nn.functional.linear(y,params[3]))+sample
                loss=(y-goal).square().mean();loss.backward();optimizer.step()
                return [loss.numpy().copy(),*[p.numpy().copy() for p in params],sample.grad.numpy().copy()]
            return step
        step=make('baseline');expected=[step() for _ in range(3)]
        for name in ('candidate','repeated_control'):
            step=make(name)
            for want in expected:
                for a,e in zip(step(),want):np.testing.assert_allclose(a,e,rtol=4e-5 if dtype=='float32' else 1e-11,atol=8e-6 if dtype=='float32' else 5e-12)
        actions={name:make(name) for name in variants}
        results.append(dict(kind='training',dtype=dtype,mode='encoder_sgd',**measure(actions)))
    return dict(status='success',baseline_version=baseline.__version__,candidate_version=candidate.__version__,source_hashes=dict(baseline=bh,candidate=ch),samples=samples,calls=calls,flows=results,scope='Actual 0.49 fused baseline preserves singleton-head broadcasting and repeats only non-singleton K/V heads that must expand for GQA; candidate grouped contractions and same-source fully repeated control. Values/all VJPs and three optimizer updates checked before timings. No RSS, native Torch, GPU or universal speed claim.')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline-source',required=True);p.add_argument('--candidate-source',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=5);p.add_argument('--calls',type=int,default=8);p.add_argument('--reverse',action='store_true');a=p.parse_args()
    Path(a.output).write_text(json.dumps(run(a.baseline_source,a.candidate_source,a.samples,a.calls,a.reverse),indent=2)+'\n')
