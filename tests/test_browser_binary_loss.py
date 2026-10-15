"""Independent native and scalar checks for stable binary logits losses."""
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from paperleet.browser import numpy_autograd as wasm


def compare(actual, expected, dtype):
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), equal_nan=True,
        atol=4e-6 if dtype == np.float32 else 2e-12, rtol=4e-6 if dtype == np.float32 else 2e-12)
    assert actual.dtype == expected.detach().numpy().dtype


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
@pytest.mark.parametrize('shape', [(), (4, 3), (2, 3, 4), (2, 0, 3)])
def test_weighted_broadcast_strided_scalar_empty_values_and_both_gradients(dtype, reduction, shape):
    rng = np.random.RandomState(241)
    count = int(np.prod(shape)) if shape else 1
    x = (rng.randn(count) * 4).astype(dtype).reshape(shape)
    y = rng.rand(count).astype(dtype).reshape(shape)
    if shape == (4, 3): x, y = x.T.copy().T, y.T.copy().T
    weight_shape = () if not shape else (2, 1, 1) if shape == (2, 3, 4) else (3,)
    positive_shape = (3, 1) if shape == (2, 3, 4) else () if not shape else (3,)
    for weighted, positive in ((False, False), (True, False), (False, True), (True, True)):
        w = rng.rand(*weight_shape).astype(dtype) if weight_shape else np.array(.4, dtype=dtype)
        p = (rng.rand(*positive_shape) * 3).astype(dtype) if positive_shape else np.array(2., dtype=dtype)
        a, ay = wasm.Tensor(x, requires_grad=True), wasm.Tensor(y, requires_grad=True)
        b, by = torch.tensor(x, requires_grad=True), torch.tensor(y, requires_grad=True)
        aw, ap = wasm.Tensor(w) if weighted else None, wasm.Tensor(p) if positive else None
        bw, bp = torch.tensor(w) if weighted else None, torch.tensor(p) if positive else None
        actual = wasm.binary_cross_entropy_with_logits(a, ay, weight=aw, pos_weight=ap, reduction=reduction)
        expected = F.binary_cross_entropy_with_logits(b, by, weight=bw, pos_weight=bp, reduction=reduction)
        compare(actual, expected, dtype)
        upstream = np.arange(actual.numel(), dtype=dtype).reshape(actual.shape) * .2 + .3
        (actual * wasm.Tensor(upstream)).sum().backward()
        (expected * torch.tensor(upstream)).sum().backward()
        compare(a.grad, b.grad, dtype); compare(ay.grad, by.grad, dtype)
        assert a.numpy().tobytes() == x.tobytes() and ay.numpy().tobytes() == y.tobytes()


def test_values_and_both_derivatives_against_independent_scalar_finite_differences():
    x = np.array([[-2., .3, 1.1], [.5, -.7, 2.]])
    y = np.array([[0., .4, 1.], [.7, .2, .9]])
    weights, positives = [.3, 2., 0.], [2., .4, 1.]
    def scalar(xs, ys):
        terms = []
        for row, labels in zip(xs, ys):
            for i, (value, target) in enumerate(zip(row, labels)):
                # Independent weighted Bernoulli log likelihood using scalar math.
                log_positive = -max(-value, 0.) - math.log1p(math.exp(-abs(value)))
                log_negative = -max(value, 0.) - math.log1p(math.exp(-abs(value)))
                terms.append(-weights[i] * (positives[i] * target * log_positive + (1-target) * log_negative))
        return sum(terms) / len(terms)
    a, ay = wasm.Tensor(x, requires_grad=True), wasm.Tensor(y, requires_grad=True)
    loss = wasm.binary_cross_entropy_with_logits(a, ay, weight=wasm.Tensor(np.array(weights)), pos_weight=wasm.Tensor(np.array(positives)))
    assert float(loss) == pytest.approx(scalar(x, y), abs=1e-14)
    loss.backward()
    for values, gradient in ((x, a.grad), (y, ay.grad)):
        for index in np.ndindex(values.shape):
            saved = values[index]
            values[index] = saved + 1e-6; above = scalar(x, y)
            values[index] = saved - 1e-6; below = scalar(x, y)
            values[index] = saved
            assert gradient.numpy()[index] == pytest.approx((above-below)/2e-6, abs=3e-10)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_extreme_logits_nonfinite_values_and_zero_weight_parity(dtype, reduction):
    x = np.array([-np.inf, np.inf, np.nan, -1000., 1000., 0., -20., 20.], dtype=dtype)
    for target in (0., 1., .3):
        for positive in (None, 0., .5, 2.):
            for weight in (None, 0., 1.7):
                y = np.full_like(x, target)
                a, ay = wasm.Tensor(x, requires_grad=True), wasm.Tensor(y, requires_grad=True)
                b, by = torch.tensor(x, requires_grad=True), torch.tensor(y, requires_grad=True)
                aw = None if weight is None else wasm.Tensor(np.array(weight, dtype=dtype))
                ap = None if positive is None else wasm.Tensor(np.array(positive, dtype=dtype))
                bw = None if weight is None else torch.tensor(weight, dtype=b.dtype)
                bp = None if positive is None else torch.tensor(positive, dtype=b.dtype)
                actual = wasm.binary_cross_entropy_with_logits(a, ay, weight=aw, pos_weight=ap, reduction=reduction)
                expected = F.binary_cross_entropy_with_logits(b, by, weight=bw, pos_weight=bp, reduction=reduction)
                actual.sum().backward(); expected.sum().backward()
                compare(actual, expected, dtype); compare(a.grad, b.grad, dtype); compare(ay.grad, by.grad, dtype)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_jvp_and_reverse_over_forward_match_native_for_logits_and_targets(dtype, reduction):
    rng = np.random.RandomState(251)
    values = [rng.randn(2, 3).astype(dtype), rng.rand(2, 3).astype(dtype),
              rng.randn(2, 3).astype(dtype), rng.randn(2, 3).astype(dtype)]
    for positive in (None, np.array([.2, 1., 3.], dtype=dtype)):
        a = [wasm.Tensor(v, requires_grad=True) for v in values]
        b = [torch.tensor(v, requires_grad=True) for v in values]
        aw, bw = wasm.Tensor(np.array([[.3], [2.]], dtype=dtype)), torch.tensor([[.3], [2.]], dtype=b[0].dtype)
        ap, bp = (None, None) if positive is None else (wasm.Tensor(positive), torch.tensor(positive))
        actual = wasm.jvp(lambda x,y:wasm.binary_cross_entropy_with_logits(x,y,weight=aw,pos_weight=ap,reduction=reduction),tuple(a[:2]),tuple(a[2:]))
        expected = torch.func.jvp(lambda x,y:F.binary_cross_entropy_with_logits(x,y,weight=bw,pos_weight=bp,reduction=reduction),tuple(b[:2]),tuple(b[2:]))
        for got, want in zip(actual, expected): compare(got, want, dtype)
        (actual[0].sum() + actual[1].square().sum()).backward()
        (expected[0].sum() + expected[1].square().sum()).backward()
        for got, want in zip(a, b): compare(got.grad, want.grad, dtype)


def test_fallbacks_shapes_no_grad_and_saved_weight_mutation():
    x, y = wasm.Tensor(np.array([.3, -.4]), requires_grad=True), wasm.Tensor(np.array([.2, .8]))
    with pytest.raises(ValueError): wasm.binary_cross_entropy_with_logits(x, y[:1])
    with pytest.raises(ValueError): wasm.binary_cross_entropy_with_logits(x, y, reduction='unknown')
    with pytest.raises(RuntimeError): wasm.binary_cross_entropy_with_logits(x, y, weight=wasm.Tensor(np.ones(3)))
    for kwargs in ({'reduce': False}, {'size_average': True}, {'weight': wasm.Tensor(np.ones(2), requires_grad=True)},
                   {'pos_weight': wasm.Tensor(np.ones(2), requires_grad=True)}):
        with pytest.raises(wasm.BrowserUnsupported): wasm.binary_cross_entropy_with_logits(x, y, **kwargs)
    with pytest.raises(wasm.BrowserUnsupported): wasm.binary_cross_entropy_with_logits(x.float(), y)
    with pytest.raises(wasm.BrowserUnsupported): wasm.binary_cross_entropy_with_logits(x, y, weight=wasm.Tensor(np.ones(2, dtype=np.float32)))
    for field in ('weight', 'pos_weight'):
        weight = wasm.Tensor(np.array([.4, 2.]))
        loss = wasm.binary_cross_entropy_with_logits(x, y, **{field: weight})
        with wasm.no_grad(): weight[0] = 3.
        with pytest.raises(wasm.BrowserUnsupported, match='modified'): loss.backward()
    weight = wasm.Tensor(np.array([.4, 2.]), requires_grad=True)
    with wasm.no_grad(): assert not wasm.binary_cross_entropy_with_logits(x, y, weight=weight).requires_grad
    with pytest.raises(wasm.BrowserUnsupported):
        wasm.jvp(lambda z:wasm.binary_cross_entropy_with_logits(z,y), (wasm.Tensor(np.array([np.inf, 0.])) ,), (wasm.Tensor(np.ones(2)),))


def test_nonfinite_targets_weights_out_of_range_labels_and_empty_jvp():
    x = np.array([-3., 0., 2., .4])
    y = np.array([np.nan, -.5, 1.5, .2])
    for weight in (None, np.array([1., np.inf, 0., np.nan])):
        a, ay = wasm.Tensor(x, requires_grad=True), wasm.Tensor(y, requires_grad=True)
        b, by = torch.tensor(x, requires_grad=True), torch.tensor(y, requires_grad=True)
        actual = wasm.binary_cross_entropy_with_logits(a, ay, weight=None if weight is None else wasm.Tensor(weight), reduction='none')
        expected = F.binary_cross_entropy_with_logits(b, by, weight=None if weight is None else torch.tensor(weight), reduction='none')
        actual.sum().backward(); expected.sum().backward()
        for got, want in ((actual, expected), (a.grad, b.grad), (ay.grad, by.grad)): compare(got, want, np.float64)
    a, b = wasm.Tensor(np.empty(0)), torch.empty(0, dtype=torch.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        actual = wasm.jvp(lambda z:wasm.binary_cross_entropy_with_logits(z,a), (a,), (a,))
    expected = torch.func.jvp(lambda z:F.binary_cross_entropy_with_logits(z,b), (b,), (b,))
    for got, want in zip(actual, expected): compare(got, want, np.float64)


def test_public_function_module_buffers_and_forward_api_in_isolation():
    script = '''from paperleet.browser.numpy_autograd import install
t=install();F=t.nn.functional
x=t.tensor([-.4,.8],dtype=t.float64,requires_grad=True);y=t.tensor([.2,.7],dtype=t.float64)
w=t.tensor([.5,2.],dtype=t.float64);p=t.tensor([2.,.3],dtype=t.float64)
m=t.nn.BCEWithLogitsLoss(weight=w,pos_weight=p,reduction='sum')
assert list(m.parameters())==[] and m.weight is w and m.pos_weight is p
assert t.allclose(m(x,y),F.binary_cross_entropy_with_logits(x,y,weight=w,pos_weight=p,reduction='sum'))
value,tangent=t.func.jvp(lambda z:F.binary_cross_entropy_with_logits(z,y),(x,),(t.ones_like(x),))
assert t.allclose(tangent,(x.sigmoid()-y).mean())
m(x,y).backward();assert x.grad is not None
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_sasrec_binary_logits_learner_passes_native_and_exported_full_browser_suite(tmp_path):
    from paperleet.browser_judge import export_browser_bundle
    from paperleet.catalog import problem
    from paperleet.judge.runner import run_problem
    entry = problem('retrieval/sasrec')
    source = (entry['directory']/'solution.py').read_text().replace('terms = F.softplus(-p) + F.softplus(n)',
        "terms = torch.nn.BCEWithLogitsLoss(reduction='none')(p, torch.ones_like(p)) + F.binary_cross_entropy_with_logits(n, torch.zeros_like(n), reduction='none')")
    learner = tmp_path/'learner.py'; learner.write_text(source)
    report = run_problem(entry['id'], learner, report_path=tmp_path/'native.json')
    assert report['passed'] and len(report['tests']) == 10
    metadata = export_browser_bundle(entry, tmp_path)
    manifest = json.loads((tmp_path/metadata['bundle_url']).read_text())
    runtime = json.loads((tmp_path/manifest['runtime_bundle_url']).read_text())
    bundle = {**manifest, 'adapter': runtime['adapter'], 'harness': runtime['harness'], 'files': {**runtime['files'], **manifest['files']}}
    payload = tmp_path/'bundle.json'; payload.write_text(json.dumps(bundle))
    script = '''import json,sys
from pathlib import Path
from paperleet.browser.harness import prepare,run_suite
b=json.load(open(sys.argv[1]));prepare(b,b['adapter'],sys.argv[3]);s=Path(sys.argv[2]).read_text()
r=run_suite(b,s,sys.argv[3]);assert r['passed'] and len(r['tests'])==10,r
wrong=s.replace('torch.zeros_like(n)','torch.ones_like(n)')
r=run_suite(b,wrong,sys.argv[3]);assert not r['passed'] and not r.get('error'),r
'''
    result = subprocess.run([sys.executable, '-c', script, str(payload), str(learner), str(tmp_path/'wasm')], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
