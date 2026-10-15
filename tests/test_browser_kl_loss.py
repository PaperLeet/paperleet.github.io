"""KL losses against native CPU, scalar likelihoods and the full distillation lesson."""
import json
import math
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from paperleet.browser import numpy_autograd as wasm


def native(x, y, **options):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return F.kl_div(x, y, **options)


def close(actual, expected, dtype):
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), equal_nan=True,
        rtol=5e-6 if dtype == np.float32 else 3e-12, atol=5e-6 if dtype == np.float32 else 3e-12)
    assert actual.dtype == expected.detach().numpy().dtype


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean', 'batchmean'])
@pytest.mark.parametrize('log_target', [False, True])
def test_scalar_strided_broadcast_empty_values_and_both_gradients(dtype, reduction, log_target):
    rng = np.random.RandomState(276)
    for xs, ys in [((), ()), ((3, 4), (3, 4)), ((2, 1, 4), (1, 3, 1)),
                   ((), (2, 3)), ((2, 0, 3), (1, 1, 3)), ((0, 3), (1, 3))]:
        x = np.asarray(rng.randn(*xs) if xs else -.7, dtype=dtype)
        y = np.asarray(rng.rand(*ys) + .1 if ys else .4, dtype=dtype)
        if log_target: y = np.log(y)
        if xs == (3, 4): x, y = x.T.copy().T, y.T.copy().T
        original = x.tobytes(), y.tobytes()
        a, ay = wasm.Tensor(x, requires_grad=True), wasm.Tensor(y, requires_grad=True)
        b, by = torch.tensor(x, requires_grad=True), torch.tensor(y, requires_grad=True)
        actual = wasm.kl_div(a, ay, reduction=reduction, log_target=log_target)
        expected = native(b, by, reduction=reduction, log_target=log_target)
        close(actual, expected, dtype)
        seed = np.arange(actual.numel(), dtype=dtype).reshape(actual.shape) * .3 + .2
        (actual * wasm.Tensor(seed)).sum().backward(); (expected * torch.tensor(seed)).sum().backward()
        close(a.grad, b.grad, dtype); close(ay.grad, by.grad, dtype)
        assert (x.tobytes(), y.tobytes()) == original


@pytest.mark.parametrize('log_target', [False, True])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean', 'batchmean'])
def test_zero_negative_nonfinite_and_extreme_targets_retain_native_behavior(log_target, reduction):
    x = np.array([0., 1., np.inf, -np.inf, np.nan, -1000., 1000.])
    for value in (0., -1., np.inf, -np.inf, np.nan, .5, 1000., -1000.):
        y = np.full_like(x, value)
        a, ay = wasm.Tensor(x, requires_grad=True), wasm.Tensor(y, requires_grad=True)
        b, by = torch.tensor(x, requires_grad=True), torch.tensor(y, requires_grad=True)
        actual = wasm.kl_div(a, ay, reduction=reduction, log_target=log_target)
        expected = native(b, by, reduction=reduction, log_target=log_target)
        with np.errstate(invalid='ignore'):
            actual.sum().backward(); expected.sum().backward()
        for got, want in ((actual, expected), (a.grad, b.grad), (ay.grad, by.grad)): close(got, want, np.float64)


@pytest.mark.parametrize('log_target', [False, True])
def test_independent_scalar_likelihood_and_two_channel_finite_differences(log_target):
    x, y = np.array([[-.8, -1.7, -.3], [-2., -.5, -1.]]), np.array([[.2, .3, .5], [.6, .1, .3]])
    if log_target: y = np.log(y)
    def scalar():
        return sum((math.exp(t) * (t - v) if log_target else t * math.log(t / math.exp(v)))
                   for row, targets in zip(x, y) for v, t in zip(row, targets)) / len(x)
    a, ay = wasm.Tensor(x, requires_grad=True), wasm.Tensor(y, requires_grad=True)
    loss = wasm.kl_div(a, ay, reduction='batchmean', log_target=log_target)
    assert float(loss) == pytest.approx(scalar(), abs=2e-15)
    loss.backward()
    for values, gradient in ((x, a.grad), (y, ay.grad)):
        for index in np.ndindex(values.shape):
            saved = values[index]
            values[index] = saved + 1e-6; above = scalar()
            values[index] = saved - 1e-6; below = scalar()
            values[index] = saved
            assert gradient.numpy()[index] == pytest.approx((above - below) / 2e-6, abs=2e-10)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean', 'batchmean'])
@pytest.mark.parametrize('log_target', [False, True])
def test_jvp_and_reverse_over_forward_with_broadcast_and_target_gradients(dtype, reduction, log_target):
    rng = np.random.RandomState(277)
    values = [rng.randn(2, 1, 3).astype(dtype), (rng.rand(1, 4, 3) + .1).astype(dtype),
              rng.randn(2, 1, 3).astype(dtype), rng.randn(1, 4, 3).astype(dtype)]
    if log_target: values[1] = np.log(values[1])
    a, b = [wasm.Tensor(v, requires_grad=True) for v in values], [torch.tensor(v, requires_grad=True) for v in values]
    actual = wasm.jvp(lambda x,y:wasm.kl_div(x,y,reduction=reduction,log_target=log_target),tuple(a[:2]),tuple(a[2:]))
    expected = torch.func.jvp(lambda x,y:native(x,y,reduction=reduction,log_target=log_target),tuple(b[:2]),tuple(b[2:]))
    for got, want in zip(actual, expected): close(got, want, dtype)
    (actual[0].sum() + .2 * actual[1].square().sum()).backward()
    (expected[0].sum() + .2 * expected[1].square().sum()).backward()
    for got, want in zip(a, b): close(got.grad, want.grad, dtype)


def test_validation_server_boundaries_no_grad_and_saved_storage_mutation():
    x, y = wasm.Tensor(np.array([-.4, -.8]), requires_grad=True), wasm.Tensor(np.array([.3, .7]))
    for options in ({'reduce':False}, {'size_average':True}):
        with pytest.raises(wasm.BrowserUnsupported): wasm.kl_div(x, y, **options)
    with pytest.raises(ValueError): wasm.kl_div(x, y, reduction='other')
    with pytest.raises(TypeError): wasm.kl_div(x, y, log_target='false')
    with pytest.raises(RuntimeError): wasm.kl_div(x, wasm.Tensor(np.ones(3)))
    with pytest.raises(wasm.BrowserUnsupported): wasm.kl_div(x.float(), y)
    for mutated in ('input', 'target'):
        a, ay = wasm.Tensor(x.numpy().copy(), requires_grad=True), wasm.Tensor(y.numpy().copy()); loss = wasm.kl_div(a, ay)
        with wasm.no_grad(): (a if mutated == 'input' else ay)[0] = .2
        with pytest.raises(wasm.BrowserUnsupported, match='modified'): loss.backward()
    with wasm.no_grad(): assert not wasm.kl_div(x, y).requires_grad
    for value, log_target in ((0.,False), (-1.,False), (np.inf,True), (-np.inf,True), (1000.,True)):
        target = wasm.Tensor(np.full(2, value))
        with pytest.raises(wasm.BrowserUnsupported):
            wasm.jvp(lambda z:wasm.kl_div(z,target,log_target=log_target),(x,),(y,))


@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean', 'batchmean'])
@pytest.mark.parametrize('log_target', [False, True])
def test_empty_and_scalar_jvp_match_native(reduction, log_target):
    for shape in ((), (2, 0, 3), (0, 3)):
        x, y = np.full(shape, -.5), np.full(shape, -.7 if log_target else .3)
        a, ay, b, by = wasm.Tensor(x), wasm.Tensor(y), torch.tensor(x), torch.tensor(y)
        with np.errstate(invalid='ignore', divide='ignore'):
            actual = wasm.jvp(lambda x,y:wasm.kl_div(x,y,reduction=reduction,log_target=log_target),(a,ay),(a,ay))
        expected = torch.func.jvp(lambda x,y:native(x,y,reduction=reduction,log_target=log_target),(b,by),(b,by))
        for got, want in zip(actual, expected): close(got, want, np.float64)


def test_installed_function_module_forward_dispatch_and_empty_jvp_in_isolation():
    script = '''from paperleet.browser.numpy_autograd import install
t=install();F=t.nn.functional
x=t.tensor([-.4,-.8],dtype=t.float64,requires_grad=True);y=t.tensor([-.3,-.7],dtype=t.float64,requires_grad=True)
m=t.nn.KLDivLoss(reduction='sum',log_target=True)
assert list(m.parameters())==[] and m.log_target is True
assert t.allclose(m(x,y),F.kl_div(x,y,reduction='sum',log_target=True))
value,tangent=t.func.jvp(lambda z:m(z,y),(x,),(t.ones_like(x),))
assert t.allclose(tangent,-y.exp().sum())
m(x,y).backward();assert x.grad is not None and y.grad is not None
for reduction in ('none','sum','mean','batchmean'):
    z=t.empty(2,0,dtype=t.float64)
    primal,tangent=t.func.jvp(lambda x:F.kl_div(x,z,reduction=reduction,log_target=True),(z,),(z,))
    assert primal.shape==tangent.shape
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_distillm2_kl_api_learner_passes_native_and_exported_browser_suites(tmp_path):
    from paperleet.browser_judge import export_browser_bundle
    from paperleet.catalog import problem
    from paperleet.judge.runner import run_problem
    entry = problem('finetuning/distillm2')
    source = (entry['directory']/'solution.py').read_text().replace('import math','import math\nimport torch.nn.functional as F')
    source = source.replace('return (lp.exp() * (lp - lm)).sum(-1)', "return F.kl_div(lm, lp, reduction='none', log_target=True).sum(-1)")
    source = source.replace('return (lq.exp() * (lq - lm)).sum(-1)', "return torch.nn.KLDivLoss(reduction='none', log_target=True)(lm, lq).sum(-1)")
    learner = tmp_path/'learner.py'; learner.write_text(source)
    report = run_problem(entry['id'],learner,report_path=tmp_path/'native.json')
    assert report['passed'] and len(report['tests']) == 9
    metadata = export_browser_bundle(entry,tmp_path)
    manifest = json.loads((tmp_path/metadata['bundle_url']).read_text());runtime = json.loads((tmp_path/manifest['runtime_bundle_url']).read_text())
    bundle = {**manifest,'adapter':runtime['adapter'],'harness':runtime['harness'],'files':{**runtime['files'],**manifest['files']}}
    payload=tmp_path/'bundle.json';payload.write_text(json.dumps(bundle))
    script = '''import json,sys
from pathlib import Path
from paperleet.browser.harness import prepare,run_suite
b=json.load(open(sys.argv[1]));prepare(b,b['adapter'],sys.argv[3]);s=Path(sys.argv[2]).read_text()
r=run_suite(b,s,sys.argv[3]);assert r['passed'] and len(r['tests'])==9,r
r=run_suite(b,s.replace('F.kl_div(lm, lp,','F.kl_div(lp, lm,'),sys.argv[3])
assert not r['passed'] and not r.get('error'),r
assert all('BrowserUnsupported' not in row.get('message','') for row in r['tests']),r
'''
    result=subprocess.run([sys.executable,'-c',script,str(payload),str(learner),str(tmp_path/'wasm')],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
