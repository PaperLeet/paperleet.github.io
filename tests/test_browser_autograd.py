"""Independent native PyTorch parity tests for the bounded WASM autodiff engine.

Do not install the adapter as torch in this process: compare both engines on the
same arrays, rather than comparing two expressions implemented by one engine.
"""
import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


def pair(data, dtype=np.float64):
    data = np.asarray(data, dtype=dtype)
    return wasm.Tensor(data.copy(), requires_grad=True), torch.tensor(data.copy(), requires_grad=True)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape,dim,keepdim', [((3, 4), -1, False), ((2, 3, 4), (0, 2), True), ((3, 4), None, False), ((2, 0, 4), -1, False), ((), None, True)])
def test_euclidean_norm_and_normalization_values_and_vjps(dtype, shape, dim, keepdim):
    data = np.random.RandomState(9).randn(*shape) if shape else np.asarray(-2.)
    if shape and data.size: data.reshape(-1)[:4] = 0
    x, native = pair(data, dtype)
    value, expected = x.norm(2, dim, keepdim), native.norm(2, dim, keepdim)
    value.sum().backward(); expected.sum().backward()
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(value.numpy(), expected.detach().numpy(), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(x.grad.numpy(), native.grad.numpy(), rtol=tolerance, atol=tolerance)
    if shape:
        x, native = pair(data, dtype)
        axis = dim if dim is not None else -1
        got, want = wasm.normalize(x, dim=axis, eps=1e-4), torch.nn.functional.normalize(native, dim=axis, eps=1e-4)
        got.square().sum().backward(); want.square().sum().backward()
        np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(x.grad.numpy(), native.grad.numpy(), rtol=tolerance, atol=tolerance)


def test_normalize_zero_and_sub_epsilon_gradients():
    x, native = pair([[0., 0.], [1e-6, -2e-6], [.3, -.4]])
    value, expected = wasm.normalize(x, dim=-1, eps=1e-4), torch.nn.functional.normalize(native, dim=-1, eps=1e-4)
    weights = np.asarray([[1., 2.], [3., -1.], [.7, -2.]])
    (value * wasm.Tensor(weights)).sum().backward(); (expected * torch.tensor(weights)).sum().backward()
    np.testing.assert_allclose(value.numpy(), expected.detach().numpy(), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(x.grad.numpy(), native.grad.numpy(), atol=1e-10, rtol=1e-12)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('dim,keepdim', [(0, False), (-1, True), ((0, 2), False)])
def test_logsumexp_stable_reductions_and_gradients(dim, keepdim, dtype):
    data = np.random.RandomState(15).randn(2, 3, 4) * 20 + 10000
    x, native = pair(data, dtype)
    value, expected = x.logsumexp(dim, keepdim), native.logsumexp(dim, keepdim)
    value.sum().backward(); expected.sum().backward()
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(value.numpy(), expected.detach().numpy(), atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(x.grad.numpy(), native.grad.numpy(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('data', [np.empty((2, 0)), np.asarray([[float('-inf'), float('-inf')], [float('inf'), 1.]])])
def test_logsumexp_empty_and_nonfinite_semantics(data):
    x, native = pair(data)
    value, expected = x.logsumexp(-1), native.logsumexp(-1)
    value.backward(wasm.Tensor(np.ones(value.shape))); expected.backward(torch.ones_like(expected))
    np.testing.assert_allclose(value.numpy(), expected.detach().numpy(), equal_nan=True)
    np.testing.assert_allclose(x.grad.numpy(), native.grad.numpy(), equal_nan=True)


@pytest.mark.parametrize('shape', [(5,), (3, 5), (5, 3), (0, 3)])
@pytest.mark.parametrize('offset', [-4, -1, 0, 2, 6])
def test_diagonal_construction_extraction_and_gradients(shape, offset):
    x, native = pair(np.arange(np.prod(shape)).reshape(shape))
    value, expected = x.diag(offset), native.diag(offset)
    value.sum().backward(); expected.sum().backward()
    np.testing.assert_array_equal(value.numpy(), expected.detach().numpy())
    np.testing.assert_array_equal(x.grad.numpy(), native.grad.numpy())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
@pytest.mark.parametrize('labels', [[0, 2, -100, 1], [-100] * 4, []])
def test_cross_entropy_independent_native_values_and_vjps(dtype, reduction, labels):
    data = np.asarray([[10000., 9999., -10000.], [-4., -2., -1.], [8., 2., -3.], [.2, .7, -.4]])[:len(labels)]
    x, native = pair(data, dtype)
    target = np.asarray(labels, dtype=np.int64)
    value = wasm.cross_entropy(x, wasm.Tensor(target), reduction=reduction)
    expected = torch.nn.functional.cross_entropy(native, torch.tensor(target), reduction=reduction)
    weight = np.linspace(.2, 1.3, len(labels)).astype(dtype) if reduction == 'none' else np.asarray(.7, dtype=dtype)
    (value * wasm.Tensor(weight)).sum().backward(); (expected * torch.tensor(weight)).sum().backward()
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(value.numpy(), expected.detach().numpy(), rtol=tolerance, atol=tolerance, equal_nan=True)
    np.testing.assert_allclose(x.grad.numpy(), native.grad.numpy(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('indexing', ['ij', 'xy'])
@pytest.mark.parametrize('empty', [False, True])
def test_meshgrid_views_scalar_empty_and_shared_input_gradients(indexing, empty):
    x, native = pair([] if empty else [-2., 0., 3.])
    scalar, native_scalar = pair(np.asarray(1.5))
    grids = wasm.meshgrid((x, scalar, x), indexing=indexing)
    expected = torch.meshgrid(native, native_scalar, native, indexing=indexing)
    for actual, reference in zip(grids, expected): np.testing.assert_array_equal(actual.numpy(), reference.detach().numpy())
    value = sum((grid.square().sum() * (i + 1) for i, grid in enumerate(grids)))
    reference = sum((grid.square().sum() * (i + 1) for i, grid in enumerate(expected)))
    value.backward(); reference.backward()
    np.testing.assert_allclose(x.grad.numpy(), native.grad.numpy())
    np.testing.assert_allclose(scalar.grad.numpy(), native_scalar.grad.numpy())


def test_contrastive_operator_unsupported_options_do_not_silently_change_objective():
    x, _ = pair([[1., 2.]])
    target = wasm.Tensor(np.asarray([0], dtype=np.int64))
    for options in ({'weight': wasm.Tensor([1., 2.])}, {'label_smoothing': .1}, {'size_average': False}):
        with pytest.raises(wasm.BrowserUnsupported): wasm.cross_entropy(x, target, **options)
    with pytest.raises(wasm.BrowserUnsupported): wasm.cross_entropy(x, wasm.Tensor([[1., 0.]]))
    with pytest.raises(wasm.BrowserUnsupported): wasm.normalize(x, p=1)
    with pytest.raises(wasm.BrowserUnsupported): x.norm(p=float('inf'))
    with pytest.raises(wasm.BrowserUnsupported): wasm.normalize(x, out=x)
    for method, args in ((x.norm, ()), (x.logsumexp, (-1,)), (x.diag, ())):
        with pytest.raises(wasm.BrowserUnsupported): method(*args, out=x)


@pytest.mark.parametrize('shapes', [((2,3,4,5),(1,3,5,6)), ((3,5),(5,)), ((5,),(2,5,4)), ((5,),(5,)), ((5,),(5,4))])
def test_batched_matmul_values_and_reverse_gradients(shapes):
    rng = np.random.RandomState(77)
    a, aa = pair(rng.randn(*shapes[0])); b, bb = pair(rng.randn(*shapes[1]))
    actual, expected = a @ b, aa @ bb
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), rtol=1e-12, atol=1e-12)
    weight = rng.randn(*actual.shape) if actual.shape else np.asarray(.7)
    got = wasm.autograd_grad((actual * wasm.Tensor(weight)).sum(), (a,b))
    want = torch.autograd.grad((expected * torch.tensor(weight)).sum(), (aa,bb))
    for g, e in zip(got, want): np.testing.assert_allclose(g.numpy(), e.numpy(), rtol=1e-11, atol=1e-11)


def test_masked_attention_float64_gradients_against_native_torch():
    rng = np.random.RandomState(18)
    pairs = [pair(rng.randn(*shape)) for shape in [(2,3,4,5),(2,3,6,5),(2,3,6,7)]]
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    mask = rng.rand(4,6) > .5
    mask[1] = False
    def expression(q,k,v, backend):
        logits = q @ k.transpose(-1,-2) / np.sqrt(5)
        allowed = backend.tensor(mask)
        logits = logits.masked_fill(~allowed, -float('inf'))
        logits = backend.where(allowed.any(-1,keepdim=True), logits, backend.zeros_like(logits))
        weights = logits.softmax(-1).masked_fill(~allowed,0.)
        return weights @ v
    # Only the primitives used by the independent expression are supplied.
    import types
    bridge = types.SimpleNamespace(tensor=wasm.tensor, where=wasm.where, zeros_like=lambda x: wasm.Tensor(np.zeros(x.shape, dtype=x.dtype)))
    got, expected = expression(*a, bridge), expression(*b, torch)
    np.testing.assert_allclose(got.numpy(),expected.detach().numpy(),atol=1e-12,rtol=1e-12)
    weight = rng.randn(*got.shape)
    grad = wasm.autograd_grad((got*wasm.Tensor(weight)).sum(),a)
    reference = torch.autograd.grad((expected*torch.tensor(weight)).sum(),b)
    for g,e in zip(grad,reference): np.testing.assert_allclose(g.numpy(),e.numpy(),atol=1e-11,rtol=1e-11)
    assert np.isfinite(got.numpy()).all()
    assert np.count_nonzero(got.numpy()[:,:,1]) == 0


def test_views_repeated_indices_broadcast_and_gradient_accumulation():
    rng = np.random.RandomState(11)
    x, xx = pair(rng.randn(3,4))
    def expression(t):
        return ((t[[1,1,0]].transpose(0,1).unsqueeze(0).repeat(2,1,1) - 2).square().mean() + t.sum(0).expand(5,4).exp().mean())
    value, expected = expression(x), expression(xx)
    value.backward(); expected.backward(retain_graph=True)
    np.testing.assert_allclose(x.grad.numpy(),xx.grad.numpy(),rtol=1e-11,atol=1e-11)
    value.backward(); expected.backward()
    np.testing.assert_allclose(x.grad.numpy(),xx.grad.numpy(),rtol=1e-11,atol=1e-11)


@pytest.mark.parametrize('dtype',[np.float64,np.float32])
def test_adam_steps_with_identical_inputs_match_native_torch(dtype):
    rng = np.random.RandomState(5)
    a, aa = pair(rng.randn(2,4)*.1,dtype); b, bb = pair(np.zeros((3,2)),dtype)
    x = rng.randn(8,4).astype(dtype); y = rng.randn(8,3).astype(dtype)
    optimizer = wasm.Adam([a,b],lr=.06)
    native = torch.optim.Adam([aa,bb],lr=.06)
    for _ in range(12):
        optimizer.zero_grad(); native.zero_grad()
        loss = (wasm.Tensor(x) @ a.T @ b.T * 2 - wasm.Tensor(y)).square().mean()
        reference = (torch.tensor(x) @ aa.T @ bb.T * 2 - torch.tensor(y)).square().mean()
        loss.backward(); reference.backward(); optimizer.step(); native.step()
    tolerance = 2e-6 if dtype==np.float32 else 1e-11
    for got, expected in [(a,aa),(b,bb)]: np.testing.assert_allclose(got.numpy(),expected.detach().numpy(),atol=tolerance,rtol=tolerance)


def test_detach_no_grad_and_unsupported_higher_order_derivatives():
    a, _ = pair([1.,2.,3.])
    detached = a.detach()
    assert not detached.requires_grad
    with wasm.no_grad():
        assert not (a*3).requires_grad
    assert (a*3).requires_grad
    with pytest.raises(wasm.BrowserUnsupported):
        wasm.autograd_grad(a.sum(),[a],create_graph=True)
    with pytest.raises(wasm.BrowserUnsupported):
        a[0] = 10


def test_einsum_vjp_and_negative_axis_permute_against_native():
    rng = np.random.RandomState(81)
    q, qq = pair(rng.randn(2,3,4,5)); k, kk = pair(rng.randn(2,3,6,5))
    actual = wasm.einsum('bhid,bhjd->bhij',q,k).permute(0,1,-1,-2)
    expected = torch.einsum('bhid,bhjd->bhij',qq,kk).permute(0,1,-1,-2)
    weight = rng.randn(*actual.shape)
    gradients = wasm.autograd_grad((actual*wasm.Tensor(weight)).sum(),(q,k))
    native = torch.autograd.grad((expected*torch.tensor(weight)).sum(),(qq,kk))
    for got, want in zip(gradients,native): np.testing.assert_allclose(got.numpy(),want.numpy(),atol=1e-11,rtol=1e-11)


def test_integer_mask_fill_preserves_float32_and_blocked_gradients():
    x, xx = pair([[1.,2.],[3.,4.]],np.float32)
    mask = np.array([[True,False],[False,True]])
    actual = x.masked_fill(wasm.Tensor(mask),0)
    expected = xx.masked_fill(torch.tensor(mask),0)
    assert actual.dtype == np.dtype('float32')
    actual.sum().backward();expected.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),xx.grad.numpy())


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_log_softmax_gather_and_masked_sequence_gradients(dtype):
    rng=np.random.RandomState(103)
    x, xx=pair(rng.randn(2,5,7)*10,dtype)
    ids=rng.randint(0,7,(2,4,1)); mask=np.array([[0,1,1,0],[1,1,0,1]],dtype=np.int64)
    actual=(x[:,:-1].log_softmax(-1).gather(-1,wasm.Tensor(ids)).squeeze(-1)*wasm.Tensor(mask)).sum(-1)
    expected=(xx[:,:-1].log_softmax(-1).gather(-1,torch.tensor(ids)).squeeze(-1)*torch.tensor(mask)).sum(-1)
    assert actual.dtype==np.dtype(dtype)
    actual.sum().backward();expected.sum().backward()
    tolerance=3e-6 if dtype==np.float32 else 1e-11
    np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),atol=tolerance,rtol=tolerance)
    np.testing.assert_allclose(x.grad.numpy(),xx.grad.numpy(),atol=tolerance,rtol=tolerance)


def test_stable_logsigmoid_clamp_and_transcendental_gradients():
    values=np.array([-10000.,-10.,-.2,0.,.5,10.,10000.])
    x,xx=pair(values)
    actual=x.logsigmoid(); expected=torch.nn.functional.logsigmoid(xx)
    np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),atol=1e-12,rtol=1e-12)
    actual.sum().backward();expected.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(),xx.grad.numpy(),atol=1e-12,rtol=1e-12)
    y,yy=pair([-.8,-.49,0.,.49,.8])
    a=(y.clamp(-.5,.5).sin()+y.cos()+y.acos()+y.tanh()).sum()
    b=(yy.clamp(-.5,.5).sin()+yy.cos()+yy.acos()+yy.tanh()).sum()
    a.backward();b.backward()
    np.testing.assert_allclose(y.grad.numpy(),yy.grad.numpy(),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('shape',[(7,),(2,3,5),(2,3,4,5)])
def test_layer_norm_oracle_values_and_gradients_against_native(shape):
    rng=np.random.RandomState(44)
    x,xx=pair(rng.randn(*shape));w,ww=pair(rng.randn(shape[-1]));b,bb=pair(rng.randn(shape[-1]))
    actual=wasm.layer_norm_oracle(x,(shape[-1],),w,b,1e-4)
    expected=torch.nn.functional.layer_norm(xx,(shape[-1],),ww,bb,1e-4)
    upstream=rng.randn(*shape)
    a=wasm.autograd_grad((actual*wasm.Tensor(upstream)).sum(),(x,w,b))
    e=torch.autograd.grad((expected*torch.tensor(upstream)).sum(),(xx,ww,bb))
    np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),atol=1e-11,rtol=1e-11)
    for got,want in zip(a,e):np.testing.assert_allclose(got.numpy(),want.numpy(),atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('correction',[0,1])
def test_variance_standard_deviation_gradient_parity(correction):
    rng=np.random.RandomState(62)
    x,xx=pair(rng.randn(2,3,5))
    actual=x.var(-1,correction=correction).mean()+x.std((0,2),correction=correction).sum()
    expected=xx.var(-1,correction=correction).mean()+xx.std((0,2),correction=correction).sum()
    actual.backward();expected.backward()
    np.testing.assert_allclose(x.grad.numpy(),xx.grad.numpy(),atol=1e-11,rtol=1e-11)


def test_mixed_matmul_dtype_is_rejected_like_native():
    with pytest.raises(RuntimeError):wasm.Tensor(np.ones((2,3),dtype=np.float32))@wasm.Tensor(np.ones((3,2),dtype=np.float64))


def test_clamp_boundary_matches_pinned_production_torch26_contract():
    # Production uses torch==2.6.0. Its FunctionsManual.cpp explicitly chooses
    # subgradient 1 at min/max; compare other kernels away from these kinks.
    x=wasm.Tensor(np.array([-.8,-.5,0.,.5,.8]),requires_grad=True)
    x.clamp(-.5,.5).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),[0.,1.,1.,1.,0.])


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_seeded_integer_and_uniform_stream_matches_native(dtype):
    native=torch.Generator().manual_seed(42);generator=wasm.Generator().manual_seed(42)
    actual=wasm.randint(0,6,(20000,2),generator=generator)
    expected=torch.randint(0,6,(20000,2),generator=native)
    np.testing.assert_array_equal(actual.numpy(),expected.numpy())
    actual=wasm.create('rand',20000,generator=generator,dtype=dtype)
    expected=torch.rand(20000,generator=native,dtype=torch.float64 if dtype==np.float64 else torch.float32)
    np.testing.assert_array_equal(actual.numpy(),expected.numpy())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_relu_gate_and_feedforward_parameter_gradients_match_native(dtype):
    rng = np.random.RandomState(317)
    pairs = [pair(rng.randn(*shape), dtype) for shape in ((2, 3, 4), (7, 4), (7,), (4, 7), (4,))]
    x, w1, b1, w2, b2 = [p[0] for p in pairs]
    xx, ww1, bb1, ww2, bb2 = [p[1] for p in pairs]
    actual = ((x @ w1.T + b1).relu() @ w2.T + b2)
    expected = torch.nn.functional.linear(torch.nn.functional.relu(torch.nn.functional.linear(xx, ww1, bb1)), ww2, bb2)
    upstream = rng.randn(*actual.shape).astype(dtype)
    a = wasm.autograd_grad((actual * wasm.Tensor(upstream)).sum(), [p[0] for p in pairs])
    b = torch.autograd.grad((expected * torch.tensor(upstream)).sum(), [p[1] for p in pairs])
    tolerance = 1e-5 if dtype == np.float32 else 1e-11
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=tolerance, rtol=tolerance)
    for got, want in zip(a, b): np.testing.assert_allclose(got.numpy(), want.numpy(), atol=tolerance, rtol=tolerance)
    gate, native_gate = pair([-np.inf, -1., -0., 0., 2., np.inf, np.nan], dtype)
    gate.relu().sum().backward(); native_gate.relu().sum().backward()
    np.testing.assert_array_equal(gate.grad.numpy(), native_gate.grad.numpy())


@pytest.mark.parametrize('stack,axis', [(False, 0), (False, -1), (True, 0), (True, -1)])
def test_concatenation_and_stacking_values_views_and_accumulated_gradients(stack, axis):
    rng = np.random.RandomState(601)
    x, xx = pair(rng.randn(2, 3, 4)); y, yy = pair(rng.randn(2, 3, 4))
    # Duplicate inputs must accumulate VJPs, including through strided views.
    a = wasm.concatenate([x.transpose(0, 1), y.transpose(0, 1), x.transpose(0, 1)], axis, stack=stack)
    op = torch.stack if stack else torch.cat
    b = op([xx.transpose(0, 1), yy.transpose(0, 1), xx.transpose(0, 1)], dim=axis)
    upstream = rng.randn(*a.shape)
    a_grad = wasm.autograd_grad((a * wasm.Tensor(upstream)).sum(), (x, y))
    b_grad = torch.autograd.grad((b * torch.tensor(upstream)).sum(), (xx, yy))
    np.testing.assert_array_equal(a.numpy(), b.detach().numpy())
    for got, want in zip(a_grad, b_grad): np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-12)
    assert x.transpose(0, 1).is_contiguous() is False
    assert x.transpose(0, 1).contiguous().is_contiguous() is True


def test_concatenation_empty_slices_and_invalid_dimensions():
    x, xx = pair(np.empty((2, 0, 3))); y, yy = pair(np.ones((2, 4, 3)))
    a = wasm.concatenate([x, y], dim=1); b = torch.cat([xx, yy], dim=1)
    a.sum().backward(); b.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), xx.grad.numpy())
    np.testing.assert_array_equal(y.grad.numpy(), yy.grad.numpy())
    with pytest.raises(IndexError): wasm.concatenate([y], dim=3)
    with pytest.raises(RuntimeError): wasm.concatenate([])
    arrays = [np.array([1.], dtype=np.float32), np.array([2], dtype=np.int64)]
    actual = wasm.concatenate([wasm.Tensor(values) for values in arrays])
    expected = torch.cat([torch.tensor(values) for values in arrays])
    assert actual.dtype == np.dtype('float32')
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('mask_kind', ['none', 'boolean', 'additive', 'causal'])
def test_independent_sdpa_oracle_matches_native_values_and_all_gradients(dtype, mask_kind):
    rng = np.random.RandomState(87)
    pairs = [pair(rng.randn(*shape), dtype) for shape in ((2, 3, 4, 5), (1, 3, 6, 5), (2, 1, 6, 7))]
    inputs, native_inputs = [p[0] for p in pairs], [p[1] for p in pairs]
    options, native_options = {}, {}
    if mask_kind in ('boolean', 'additive'):
        mask = rng.rand(4, 6) > .4
        mask[1] = False
        if mask_kind == 'boolean':
            options['attn_mask'], native_options['attn_mask'] = wasm.Tensor(mask), torch.tensor(mask)
        else:
            values = np.where(mask, rng.randn(4, 6), -np.inf).astype(dtype)
            a_mask, b_mask = pair(values, dtype)
            options['attn_mask'], native_options['attn_mask'] = a_mask, b_mask
            inputs.append(a_mask); native_inputs.append(b_mask)
    options['is_causal'] = native_options['is_causal'] = mask_kind == 'causal'
    options['scale'] = native_options['scale'] = .37
    a = wasm.scaled_dot_product_attention_oracle(*inputs[:3], **options)
    b = torch.nn.functional.scaled_dot_product_attention(*native_inputs[:3], **native_options)
    upstream = rng.randn(*a.shape).astype(dtype)
    a_grad = wasm.autograd_grad((a * wasm.Tensor(upstream)).sum(), inputs)
    b_grad = torch.autograd.grad((b * torch.tensor(upstream)).sum(), native_inputs)
    tolerance = 3e-6 if dtype == np.float32 else 1e-11
    np.testing.assert_allclose(a.numpy(), b.detach().numpy(), atol=tolerance, rtol=tolerance)
    for got, want in zip(a_grad, b_grad): np.testing.assert_allclose(got.numpy(), want.numpy(), atol=tolerance, rtol=tolerance)
    if mask_kind in ('boolean', 'additive'):
        assert np.count_nonzero(a.numpy()[:, :, 1]) == 0
        assert np.count_nonzero(a_grad[0].numpy()[:, :, 1]) == 0
    with pytest.raises(wasm.BrowserUnsupported): wasm.scaled_dot_product_attention_oracle(*inputs[:3], dropout_p=.1)
    with pytest.raises(wasm.BrowserUnsupported): wasm.scaled_dot_product_attention_oracle(*inputs[:3], enable_gqa=True)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_stateful_adamw_matches_native_with_decay_lr_changes_and_missing_gradients(dtype):
    rng = np.random.RandomState(707)
    pairs = [pair(rng.randn(2, 3), dtype), pair(rng.randn(4), dtype)]
    a = wasm.AdamW([p[0] for p in pairs], lr=.04, betas=(.65, .8), eps=1e-5, weight_decay=.13)
    b = torch.optim.AdamW([p[1] for p in pairs], lr=.04, betas=(.65, .8), eps=1e-5, weight_decay=.13)
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    for step in range(1, 14):
        for optimizer in (a, b):
            optimizer.zero_grad(set_to_none=bool(step % 2))
            optimizer.param_groups[0]['lr'] = .04 / step
        for index, (actual, native) in enumerate(pairs):
            gradient = rng.randn(*actual.shape).astype(dtype)
            actual.grad, native.grad = wasm.Tensor(gradient.copy()), torch.tensor(gradient.copy())
            if index == 1 and step % 3:
                actual.grad = native.grad = None
        a.step(); b.step()
        for actual, native in pairs:
            np.testing.assert_allclose(actual.numpy(), native.detach().numpy(), atol=tolerance, rtol=tolerance)
            if native not in b.state:
                assert actual not in a.state
            else:
                assert a.state[actual]['step'].item() == b.state[native]['step'].item()
                for name in ('exp_avg', 'exp_avg_sq'):
                    np.testing.assert_allclose(a.state[actual][name].numpy(), b.state[native][name].numpy(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('queries,keys', [(0, 5), (3, 0)])
def test_sdpa_empty_sequence_values_and_gradients_match_native(queries, keys):
    rng = np.random.RandomState(930)
    pairs = [pair(rng.randn(*shape)) for shape in ((2, 3, queries, 4), (2, 3, keys, 4), (2, 3, keys, 6))]
    a = wasm.scaled_dot_product_attention_oracle(*[p[0] for p in pairs])
    b = torch.nn.functional.scaled_dot_product_attention(*[p[1] for p in pairs])
    np.testing.assert_array_equal(a.numpy(), b.detach().numpy())
    a_grad = wasm.autograd_grad(a.sum(), [p[0] for p in pairs])
    b_grad = torch.autograd.grad(b.sum(), [p[1] for p in pairs])
    for got, want in zip(a_grad, b_grad): np.testing.assert_array_equal(got.numpy(), want.numpy())


def test_unimplemented_optimizer_interfaces_explicitly_request_server_fallback():
    x, _ = pair([1., 2.])
    for cls in (wasm.Adam, wasm.AdamW):
        optimizer = cls([x])
        with pytest.raises(wasm.BrowserUnsupported): optimizer.state_dict()
        with pytest.raises(wasm.BrowserUnsupported): optimizer.add_param_group({'params': [x]})
    with pytest.raises(wasm.BrowserUnsupported): wasm.AdamW([{'params': [x]}])
    with pytest.raises(wasm.BrowserUnsupported): wasm.AdamW([x], amsgrad=True)


def test_pdist_values_and_gradients_against_native():
    rng=np.random.RandomState(74);x,xx=pair(rng.randn(6,4))
    actual=wasm.pdist(x);expected=torch.pdist(xx)
    np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),atol=1e-12,rtol=1e-12)
    actual.sum().backward();expected.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(),xx.grad.numpy(),atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('axis', [0, -1])
def test_cumprod_zero_safe_values_and_gradients_against_native(dtype, axis):
    values = np.array([[.2, 0., -.3, 2.], [0., .7, 0., .5], [-.6, .1, .5, 0.]], dtype=dtype)
    x, xx = pair(values, dtype)
    actual, expected = x.cumprod(axis), xx.cumprod(axis)
    weight = np.random.RandomState(91).randn(*values.shape).astype(dtype)
    (actual * wasm.Tensor(weight)).sum().backward(); (expected * torch.tensor(weight)).sum().backward()
    tolerance = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(x.grad.numpy(), xx.grad.numpy(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('shape', [(), (2, 0)])
def test_cumprod_scalar_and_empty_inputs(shape):
    x, xx = pair(np.ones(shape))
    actual, expected = x.cumprod(-1), xx.cumprod(-1)
    actual.sum().backward(); expected.sum().backward()
    np.testing.assert_array_equal(actual.numpy(), expected.detach().numpy())
    np.testing.assert_array_equal(x.grad.numpy(), xx.grad.numpy())


@pytest.mark.parametrize('maximum', [False, True])
def test_elementwise_extrema_ties_nan_broadcast_and_gradient_sharing(maximum):
    a, aa = pair([[.2, 1., -1., np.nan], [2., 1., 0., .5]])
    b, bb = pair([.2, .5, 0., 1.])
    actual = wasm.elementwise_extreme(a, b, maximum)
    expected = (torch.maximum if maximum else torch.minimum)(aa, bb)
    actual.sum().backward(); expected.sum().backward()
    np.testing.assert_array_equal(actual.numpy(), expected.detach().numpy())
    np.testing.assert_array_equal(a.grad.numpy(), aa.grad.numpy())
    np.testing.assert_array_equal(b.grad.numpy(), bb.grad.numpy())


def test_ppo_clipped_objective_gradients_match_independent_native_expression():
    new, native_new = pair(np.log([.5, .9, 1., 1.1, 1.5, .5, 1.5]))
    old, native_old = pair(np.zeros(7)); adv, native_adv = pair([1., -2., 3., -1., 2., -1., -2.])
    ratio = (new - old.detach()).exp(); native_ratio = (native_new - native_old.detach()).exp()
    actual = -wasm.elementwise_extreme(ratio * adv.detach(), ratio.clamp(.8, 1.2) * adv.detach()).mean()
    expected = -torch.minimum(native_ratio * native_adv.detach(), native_ratio.clamp(.8, 1.2) * native_adv.detach()).mean()
    actual.backward(); expected.backward()
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(new.grad.numpy(), native_new.grad.numpy(), atol=1e-12)
    assert old.grad is adv.grad is None and native_old.grad is native_adv.grad is None


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('seed', [7, 42, 2026])
def test_multinomial_and_permutation_match_native_mixed_rng_stream(dtype, seed):
    probabilities = np.array([[0., .1, .3, .6], [.7, 0., .2, .1]], dtype=dtype).T.copy().T
    generator = wasm.Generator().manual_seed(seed); native = torch.Generator().manual_seed(seed)
    for _ in range(3):
        actual = wasm.multinomial(wasm.Tensor(probabilities), 2000, replacement=True, generator=generator)
        expected = torch.multinomial(torch.tensor(probabilities), 2000, replacement=True, generator=native)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        for count in (0, 1, 17, 256):
            np.testing.assert_array_equal(wasm.randperm(count, generator=generator).numpy(), torch.randperm(count, generator=native).numpy())
    np.testing.assert_array_equal(wasm.create('rand', 100, generator=generator, dtype=dtype).numpy(),
                                  torch.rand(100, generator=native, dtype=torch.float64 if dtype == np.float64 else torch.float32).numpy())
    with pytest.raises(wasm.BrowserUnsupported): wasm.multinomial(wasm.Tensor(probabilities), 2, replacement=False)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('columns', [None, 2])
def test_linear_solve_values_and_analytic_gradients_match_native(dtype, columns):
    rng = np.random.RandomState(173)
    values = rng.randn(4, 4); values = values.T @ values + np.eye(4)
    a, aa = pair(values, dtype); b, bb = pair(rng.randn(*((4,) if columns is None else (4, columns))), dtype)
    actual, expected = wasm.solve(a, b), torch.linalg.solve(aa, bb)
    actual.square().sum().backward(); expected.square().sum().backward()
    tolerance = 2e-6 if dtype == np.float32 else 1e-11
    for got, want in [(actual, expected.detach()), (a.grad, aa.grad), (b.grad, bb.grad)]:
        np.testing.assert_allclose(got.numpy(), want.numpy(), atol=tolerance, rtol=tolerance)


def test_split_repeat_interleave_and_integer_cast_contracts():
    x, xx = pair(np.arange(14).reshape(2, 7))
    for sections in (3, [2, 0, 5]):
        got, want = x.split(sections, dim=-1), xx.split(sections, dim=-1)
        a = sum((chunk * (i+1)).sum() for i, chunk in enumerate(got))
        b = sum((chunk * (i+1)).sum() for i, chunk in enumerate(want))
        np.testing.assert_allclose(wasm.autograd_grad(a, x)[0].numpy(), torch.autograd.grad(b, xx)[0].numpy(), atol=1e-12)
    counts = np.array([0, 1, 2, 0, 1, 2, 3])
    a = x.repeat_interleave(wasm.Tensor(counts), dim=-1)
    b = xx.repeat_interleave(torch.tensor(counts), dim=-1)
    a.sum().backward(); b.sum().backward()
    np.testing.assert_array_equal(a.numpy(), b.detach().numpy())
    np.testing.assert_array_equal(x.grad.numpy(), xx.grad.numpy())
    repeated, native_repeated = x.repeat_interleave(2, dim=-1), xx.repeat_interleave(2, dim=-1)
    np.testing.assert_array_equal(repeated.numpy(), native_repeated.detach().numpy())
    np.testing.assert_array_equal(wasm.autograd_grad(repeated.sum(), x)[0].numpy(), torch.autograd.grad(native_repeated.sum(), xx)[0].numpy())
    assert not x.long().requires_grad and not x.bool().requires_grad
    target = wasm.Tensor(np.zeros(2))
    with pytest.raises(wasm.BrowserUnsupported): target[0] = x[0, 0]


def test_direct_and_nested_module_parameters_and_forward_gradients_match_native():
    def build(module_type, parameter_type, make):
        class Layer(module_type):
            def __init__(self, scale):
                super().__init__(); self.weight = parameter_type(make(np.eye(3) * scale))
            def forward(self, x): return x @ self.weight
        class Model(module_type):
            def __init__(self):
                super().__init__(); self.left = Layer(.5); self.right = Layer(2.)
                self.bias = parameter_type(make(np.array([.1, -.2, .4])))
                self.alias = self.left.weight
            def forward(self, x): return self.left(x) + self.right(x) + self.bias
        return Model()
    model = build(wasm.Module, wasm.Parameter, wasm.Tensor)
    native = build(torch.nn.Module, torch.nn.Parameter, torch.tensor)
    params, native_params = list(model.parameters()), list(native.parameters())
    assert len(params) == len(native_params) == 3
    assert len(list(model.parameters(recurse=False))) == 2
    x, xx = pair(np.arange(6).reshape(2, 3))
    actual, expected = model(x), native(xx)
    actual.square().sum().backward(); expected.square().sum().backward()
    for got, want in zip([x, *params], [xx, *native_params]):
        np.testing.assert_allclose(got.numpy(), want.detach().numpy(), atol=1e-12)
        np.testing.assert_allclose(got.grad.numpy(), want.grad.numpy(), atol=1e-12)
    state, native_state = model.state_dict(), native.state_dict()
    assert list(state) == list(native_state)
    for name in state:
        np.testing.assert_array_equal(state[name].numpy(), native_state[name].numpy())
