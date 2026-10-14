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
