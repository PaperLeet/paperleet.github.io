"""Native parity at optimized indexing, shared SDPA VJP and Adam boundaries."""
import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('index', [
    (slice(None, None, 2), slice(1, 5)), (1, Ellipsis), (None, slice(None), 2),
    (slice(2, 2), slice(None)), (), (Ellipsis, None),
    [1, 1, 0], (np.array([0, 0, 2]), np.array([1, 1, 3])),
    np.array([True, False, True]), True, False,
])
def test_indexed_backward_keeps_native_accumulation_and_reuse(dtype, index):
    data = np.arange(18, dtype=dtype).reshape(3, 6) / 10
    actual = wasm.Tensor(data.T.copy().T, requires_grad=True)
    native = torch.tensor(data.T.copy()).T.requires_grad_()
    # Include non-contiguous input storage, repeated destinations, scalar bools,
    # empty selections, inserted axes and a shared leaf reached by two paths.
    got, want = actual[index], native[index]
    for coefficient in (1., -.3):
        actual.grad = native.grad = None
        ((got.square().sum() + actual.sum()) * coefficient).backward(retain_graph=True)
        ((want.square().sum() + native.sum()) * coefficient).backward(retain_graph=True)
        np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(actual.grad.numpy(), native.grad.numpy(), rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize('active', [(True, True, True, True), (True, False, False, False),
                                  (False, False, True, False), (False, False, False, True)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_attention_shared_pullback_handles_repeated_seeds_and_frozen_inputs(active, dtype):
    rng = np.random.default_rng(815)
    shapes = [(2, 3, 4, 5), (1, 3, 6, 5), (2, 1, 6, 7), (4, 6)]
    arrays = [rng.normal(size=shape).astype(dtype) for shape in shapes]
    arrays[-1][1] = -np.inf
    actual = [wasm.Tensor(data.copy(), requires_grad=flag) for data, flag in zip(arrays, active)]
    native = [torch.tensor(data, requires_grad=flag) for data, flag in zip(arrays, active)]
    output = wasm.scaled_dot_product_attention_oracle(*actual[:3], attn_mask=actual[3])
    expected = torch.nn.functional.scaled_dot_product_attention(*native[:3], attn_mask=native[3])
    assert output.dtype == expected.detach().numpy().dtype
    for _ in range(2):
        seed = rng.normal(size=output.shape).astype(dtype)
        got = wasm.autograd_grad(output, [p for p in actual if p.requires_grad], wasm.Tensor(seed), retain_graph=True)
        want = torch.autograd.grad(expected, [p for p in native if p.requires_grad], torch.tensor(seed), retain_graph=True)
        tolerance = 4e-6 if dtype == np.float32 else 2e-12
        np.testing.assert_allclose(output.numpy(), expected.detach().numpy(), rtol=tolerance, atol=tolerance)
        for a, b in zip(got, want):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('mutated', [0, 1, 2, 3])
def test_attention_retains_saved_operand_version_guards(mutated):
    operands = [wasm.Tensor(np.ones(shape), requires_grad=i == 2)
                for i, shape in enumerate([(1, 3, 2), (1, 3, 2), (1, 3, 2), (3, 3)])]
    output = wasm.scaled_dot_product_attention_oracle(*operands[:3], attn_mask=operands[3])
    with wasm.no_grad():
        operands[mutated].detach().fill_(.5)
    with pytest.raises(wasm.BrowserUnsupported, match='modified in-place'):
        output.sum().backward()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('mask_kind', ['none', 'boolean', 'shared_view'])
def test_attention_shared_query_key_value_and_mask_view_accumulate_like_native(dtype, mask_kind):
    rng = np.random.default_rng(819)
    data = rng.normal(size=(2, 3, 4)).astype(dtype)
    actual = wasm.Tensor(data.copy(), requires_grad=True)
    native = torch.tensor(data, requires_grad=True)
    options, native_options = {}, {}
    if mask_kind == 'shared_view':
        options['attn_mask'] = actual[0, :, :3]
        native_options['attn_mask'] = native[0, :, :3]
    elif mask_kind == 'boolean':
        mask = np.array([[True, True, False], [False, False, False], [True, False, True]])
        options['attn_mask'] = wasm.Tensor(mask)
        native_options['attn_mask'] = torch.tensor(mask)
    output = wasm.scaled_dot_product_attention_oracle(actual, actual, actual, **options)
    expected = torch.nn.functional.scaled_dot_product_attention(native, native, native, **native_options)
    for _ in range(2):
        actual.grad = native.grad = None
        seed = rng.normal(size=output.shape).astype(dtype)
        (output + output).backward(wasm.Tensor(seed), retain_graph=True)
        (expected + expected).backward(torch.tensor(seed), retain_graph=True)
        tolerance = 4e-6 if dtype == np.float32 else 2e-12
        np.testing.assert_allclose(actual.grad.numpy(), native.grad.numpy(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('weight_decay', [0., .13])
def test_adam_native_missing_gradient_steps_decay_and_grad_buffer_identity(dtype, weight_decay):
    rng = np.random.default_rng(816)
    arrays = [rng.normal(size=shape).astype(dtype) for shape in [(2, 3), (4,), (2,)]]
    actual = [wasm.Tensor(data.copy(), requires_grad=True) for data in arrays]
    native = [torch.tensor(data, requires_grad=True) for data in arrays]
    options = dict(lr=.04, betas=(.65, .8), eps=1e-5, weight_decay=weight_decay)
    optimizer, reference = wasm.Adam(actual, **options), torch.optim.Adam(native, **options)
    for step in range(12):
        for i, (a, b) in enumerate(zip(actual, native)):
            gradient = rng.normal(size=a.shape).astype(dtype)
            # The third parameter never gets a gradient; the second starts late
            # and intermittently skips updates while the first always trains.
            missing = i == 2 or i == 1 and (step < 3 or step % 2)
            a.grad = None if missing else wasm.Tensor(gradient.copy())
            b.grad = None if missing else torch.tensor(gradient)
        saved = [a.grad for a in actual]
        optimizer.step(); reference.step()
        tolerance = 3e-6 if dtype == np.float32 else 1e-12
        for a, b in zip(actual, native):
            np.testing.assert_allclose(a.numpy(), b.detach().numpy(), rtol=tolerance, atol=tolerance)
        optimizer.zero_grad(set_to_none=False); reference.zero_grad(set_to_none=False)
        for a, b, old in zip(actual, native, saved):
            assert a.grad is old
            assert (a.grad is None) == (b.grad is None)
            if old is not None:
                np.testing.assert_array_equal(old.numpy(), b.grad.numpy())


@pytest.mark.parametrize('options', [dict(lr=-1), dict(eps=-1), dict(betas=(1., .9)), dict(weight_decay=-1)])
def test_adam_rejects_invalid_hyperparameters(options):
    with pytest.raises(ValueError):
        wasm.Adam([wasm.Tensor([1.], requires_grad=True)], **options)
