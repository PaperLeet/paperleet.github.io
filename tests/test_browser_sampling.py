"""Independent native parity for diffusion browser dependencies."""
import copy

import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size', [0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 4000])
def test_cpu_normal_samples_and_following_uniform_stream_match_native(dtype, size):
    generator, native = wasm.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    actual = wasm.create('randn', size, dtype=dtype, generator=generator)
    expected = torch.randn(size, dtype=torch.float32 if dtype == np.float32 else torch.float64, generator=native)
    # float32 SIMD transcendentals differ across CPU architectures. Linux AVX
    # native values differ by up to 1.2e-6 near zero; uniform stream state
    # below remains an exact equality check.
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=3e-6 if dtype == np.float32 else 1e-13, atol=2e-6 if dtype == np.float32 else 1e-14)
    following = wasm.create('rand', 10, dtype=np.float64, generator=generator)
    np.testing.assert_array_equal(following.numpy(), torch.rand(10, dtype=torch.float64, generator=native).numpy())


def test_normal_scalar_cache_survives_mixed_dtypes_uniform_draws_and_vector_tail_overwrite():
    generator, native = wasm.Generator().manual_seed(93), torch.Generator().manual_seed(93)
    for size, dtype in [(3, np.float32), (17, np.float64), (1, np.float64), (31, np.float32), (7, np.float64), (16, np.float64), (1, np.float32)]:
        actual = wasm.create('randn', size, generator=generator, dtype=dtype)
        expected = torch.randn(size, generator=native, dtype=torch.float32 if dtype == np.float32 else torch.float64)
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=4e-6, atol=1e-7)
        np.testing.assert_array_equal(wasm.create('rand', 3, generator=generator, dtype=np.float64).numpy(), torch.rand(3, generator=native, dtype=torch.float64).numpy())


def test_fork_rng_restores_pending_normal_pair_after_exception():
    before, normal = np.random.get_state(), wasm._default_generator.normal
    try:
        wasm._default_generator.manual_seed(45)
        wasm.create('randn', 1)
        state, pending = np.random.get_state(), wasm._default_generator.normal
        expected = wasm.create('randn', 3).numpy()
        np.random.set_state(state); wasm._default_generator.normal = pending
        with pytest.raises(ValueError), wasm.fork_rng():
            wasm._default_generator.manual_seed(42)
            wasm.create('randn', 1)
            raise ValueError('fixture')
        np.testing.assert_array_equal(wasm.create('randn', 3).numpy(), expected)
    finally:
        np.random.set_state(before); wasm._default_generator.normal = normal


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape1,shape2', [((4, 3), (5, 3)), ((2, 1, 4, 3), (1, 3, 5, 3)), ((0, 3), (4, 3)), ((4, 0), (3, 0))])
def test_cdist_values_and_broadcast_vjps_match_native(dtype, shape1, shape2):
    rng = np.random.RandomState(77)
    data, other = rng.randn(*shape1).astype(dtype), rng.randn(*shape2).astype(dtype)
    x, y = wasm.Tensor(data, requires_grad=True), wasm.Tensor(other, requires_grad=True)
    xx, yy = torch.tensor(data, requires_grad=True), torch.tensor(other, requires_grad=True)
    actual, expected = wasm.cdist(x, y), torch.cdist(xx, yy, compute_mode='donot_use_mm_for_euclid_dist')
    upstream = rng.randn(*actual.shape).astype(dtype)
    (actual * wasm.Tensor(upstream)).sum().backward(); (expected * torch.tensor(upstream)).sum().backward()
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    for a, b in [(actual, expected.detach()), (x.grad, xx.grad), (y.grad, yy.grad)]:
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cdist_shared_inputs_ties_strides_zero_gradients_and_modes(dtype):
    data = np.asarray([[1., 2., 3.], [1., 2., 3.], [-1., .5, .2]], dtype=dtype).T
    x, xx = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    for mode in ('use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist'):
        np.testing.assert_allclose(wasm.cdist(x, x, compute_mode=mode).numpy(), torch.cdist(xx, xx, compute_mode=mode).detach().numpy(), atol=2e-3, rtol=3e-6)
    got, want = wasm.cdist(x, x), torch.cdist(xx, xx, compute_mode='donot_use_mm_for_euclid_dist')
    got.sum().backward(); want.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), xx.grad.numpy(), atol=1e-6, rtol=3e-6)
    saved = wasm.cdist(x, x).sum()
    with wasm.no_grad(): x.detach().fill_(1.)
    with pytest.raises(wasm.BrowserUnsupported): saved.backward()
    with pytest.raises(wasm.BrowserUnsupported): wasm.cdist(x, x, p=1)
    with pytest.raises(wasm.BrowserUnsupported): wasm.cdist(x, x, compute_mode='unknown')
    with pytest.raises(RuntimeError): wasm.cdist(x, wasm.Tensor(np.zeros((2, 4), dtype=dtype)))
    # Bound memory before constructing the broadcast pairwise allocation.
    with pytest.raises(wasm.BrowserUnsupported): wasm.cdist(wasm.Tensor(np.zeros((10000, 3), dtype=dtype)), wasm.Tensor(np.zeros((10000, 3), dtype=dtype)))
    with pytest.raises(wasm.BrowserUnsupported): wasm.cdist(wasm.Tensor(np.empty((10000, 0), dtype=dtype)), wasm.Tensor(np.empty((10000, 0), dtype=dtype)))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('eps', [None, 0., .1, .5])
def test_logit_values_and_gradients_match_native_at_clamped_boundaries(dtype, eps):
    data = np.asarray([-.1, 0., .05, .1, .4, .5, .9, .95, 1., 1.1, np.nan], dtype=dtype)
    x, xx = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        actual, expected = x.logit(eps), xx.logit(eps)
        actual.sum().backward(); expected.sum().backward()
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=1e-6, rtol=3e-6, equal_nan=True)
    np.testing.assert_allclose(x.grad.numpy(), xx.grad.numpy(), atol=1e-5, rtol=3e-6, equal_nan=True)
    with pytest.raises(wasm.BrowserUnsupported): x.logit(-.1)


def test_rng_fork_restores_after_nesting_exceptions_and_disabled_contexts():
    original = np.random.get_state()
    try:
        np.random.seed(871)
        expected = wasm.create('rand', 10).numpy()
        np.random.seed(871)
        with pytest.raises(ValueError), wasm.fork_rng():
            np.random.seed(42)
            with wasm.fork_rng(devices=[]):
                np.random.seed(93); wasm.create('randn', 30)
            raise ValueError('fixture')
        np.testing.assert_array_equal(wasm.create('rand', 10).numpy(), expected)
        np.random.seed(871)
        with wasm.fork_rng(enabled=False): wasm.create('rand', 10)
        second = wasm.create('rand', 10).numpy()
        np.random.seed(871); wasm.create('rand', 10)
        np.testing.assert_array_equal(wasm.create('rand', 10).numpy(), second)
        with pytest.raises(wasm.BrowserUnsupported), wasm.fork_rng(devices=[0]): pass
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(871)
            np.testing.assert_array_equal(expected, torch.rand(10).numpy())
    finally: np.random.set_state(original)


def test_registered_buffers_stay_outside_parameters_and_copy_independently():
    actual, expected = wasm.Module(), torch.nn.Module()
    actual.weight = wasm.Parameter(wasm.Tensor([2.], dtype=np.float32))
    expected.weight = torch.nn.Parameter(torch.tensor([2.]))
    actual.register_buffer('frequency', wasm.Tensor([1., 3.], dtype=np.float32))
    expected.register_buffer('frequency', torch.tensor([1., 3.]))
    actual.register_buffer('extra', None, persistent=False)
    expected.register_buffer('extra', None, persistent=False)
    assert len(list(actual.parameters())) == len(list(expected.parameters())) == 1
    np.testing.assert_array_equal(actual.frequency.numpy(), expected.frequency.numpy())
    cloned = copy.deepcopy(actual)
    cloned.frequency.fill_(0.)
    np.testing.assert_array_equal(actual.frequency.numpy(), expected.frequency.numpy())
    actual.requires_grad_(False)
    assert not actual.weight.requires_grad and not actual.frequency.requires_grad
    for name in ('', 'contains.dot'):
        with pytest.raises(ValueError): actual.register_buffer(name, None)
    with pytest.raises(KeyError): actual.register_buffer('weight', None)
    with pytest.raises(TypeError): actual.register_buffer('bad', [1.])
