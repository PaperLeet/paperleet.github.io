import numpy as np
import pytest
from torchwasm import as_torch, torch

def test_jacobian_supports_ordinary_torch_import_alias():
    with as_torch() as alias:
        from torch.autograd.functional import jacobian
        assert jacobian is alias.autograd.functional.jacobian
        actual = jacobian(lambda a: a.square(), alias.tensor([2., 3.]))
        assert actual.tolist() == [[4., 0.], [0., 6.]]

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_strictness_checks_each_output_against_each_input(dtype):
    x = torch.tensor([.2, .4], dtype=dtype)
    y = torch.tensor([.6, .8], dtype=dtype)
    function = lambda a, b: (a.square(), b.sin())
    result = torch.autograd.functional.jacobian(function, (x, y))
    np.testing.assert_allclose(result[0][0].numpy(), np.diag([.4, .8]), rtol=1e-6)
    np.testing.assert_array_equal(result[0][1].numpy(), np.zeros((2, 2)))
    np.testing.assert_array_equal(result[1][0].numpy(), np.zeros((2, 2)))
    np.testing.assert_allclose(result[1][1].numpy(), np.diag(np.cos([.6, .8])), rtol=1e-6)
    with pytest.raises(RuntimeError):
        torch.autograd.functional.jacobian(function, (x, y), strict=True)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_repeated_arguments_and_captured_original_are_distinct(dtype):
    x = torch.tensor([.2, .4], dtype=dtype, requires_grad=True)
    buffer = torch.tensor([9., 8.], dtype=dtype)
    x.grad = buffer
    hooks = []
    handle = x.register_hook(lambda grad: hooks.append(grad))
    repeated = torch.autograd.functional.jacobian(lambda a, b: a.square() + b.sin(), (x, x))
    np.testing.assert_allclose(repeated[0].numpy(), np.diag([.4, .8]), rtol=1e-6)
    np.testing.assert_allclose(repeated[1].numpy(), np.diag(np.cos([.2, .4])), rtol=1e-6)
    calls = []
    def callback(a):
        calls.append(1)
        return a.sin() + x.square()
    result = torch.autograd.functional.jacobian(callback, x)
    np.testing.assert_allclose(result.numpy(), np.diag(np.cos([.2, .4])), rtol=1e-6)
    assert calls == [1] and not hooks and x.grad is buffer
    assert x.grad.tolist() == [9., 8.] and x.requires_grad
    handle.remove()

def test_empty_input_tensor_and_empty_output_follow_native_boundaries():
    empty = torch.empty((0, 2), dtype=torch.float64)
    result = torch.autograd.functional.jacobian(lambda a: a.sum(), empty)
    assert result.shape == (0, 2) and result.dtype == torch.float64 and not result.requires_grad
    with pytest.raises(RuntimeError):
        torch.autograd.functional.jacobian(lambda a: a[:0], torch.ones(2))

@pytest.mark.parametrize('export', [lambda a: a.numpy(), np.asarray])
def test_implicit_numpy_export_rejects_without_leaking_transform_state(export):
    x = torch.ones(2, requires_grad=True)
    with pytest.raises(RuntimeError):
        torch.autograd.functional.jacobian(lambda a: torch.from_numpy(export(a)), x)
    assert x.numpy().tolist() == [1., 1.] and x.grad is None

@pytest.mark.parametrize('mode', ['detach', 'force', 'no_grad'])
def test_explicit_numpy_disconnection_has_zero_jacobian_or_strict_error(mode):
    def callback(a):
        if mode == 'detach': return torch.from_numpy(a.detach().numpy())
        if mode == 'force': return torch.from_numpy(a.numpy(force=True))
        with torch.no_grad(): return torch.from_numpy(a.numpy())
    x = torch.tensor([.2, .4])
    np.testing.assert_array_equal(torch.autograd.functional.jacobian(callback, x).numpy(), np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        torch.autograd.functional.jacobian(callback, x, strict=True)

def test_complex_jacobian_inputs_reject_before_callback():
    calls = []
    with pytest.raises(RuntimeError):
        torch.autograd.functional.jacobian(lambda a: calls.append(a), torch.tensor([1+2j], dtype=torch.complex64))
    assert not calls
