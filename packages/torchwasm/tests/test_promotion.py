"""Independent NumPy-only selection/join workflows, also run in real WASM."""
import gc
import warnings
import weakref

import numpy as np
import pytest

from torchwasm import as_torch, torch, UnsupportedOperation


def test_tensor_scalar_categories_and_native_keyword_forms():
    a = torch.ones(2, dtype=torch.float32); b = torch.tensor(2., dtype=torch.float64)
    assert torch.result_type(tensor=a, other=b) == torch.float32
    assert torch.result_type(tensor=a, other=2.) == torch.float32
    assert torch.result_type(scalar=2., tensor=a) == torch.float32
    assert torch.result_type(scalar1=2, scalar2=.5) == torch.float32
    assert torch.result_type(torch.ones(2, dtype=torch.int16), b) == torch.float64
    assert torch.result_type(torch.tensor(1, dtype=torch.int16), 2) == torch.int16
    assert torch.result_type(torch.tensor(True), np.bool_(True)) == torch.float32
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert torch.result_type(a, np.complex64(1+.5j)) == torch.float32
    with pytest.raises(TypeError): torch.result_type(tensor1=a, tensor2=b)
    with pytest.raises(TypeError): torch.result_type(np.uint64(2**63), a)
    with pytest.raises(UnsupportedOperation): torch.result_type(torch.ones(2, dtype=torch.float16), 1+.5j)


def test_complex_tensor_join_values_and_aliases_are_preserved():
    real = torch.tensor([1., 2.]); complex_value = torch.tensor([3+.25j, 4+.5j])
    for name in ['cat', 'concat', 'concatenate']:
        result = getattr(torch, name)(tensors=[real, complex_value], dim=0)
        assert result.dtype == torch.complex64
        np.testing.assert_array_equal(result.numpy(), [1., 2., 3+.25j, 4+.5j])
        np.testing.assert_array_equal(getattr(torch, name)([real, complex_value], axis=0).numpy(), result.numpy())
        with pytest.raises(TypeError): getattr(torch, name)([real, complex_value], dim=0, axis=0)
    result = torch.stack([real, complex_value], dim=-1)
    assert result.dtype == torch.complex64
    np.testing.assert_array_equal(result.numpy(), [[1., 3+.25j], [2., 4+.5j]])
    result = real.where(condition=torch.tensor([True, False]), other=0)
    assert result.dtype == torch.float32
    np.testing.assert_array_equal(result.numpy(), [1., 0.])


def test_where_indices_condition_validation_and_empty_join():
    source = torch.tensor([[0., 2., 0.], [3., 0., 4.]], requires_grad=True)
    rows, columns = torch.where(source)
    assert rows.dtype == columns.dtype == torch.int64 and not rows.requires_grad
    np.testing.assert_array_equal(rows.numpy(), [0, 1, 1]); np.testing.assert_array_equal(columns.numpy(), [1, 0, 2])
    np.testing.assert_array_equal(torch.where(torch.tensor(True))[0].numpy(), [0])
    assert torch.where(torch.zeros(2, 0))[0].shape == (0,)
    with pytest.raises(TypeError): torch.where(torch.tensor(True), out=None)
    with pytest.raises(RuntimeError): torch.where(source, 1, 0)
    with pytest.raises(UnsupportedOperation): torch.where(torch.ones(2, dtype=torch.uint16))
    a = torch.empty(0, dtype=torch.float64, requires_grad=True)
    b = torch.ones(2, 3, requires_grad=True)
    result = torch.cat([a, b], dim=1)
    assert result.dtype == torch.float64 and result.shape == (2, 3)
    result.sum().backward(); assert a.grad.shape == (0,) and b.grad.dtype == torch.float32
    np.testing.assert_array_equal(b.grad.numpy(), np.ones((2, 3)))
    with pytest.raises(TypeError): torch.cat([b, b], dim=True)
    with pytest.raises(RuntimeError): torch.stack([a, b])


def test_mixed_dtype_broadcast_gradients_and_duplicate_join_edges():
    a = torch.tensor([[1.], [2.]], requires_grad=True)
    b = torch.tensor([[3., 4., 5.]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    y = torch.where(mask, a, b)
    assert y.dtype == torch.float64
    y.square().sum().backward()
    assert a.grad.dtype == torch.float32 and b.grad.dtype == torch.float64
    np.testing.assert_array_equal(a.grad.numpy(), [[4.], [4.]])
    np.testing.assert_array_equal(b.grad.numpy(), [[6., 8., 10.]])
    a = torch.tensor([1., 2.], requires_grad=True); b = torch.tensor([3., 4.], dtype=torch.float64, requires_grad=True)
    y = torch.stack([a, b, a]); y.square().sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), [4., 8.]); np.testing.assert_array_equal(b.grad.numpy(), [6., 8.])


def test_where_mask_lifetime_and_selective_mutation_guards():
    x = torch.tensor([2., 3.], requires_grad=True); mask = torch.tensor([True, False])
    a, b = x + 1, x * .5
    refs = weakref.ref(a), weakref.ref(b)
    y = torch.where(mask, a, b)
    del a, b; gc.collect(); assert all(ref() is None for ref in refs)
    y.sum().backward(retain_graph=True)
    np.testing.assert_array_equal(x.grad.numpy(), [1., .5])
    x.grad = None; y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [1., .5])
    with pytest.raises(RuntimeError): y.sum().backward()
    # Isolate where's mutation contract from upstream operators' saved state.
    a = torch.tensor([2., 3.], requires_grad=True); b = torch.tensor([5., 7.], requires_grad=True)
    y = torch.where(mask, a, b)
    with torch.no_grad(): a.detach()[0] = 99.; b.detach()[0] = 99.; y.detach()[0] = 9.
    y.sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), [1., 0.]); np.testing.assert_array_equal(b.grad.numpy(), [0., 1.])
    y = torch.where(mask, x, 0); mask[0] = False
    with pytest.raises(UnsupportedOperation): y.sum().backward()


def test_forward_reverse_over_forward_and_checked_scalar_paths():
    for name in ['where', 'cat', 'stack']:
        x = torch.tensor([1., 2.], requires_grad=True); direction = torch.tensor([.25, -.5])
        mask = torch.tensor([True, False])
        def function(value):
            a, b = value.square(), value * 3
            return torch.where(mask, a, b) if name == 'where' else getattr(torch, name)([a, b])
        y, dy = torch.func.jvp(function, (x,), (direction,))
        dy.sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), [.5, 0.] if name == 'where' else [.5, -1.])
        if name == 'where': np.testing.assert_array_equal(dy.numpy(), [.5, -1.5])
    with pytest.raises(RuntimeError):
        torch.func.jvp(lambda x: torch.where(mask, x, 1e300), (torch.ones(2),), (torch.ones(2),))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y, dy = torch.func.jvp(lambda x: torch.where(mask, x, np.complex64(1+.5j)),
                               (torch.ones(2),), (torch.ones(2),))
    assert y.dtype == torch.float32
    np.testing.assert_array_equal(y.numpy(), [1., 1.]); np.testing.assert_array_equal(dy.numpy(), [1., 0.])


def test_scalar_overflow_and_tensor_casts_keep_distinct_contracts():
    before = np.geterr()
    with np.errstate(all='raise'):
        mask = torch.tensor([True, False]); a = torch.ones(2)
        for number in [0., 1e-45, 1e-300, float('inf'), -float('inf'), float('nan')]:
            value = torch.where(mask, a, number)
            assert value.dtype == torch.float32
        with pytest.raises(RuntimeError): torch.where(mask, a, 1e300)
        result = torch.where(mask, a, torch.tensor(1e300, dtype=torch.float64))
        assert result.dtype == torch.float32 and np.isposinf(result.numpy()[1])
        with pytest.raises(OverflowError): torch.where(mask, a, 2**64)
        np.testing.assert_array_equal(torch.where(mask, torch.ones(2, dtype=torch.uint64), -1).numpy(), [1, 2**64-1])
        assert np.geterr() == dict.fromkeys(before, 'raise')
    assert np.geterr() == before


def test_unchanged_imports_two_layer_training_matches_raw_numpy():
    with as_torch():
        import torch as aliased
        from torch import nn
        first, second = nn.Linear(2, 2, bias=False), nn.Linear(2, 1, bias=False)
        w1 = np.array([[.5, -.25], [.75, .5]], dtype='f4'); w2 = np.array([[.25, -.5]], dtype='f4')
        x = np.array([[1., 2.], [3., -1.]], dtype='f4'); target = np.array([[.5], [-.25]], dtype='f4')
        mask = np.array([[True, False], [False, True]])
        with aliased.no_grad(): first.weight.copy_(aliased.tensor(w1)); second.weight.copy_(aliased.tensor(w2))
        optimizer = aliased.optim.SGD([*first.parameters(), *second.parameters()], lr=.05)
        for _ in range(6):
            hidden = x @ w1.T; selected = np.where(mask, hidden, np.zeros_like(hidden)); prediction = selected @ w2.T
            error = prediction - target; expected_loss = np.mean(error**2)
            dy = error * np.float32(2 / error.size)
            g2 = dy.T @ selected; dh = np.where(mask, dy @ w2, np.zeros_like(hidden)); g1 = dh.T @ x
            optimizer.zero_grad()
            selected_tensor = aliased.where(aliased.tensor(mask), first(aliased.tensor(x)), 0)
            assert selected_tensor.dtype == aliased.float32
            loss = (second(selected_tensor) - aliased.tensor(target)).square().mean()
            np.testing.assert_allclose(loss.item(), expected_loss, rtol=1e-6, atol=1e-7)
            loss.backward(); optimizer.step()
            w1 -= np.float32(.05) * g1; w2 -= np.float32(.05) * g2
            np.testing.assert_allclose(first.weight.numpy(), w1, atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(second.weight.numpy(), w2, atol=1e-7, rtol=1e-6)
