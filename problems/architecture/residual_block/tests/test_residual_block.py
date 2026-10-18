import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.residual_block.experiment import run


@pytest.mark.t0
def test_contract(impl):
    """The two-layer branch and explicit projection return [B,D_out]."""
    x = torch.randn(4, 3, dtype=torch.float64)
    before = x.clone()
    y = impl.residual_block(x, torch.randn(3, 5, dtype=x.dtype), torch.randn(5, 2, dtype=x.dtype), torch.randn(3, 2, dtype=x.dtype))
    assert y.shape == (4, 2) and y.dtype == x.dtype and y.device == x.device
    close(x, before, rtol=0, atol=0)


@pytest.mark.t1
def test_scalar_oracle(impl):
    """Equation (1)/(2) is independently evaluated with scalar dot products."""
    x = torch.tensor([[1., -2.], [-3., 4.]], dtype=torch.float64)
    w1 = torch.tensor([[2., -1., .5], [1., 3., -2.]], dtype=x.dtype)
    w2 = torch.tensor([[1., -2.], [-.5, 1.], [2., -1.]], dtype=x.dtype)
    for projection in (None, torch.tensor([[.5, 2.], [-1., .25]], dtype=x.dtype)):
        expected = []
        for row in x.tolist():
            hidden = [max(0., sum(row[i] * w1[i, j].item() for i in range(2))) for j in range(3)]
            shortcut = row if projection is None else [sum(row[i] * projection[i, j].item() for i in range(2)) for j in range(2)]
            expected.append([max(0., sum(hidden[k] * w2[k, j].item() for k in range(3)) + shortcut[j]) for j in range(2)])
        close(impl.residual_block(x, w1, w2, projection), torch.tensor(expected, dtype=x.dtype), rtol=0, atol=1e-12)


@pytest.mark.t2
def test_zero_residual_is_post_activation_identity(impl):
    """Figure 2 applies ReLU after addition, so a zero branch returns ReLU(x)."""
    x = torch.tensor([[-2., .5, 3.]], dtype=torch.float64, requires_grad=True)
    y = impl.residual_block(x, torch.eye(3, dtype=x.dtype), torch.zeros(3, 3, dtype=x.dtype))
    close(y, torch.tensor([[0., .5, 3.]], dtype=x.dtype), rtol=0, atol=0)
    y.sum().backward()
    assert x.grad is not None, 'The shortcut must retain its input gradient'
    close(x.grad, torch.tensor([[0., 1., 1.]], dtype=x.dtype), rtol=0, atol=0)


@pytest.mark.t2
def test_projection_is_learned_shortcut(impl):
    """A square projection still replaces identity; every projection entry can learn."""
    x = torch.tensor([[1., 2.], [3., 1.]], dtype=torch.float64)
    p = torch.tensor([[2., .5], [.25, 3.]], dtype=x.dtype, requires_grad=True)
    y = impl.residual_block(x, torch.ones(2, 3, dtype=x.dtype), torch.zeros(3, 2, dtype=x.dtype), p)
    close(y, torch.tensor([[2.5, 6.5], [6.25, 4.5]], dtype=x.dtype), rtol=0, atol=0)
    y.sum().backward()
    assert p.grad is not None, 'The projection shortcut is trainable'
    close(p.grad, torch.tensor([[4., 4.], [3., 3.]], dtype=x.dtype), rtol=0, atol=0)


@pytest.mark.t2
def test_activation_follows_addition(impl):
    """A negative residual may cancel positive input before the final ReLU."""
    x = torch.tensor([[2., 1.]], dtype=torch.float64)
    close(impl.residual_block(x, torch.eye(2, dtype=x.dtype), -2 * torch.eye(2, dtype=x.dtype)), torch.zeros_like(x), rtol=0, atol=0)


@pytest.mark.t2
def test_all_branch_gradients(impl):
    """Away from ReLU kinks, dY/dX includes identity and both matrix factors."""
    x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64, requires_grad=True)
    a = torch.full((2, 3), .2, dtype=x.dtype, requires_grad=True)
    b = torch.full((3, 2), .3, dtype=x.dtype, requires_grad=True)
    y = impl.residual_block(x, a, b)
    y.sum().backward()
    for value, expected in [(x, torch.full_like(x, 1.36)),
                            (a, torch.tensor([[2.4] * 3, [3.6] * 3], dtype=x.dtype)),
                            (b, torch.full_like(b, 2.))]:
        assert value.grad is not None, 'Both residual layers and shortcut must preserve gradients'
        close(value.grad, expected, rtol=1e-10, atol=1e-10)
    inputs = tuple(v.detach().requires_grad_() for v in (x, a, b))
    assert torch.autograd.gradcheck(impl.residual_block, inputs)


@pytest.mark.t2
def test_strided_empty_and_float32(impl):
    """Matrix layout and an empty batch do not change the residual definition."""
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
    a, b = torch.eye(3), torch.eye(3)
    close(impl.residual_block(x, a, b), 2 * x, rtol=0, atol=0)
    assert impl.residual_block(x[:0], a, b).shape == (0, 3)


@pytest.mark.t2
def test_invalid_widths_do_not_broadcast(impl):
    """Equation (2) requires an explicit learned projection for changing widths."""
    with pytest.raises(ValueError):
        impl.residual_block(torch.ones(2, 1), torch.ones(1, 3), torch.ones(3, 2))
    with pytest.raises(ValueError):
        impl.residual_block(torch.ones(2, 2), torch.ones(2, 3), torch.ones(3, 2), torch.ones(1, 2))


@pytest.mark.t3
def test_nonlinear_residual_training(impl, artifact):
    """Four learner-defined blocks must learn a nonlinear correction in 80 actual updates."""
    result = run(impl.residual_block)
    artifact('residual_training', result)
    metrics = result['metrics']
    assert metrics['final_mse'] < .003, f"final MSE: expected < .003, got {metrics['final_mse']}"
    assert metrics['final_mse'] < .05 * metrics['initial_mse'], metrics
    assert metrics['learned_residual_norm'] > .01, metrics
