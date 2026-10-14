import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.rope.experiment import run


def scalar_oracle(x, positions, base):
    """Independent Python scalar rotations, with no call to the reference."""
    rows = []
    for batch in x.tolist():
        heads = []
        for head in batch:
            tokens = []
            for p, token in zip(positions.tolist(), head):
                output = []
                for pair in range(len(token) // 2):
                    angle = p * math.pow(base, -2 * pair / len(token))
                    a, b = token[2 * pair:2 * pair + 2]
                    output.extend([a * math.cos(angle) - b * math.sin(angle),
                                   a * math.sin(angle) + b * math.cos(angle)])
                tokens.append(output)
            heads.append(tokens)
        rows.append(heads)
    return torch.tensor(rows, dtype=x.dtype, device=x.device)


@pytest.mark.t0
def test_layout_precision_empty_axes_and_input_preservation(impl):
    """RoPE only rotates feature pairs; noncontiguous storage and empty batch/token axes are valid."""
    for dtype in (torch.float32, torch.float64):
        for batch, length in ((2, 3), (0, 3), (2, 0)):
            x = torch.randn(batch, 2, length, 12, dtype=dtype)[..., ::2]
            positions = torch.arange(length, dtype=torch.int64) * 3 - 2
            before, before_positions = x.clone(), positions.clone()
            result = impl.apply_rope(x, positions)
            assert result.shape == x.shape and result.dtype == x.dtype and result.device == x.device
            close(x, before); close(positions, before_positions)
            assert torch.isfinite(result).all()


@pytest.mark.t1
def test_hand_computed_quarter_turns_and_zero_position(impl):
    """The paper's 2D positive-angle matrix sends (a,b) to (-b,a) at pi/2."""
    x = torch.tensor([[[[2., 3.], [4., -1.], [-2., 5.]]]], dtype=torch.float64)
    p = torch.tensor([math.pi / 2, -math.pi / 2, 0.], dtype=x.dtype)
    expected = torch.tensor([[[[-3., 2.], [-1., -4.], [-2., 5.]]]], dtype=x.dtype)
    close(impl.apply_rope(x, p), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_independent_scalar_frequencies_and_arbitrary_positions(impl):
    """Every adjacent pair has its own base**(-2*i/D) frequency; positions need not be consecutive."""
    for width, base in ((4, 10000.), (6, 17.), (10, 257.)):
        x = torch.randn(2, 3, 4, width, dtype=torch.float64)
        p = torch.tensor([-7., 0., 3., 19.], dtype=x.dtype)
        close(impl.apply_rope(x, p, base), scalar_oracle(x, p, base), atol=1e-12, rtol=1e-11)


@pytest.mark.t2
def test_norm_inverse_and_rotation_composition(impl):
    """Orthogonal planar rotations preserve each pair's norm and compose by adding positions."""
    x = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    p, delta = torch.tensor([-5., 1., 2., 8., 31.]), torch.tensor([2., -3., 7., 0., 9.])
    y = impl.apply_rope(x, p, 997.)
    close(y.reshape(2, 3, 5, 4, 2).square().sum(-1), x.reshape(2, 3, 5, 4, 2).square().sum(-1), atol=1e-11, rtol=1e-11)
    close(impl.apply_rope(y, -p, 997.), x, atol=1e-11, rtol=1e-11)
    close(impl.apply_rope(y, delta, 997.), impl.apply_rope(x, p + delta, 997.), atol=1e-11, rtol=1e-11)


@pytest.mark.t2
def test_relative_inner_product_and_joint_translation(impl):
    """<R(m)q,R(n)k> equals <q,R(n-m)k>; jointly shifting both positions leaves logits unchanged."""
    q, k = torch.randn(2, 3, 1, 6, dtype=torch.float64), torch.randn(2, 3, 1, 6, dtype=torch.float64)
    m, n = torch.tensor([7.]), torch.tensor([-3.])
    left = (impl.apply_rope(q, m, 31.) * impl.apply_rope(k, n, 31.)).sum(-1)
    expected = (q * scalar_oracle(k, n - m, 31.)).sum(-1)
    close(left, expected, atol=1e-11, rtol=1e-11)
    shifted = (impl.apply_rope(q, m + 101, 31.) * impl.apply_rope(k, n + 101, 31.)).sum(-1)
    close(shifted, left, atol=1e-11, rtol=1e-11)


@pytest.mark.t2
def test_batch_head_broadcast_and_token_reordering(impl):
    """One position vector broadcasts only over B/H; jointly reordering tokens and positions commutes."""
    x = torch.randn(2, 3, 4, 6, dtype=torch.float64)
    p, order = torch.tensor([5., -2., 0., 11.]), torch.tensor([3, 1, 0, 2])
    y = impl.apply_rope(x, p)
    close(impl.apply_rope(x[:, :, order], p[order]), y[:, :, order], atol=1e-12, rtol=1e-12)
    for b in range(2):
        for h in range(3):
            close(y[b:b+1, h:h+1], impl.apply_rope(x[b:b+1, h:h+1], p), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_independent_analytical_input_and_position_gradients(impl):
    """The input VJP is inverse rotation; position VJPs sum theta*J*R(x) contributions over batch/heads/pairs."""
    x = torch.randn(2, 3, 3, 6, dtype=torch.float64, requires_grad=True)
    p = torch.tensor([-2.5, 0., 4.25], dtype=x.dtype, requires_grad=True)
    upstream = torch.randn_like(x)
    output = impl.apply_rope(x, p, 23.)
    dx, dp = torch.autograd.grad((output * upstream).sum(), (x, p), allow_unused=True)
    assert dx is not None and dp is not None, 'Neither features nor continuous positions may be detached'
    close(dx, scalar_oracle(upstream, -p.detach(), 23.), atol=1e-11, rtol=1e-11)
    expected = []
    values, gradients = x.detach().tolist(), upstream.tolist()
    for t, position in enumerate(p.detach().tolist()):
        total = 0.
        for b in range(2):
            for h in range(3):
                for pair in range(3):
                    theta = math.pow(23., -2 * pair / 6)
                    a, c = values[b][h][t][2 * pair:2 * pair + 2]
                    u, v = gradients[b][h][t][2 * pair:2 * pair + 2]
                    angle = position * theta
                    total += theta * (u * (-a * math.sin(angle) - c * math.cos(angle)) + v * (a * math.cos(angle) - c * math.sin(angle)))
        expected.append(total)
    close(dp, torch.tensor(expected, dtype=x.dtype), atol=1e-11, rtol=1e-11)


@pytest.mark.t2
def test_small_numerical_jacobians(impl):
    """Independent finite differences validate both feature and position derivatives."""
    x = torch.randn(1, 1, 2, 4, dtype=torch.float64, requires_grad=True)
    p = torch.tensor([-1.5, 3.25], dtype=x.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: impl.apply_rope(a, b, 19.), (x, p), eps=1e-6, atol=1e-6, rtol=1e-5, raise_exception=False)


@pytest.mark.t2
def test_invalid_dimensions_positions_and_base(impl):
    """Reject ambiguous feature pairing and mismatched token positions instead of silently truncating/broadcasting."""
    x, p = torch.randn(2, 3, 4, 6), torch.arange(4)
    for bad in (0, 1, -2, float('nan'), float('inf'), True, False, '10', None):
        with pytest.raises(ValueError): impl.apply_rope(x, p, bad)
    for features, positions in ((x[0], p), (x[..., :5], p), (x[..., :0], p), (x, p[:3]), (x, p[None])):
        with pytest.raises(ValueError): impl.apply_rope(features, positions)


@pytest.mark.t3
def test_relative_logits_and_cached_decoding_experiment(impl, artifact):
    """The submitted rotation drives every logit and cache chunk; absolute offsets must agree with a scalar oracle."""
    result = run(impl.apply_rope)
    artifact('rope_relative_position_experiment', result)
    metrics = result['metrics']
    assert metrics['maximum_scalar_error'] < 1e-10
    assert metrics['joint_shift_error'] < 1e-10
    assert metrics['correct_cache_logits_error'] < 1e-10
    assert metrics['reset_cache_logits_gap'] > .1
    assert metrics['maximum_pair_norm_error'] < 1e-10
