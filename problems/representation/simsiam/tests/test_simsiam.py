import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.simsiam.toy import fixture, scalar_loss
from problems.representation.simsiam.experiment import run


@pytest.mark.t0
def test_scalar_dtype_noncontiguous_inputs_and_no_mutation(impl):
    """Row-wise objectives preserve float dtype/device and do not change strided inputs."""
    for dtype in (torch.float32, torch.float64):
        values = tuple(x[:, ::2] for x in fixture(dtype))
        snapshots = tuple(x.clone() for x in values)
        result = impl.simsiam_loss(*values)
        expected = (scalar_loss(values[0].tolist(), values[3].tolist()) + scalar_loss(values[1].tolist(), values[2].tolist())) / 2
        assert result.shape == () and result.dtype == dtype and result.device == values[0].device
        close(result, torch.tensor(expected, dtype=dtype), atol=1e-6, rtol=1e-6)
        for value, snapshot in zip(values, snapshots):
            close(value, snapshot)


@pytest.mark.t0
def test_reject_invalid_shape_dtype_and_nonfinite_values(impl):
    """The provided contract rejects empty axes, broadcasting, mixed dtype and nonfinite vectors."""
    a, b, c, d = fixture()
    for x, y in ((a[:0], b[:0]), (a[:, :0], b[:, :0]), (a.flatten(), b.flatten()),
                 (a, b[:1]), (a, b.float()), (a.long(), b.long()), (a * float('nan'), b), (a, b * float('inf'))):
        with pytest.raises(ValueError):
            impl.negative_cosine_similarity(x, y)
    for values in ((a, b[:1], c, d), (a, b, c.float(), d), (a, b, c, d * float('nan'))):
        with pytest.raises(ValueError):
            impl.simsiam_loss(*values)


@pytest.mark.t1
def test_signed_endpoints_and_asymmetric_crossed_scalar_oracle(impl):
    """Identical unit directions give -1, opposite directions +1, and symmetric loss crosses the views."""
    a, b, c, d = fixture()
    close(impl.negative_cosine_similarity(a, a), torch.tensor(-1., dtype=a.dtype))
    close(impl.negative_cosine_similarity(a, -a), torch.tensor(1., dtype=a.dtype))
    first, second = scalar_loss(a.tolist(), d.tolist()), scalar_loss(b.tolist(), c.tolist())
    close(impl.negative_cosine_similarity(a, d), torch.tensor(first, dtype=a.dtype), atol=1e-12, rtol=1e-12)
    close(impl.simsiam_loss(a, b, c, d), torch.tensor((first + second) / 2, dtype=a.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_prediction_finite_difference_and_target_stop(impl):
    """Equation 3 stops the target while prediction derivatives agree with a fixed-target scalar oracle."""
    p, z, _, _ = fixture(); p = p.clone().requires_grad_(); z = z.clone().requires_grad_()
    loss = impl.negative_cosine_similarity(p, z)
    assert loss.requires_grad, 'The prediction path must remain differentiable'
    dp, dz = torch.autograd.grad(loss, (p, z), allow_unused=True)
    assert dp is not None and dz is None, 'Detach only the target branch'
    rows, target, eps = p.detach().tolist(), z.detach().tolist(), 1e-6
    numerical = []
    for row in rows:
        entries = []
        for j, old in enumerate(row):
            row[j] = old + eps; plus = scalar_loss(rows, target)
            row[j] = old - eps; minus = scalar_loss(rows, target)
            row[j] = old; entries.append((plus - minus) / (2 * eps))
        numerical.append(entries)
    close(dp, torch.tensor(numerical, dtype=p.dtype), atol=2e-9, rtol=2e-6)


@pytest.mark.t2
def test_symmetric_prediction_gradients_and_shared_encoder_paths(impl):
    """Each prediction receives half a directional derivative; shared z still learns through p=h(z)."""
    values = tuple(x.clone().requires_grad_() for x in fixture())
    loss = impl.simsiam_loss(*values)
    assert loss.requires_grad
    gradients = torch.autograd.grad(loss, values, allow_unused=True)
    assert gradients[0] is not None and gradients[1] is not None
    assert gradients[2] is None and gradients[3] is None
    a, b, c, d = fixture()
    for index, p, target in ((0, a, d), (1, b, c)):
        p = p.clone().requires_grad_()
        directional = torch.autograd.grad(impl.negative_cosine_similarity(p, target), p)[0]
        close(gradients[index], directional / 2, atol=1e-12, rtol=1e-12)
    z1, z2 = a.clone().requires_grad_(), b.clone().requires_grad_()
    weight = torch.tensor([[1., .3, -.2], [-.1, .7, .4], [.2, -.3, 1.1]], dtype=a.dtype)
    full = impl.simsiam_loss(z1 @ weight, z2 @ weight, z1, z2)
    got = torch.autograd.grad(full, (z1, z2))
    q1, q2 = a.clone().requires_grad_(), b.clone().requires_grad_()
    explicit = (impl.negative_cosine_similarity(q1 @ weight, b)
                + impl.negative_cosine_similarity(q2 @ weight, a)) / 2
    expected = torch.autograd.grad(explicit, (q1, q2))
    for actual, wanted in zip(got, expected):
        close(actual, wanted, atol=1e-12, rtol=1e-12)
        assert actual.abs().max().item() > 1e-4


@pytest.mark.t2
def test_pair_permutation_swap_and_positive_row_scale_invariance(impl):
    """Direction averaging and per-row normalization preserve view swaps and paired transformations."""
    a, b, c, d = fixture(); base = impl.simsiam_loss(a, b, c, d)
    order = torch.tensor([2, 0, 3, 1])
    close(impl.simsiam_loss(a[order], b[order], c[order], d[order]), base, atol=1e-12, rtol=1e-12)
    close(impl.simsiam_loss(b, a, d, c), base, atol=1e-12, rtol=1e-12)
    scales = torch.tensor([[.1], [7.], [1e3], [.03]], dtype=a.dtype)
    close(impl.simsiam_loss(a * scales, b / scales, c * scales * 2, d / scales), base, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_zero_near_zero_and_singleton_contract(impl):
    """The explicit 1e-12 norm clamp defines finite zero-vector behavior and preserves singleton batches."""
    p = torch.tensor([[0., 0.], [1e-14, -2e-14], [3., -4.]], dtype=torch.float64, requires_grad=True)
    z = torch.tensor([[2., -1.], [2e-14, 1e-14], [.2, .8]], dtype=p.dtype, requires_grad=True)
    got = impl.negative_cosine_similarity(p, z)
    close(got, torch.tensor(scalar_loss(p.detach().tolist(), z.detach().tolist()), dtype=p.dtype), atol=1e-12, rtol=1e-12)
    dp, dz = torch.autograd.grad(got, (p, z), allow_unused=True)
    assert dp is not None and bool(torch.isfinite(dp).all()) and dz is None
    close(dp[0], -z[0].detach() / z[0].detach().norm() / (3e-12), atol=1e-3, rtol=1e-12)
    close(impl.negative_cosine_similarity(p[2:], z[2:]), torch.tensor(scalar_loss(p.detach().tolist()[2:], z.detach().tolist()[2:]), dtype=p.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.t3
def test_real_sgd_aligns_predictions_without_moving_targets(impl, artifact):
    """80 real SGD updates improve crossed prediction alignment while stopped targets stay fixed."""
    result = run(impl.simsiam_loss); artifact('simsiam_training', result); metrics = result['metrics']
    assert metrics['updates'] == 80 and metrics['trainable_prediction_parameters'] == 24
    assert metrics['maximum_scalar_error'] < 1e-11
    assert metrics['target_movement'] == 0
    assert metrics['final_cosine'] > .98 and metrics['final_cosine'] > metrics['initial_cosine'] + .5
    assert metrics['final_loss'] < -.98
