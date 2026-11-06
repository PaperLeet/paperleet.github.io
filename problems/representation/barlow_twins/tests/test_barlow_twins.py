import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.barlow_twins.experiment import run


def oracle(a, b, eps):
    """Direct scalar covariance divided by the two population standard deviations."""
    n, d = len(a), len(a[0])
    ma = [sum(row[j] for row in a) / n for j in range(d)]
    mb = [sum(row[j] for row in b) / n for j in range(d)]
    va = [sum((row[j] - ma[j])**2 for row in a) / n for j in range(d)]
    vb = [sum((row[j] - mb[j])**2 for row in b) / n for j in range(d)]
    return [[sum((a[k][i]-ma[i])*(b[k][j]-mb[j]) for k in range(n)) /
             (n*math.sqrt(va[i]+eps)*math.sqrt(vb[j]+eps)) for j in range(d)] for i in range(d)]


def objective(a, b, lambd, eps):
    c = oracle(a, b, eps)
    return sum((c[i][j]-1)**2 if i == j else lambd*c[i][j]**2
               for i in range(len(c)) for j in range(len(c)))


def pair():
    x = torch.tensor([[1., .1, -1.], [2., -1., 3.], [-.5, 2., .7], [3., .5, 1.]], dtype=torch.float64)
    y = torch.tensor([[.2, 2., -3.], [1., -.5, 1.], [-1., .2, 1.5], [2., 1., .4]], dtype=torch.float64)
    return x, y


@pytest.mark.t0
def test_shapes_dtypes_strides_and_no_mutation(impl):
    """Feature correlation is [D,D]; the objective is scalar and preserves input storage."""
    for dtype in (torch.float32, torch.float64):
        x = torch.arange(24, dtype=dtype).reshape(3, 8).T[::2]
        y = x.flip(0) + .4
        before = x.clone(), y.clone()
        c, loss = impl.cross_correlation(x, y), impl.barlow_twins_loss(x, y)
        assert c.shape == (3, 3) and loss.shape == ()
        assert c.dtype == loss.dtype == dtype and c.device == loss.device == x.device
        close(x, before[0]); close(y, before[1])


@pytest.mark.t0
def test_provided_validation(impl):
    """BatchNorm training needs B>=2; finite scalars and matching floating views are contractual."""
    for x in (torch.ones(1, 2), torch.ones(0, 2), torch.ones(3, 0), torch.ones(3),
              torch.ones(3, 2).long(), torch.full((3, 2), float('nan')), torch.full((3, 2), float('inf'))):
        with pytest.raises(ValueError): impl.cross_correlation(x, x)
    x = torch.ones(3, 2)
    for y in (x.double(), torch.ones(2, 3)):
        with pytest.raises(ValueError): impl.cross_correlation(x, y)
    for eps in (0., -1., float('nan'), float('inf'), True, torch.tensor(.1)):
        with pytest.raises(ValueError): impl.cross_correlation(x, x, eps)
    for weight in (-1., float('nan'), float('inf'), False, torch.tensor(.1)):
        with pytest.raises(ValueError): impl.barlow_twins_loss(x, x, weight)


@pytest.mark.t1
def test_independent_scalar_covariance_and_weighted_sum(impl, artifact):
    """Equations 1–2 use B normalization and every ordered off-diagonal pair, without averaging."""
    x, y = pair()
    for eps in (1e-5, .13):
        expected = x.new_tensor(oracle(x.tolist(), y.tolist(), eps))
        c = impl.cross_correlation(x, y, eps)
        close(c, expected, atol=1e-12, rtol=1e-12)
        assert abs(float(expected[0, 1]-expected[1, 0])) > .1
        for weight in (0., .0051, .4):
            close(impl.barlow_twins_loss(x, y, weight, eps),
                  x.new_tensor(objective(x.tolist(), y.tolist(), weight, eps)), atol=1e-12, rtol=1e-12)
    artifact('cross_correlation', c)
    artifact('expected_cross_correlation', expected)


@pytest.mark.t1
def test_orthogonal_duplicate_and_single_feature_examples(impl):
    """Unit-variance orthogonal columns isolate the diagonal; duplicates create two ordered penalties."""
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=torch.float64)
    eps, weight = .2, .3
    c = 1 / (1 + eps)
    close(impl.cross_correlation(x, x, eps), torch.eye(2, dtype=x.dtype)*c, atol=1e-12, rtol=1e-12)
    close(impl.barlow_twins_loss(x, x, weight, eps), x.new_tensor(2*(c-1)**2), atol=1e-12, rtol=1e-12)
    duplicate = x[:, :1].repeat(1, 2)
    close(impl.barlow_twins_loss(duplicate, duplicate, weight, eps),
          x.new_tensor(2*(c-1)**2 + 2*weight*c*c), atol=1e-12, rtol=1e-12)
    one = torch.tensor([[-1.], [1.]], dtype=x.dtype)
    close(impl.barlow_twins_loss(one, one, 19., eps), x.new_tensor((c-1)**2), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_translation_view_swap_and_paired_permutations(impl):
    """Centering removes independent feature offsets; swapping views transposes C and preserves loss."""
    x, y = pair()
    c = impl.cross_correlation(x, y, .03)
    loss = impl.barlow_twins_loss(x, y, .2, .03)
    close(impl.cross_correlation(x + x.new_tensor([8., -3., 1.]), y - y.new_tensor([2., .3, -7.]), .03), c, atol=1e-12, rtol=1e-12)
    close(impl.cross_correlation(y, x, .03), c.T, atol=1e-12, rtol=1e-12)
    close(impl.barlow_twins_loss(y, x, .2, .03), loss, atol=1e-12, rtol=1e-12)
    idx = [2, 0, 1]
    close(impl.cross_correlation(x.flip(0)[:, idx], y.flip(0)[:, idx], .03), c[idx][:, idx], atol=1e-12, rtol=1e-12)
    close(impl.barlow_twins_loss(x.flip(0)[:, idx], y.flip(0)[:, idx], .2, .03), loss, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_batch_replication_and_epsilon_scaling(impl):
    """Population normalization is invariant to repeated batches; epsilon scales with squared input scale."""
    x, y = pair()
    c = impl.cross_correlation(x, y, .04)
    close(impl.cross_correlation(x.repeat(3, 1), y.repeat(3, 1), .04), c, atol=1e-12, rtol=1e-12)
    close(impl.cross_correlation(3*x, 3*y, .36), c, atol=1e-12, rtol=1e-12)
    close(impl.cross_correlation(-x, -y, .04), c, atol=1e-12, rtol=1e-12)
    # Fixed epsilon makes scale invariance approximate, not an exact identity.
    assert float((impl.cross_correlation(.001*x, .001*y, .04)-c).abs().max()) > .1


@pytest.mark.t2
def test_loss_gradient_both_views_matches_scalar_finite_difference(impl):
    """No stop-gradient: each coordinate of both views differentiates the complete scalar objective."""
    x, y = pair()
    x.requires_grad_(); y.requires_grad_()
    grads = torch.autograd.grad(impl.barlow_twins_loss(x, y, .23, .07), (x, y), allow_unused=True)
    h = 1e-6
    for side, gradient in enumerate(grads):
        assert gradient is not None, 'Both Barlow Twins views must receive gradients'
        expected = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                a, b = x.detach().tolist(), y.detach().tolist()
                rows = (a, b)[side]
                rows[i][j] += h
                plus = objective(a, b, .23, .07)
                rows[i][j] -= 2*h
                minus = objective(a, b, .23, .07)
                expected[i, j] = (plus-minus)/(2*h)
        close(gradient, expected, atol=2e-8, rtol=2e-6)
        assert float(gradient.abs().max()) > .01
        close(gradient.sum(0), torch.zeros(x.shape[1], dtype=x.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_cross_correlation_vjp_matches_independent_scalar_difference(impl):
    """The full cross-correlation matrix carries gradients, including asymmetric off-diagonal entries."""
    x, y = pair()
    x.requires_grad_(); y.requires_grad_()
    weights = x.new_tensor([[.2, -.5, .7], [1., -.3, .1], [-.2, .4, .9]])
    grads = torch.autograd.grad((impl.cross_correlation(x, y, .02)*weights).sum(), (x, y), allow_unused=True)
    h = 1e-6
    for side, gradient in enumerate(grads):
        assert gradient is not None, 'The correlation function must retain both gradient paths'
        expected = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                a, b = x.detach().tolist(), y.detach().tolist()
                rows = (a, b)[side]
                rows[i][j] += h
                plus = oracle(a, b, .02)
                rows[i][j] -= 2*h
                minus = oracle(a, b, .02)
                expected[i, j] = sum(float(weights[k, l])*(plus[k][l]-minus[k][l])/(2*h)
                                     for k in range(3) for l in range(3))
        close(gradient, expected, atol=2e-8, rtol=2e-6)


@pytest.mark.t2
def test_exact_collapse_and_tiny_variance(impl):
    """Epsilon gives finite constant features; exact double collapse has loss D and zero gradient."""
    x = torch.full((4, 3), 7., dtype=torch.float64, requires_grad=True)
    y = torch.full((4, 3), -2., dtype=torch.float64, requires_grad=True)
    close(impl.cross_correlation(x, y), torch.zeros(3, 3, dtype=x.dtype))
    loss = impl.barlow_twins_loss(x, y)
    close(loss, x.new_tensor(3.))
    for gradient in torch.autograd.grad(loss, (x, y)):
        close(gradient, torch.zeros_like(x))
    a, b = pair()
    a, b = a * 1e-8, b * 1e-8
    close(impl.cross_correlation(a, b, 1e-5), a.new_tensor(oracle(a.tolist(), b.tolist(), 1e-5)), atol=1e-20, rtol=1e-10)


@pytest.mark.t3
def test_actual_embedding_decorrelation_experiment(impl, artifact):
    """The submitted differentiable loss learns aligned, nonredundant features through 120 real Adam steps."""
    result = run(impl.cross_correlation, impl.barlow_twins_loss)
    artifact('barlow_twins_geometry', result)
    m = result['metrics']
    assert m['updates'] == 120
    assert m['final_loss'] < .01 * m['initial_loss']
    assert m['diagonal_error'] < .005 and m['off_diagonal_energy'] < .01
    assert abs(m['oracle_final_loss']-m['final_loss']) < 1e-10
    assert m['correlation_error'] < 1e-12
