import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.multimodal.siglip.experiment import run


@pytest.mark.t0
def test_contract_and_unchanged_inputs(impl):
    """Logits and scalar loss preserve floating dtype/device and leave inputs unchanged."""
    for dtype in (torch.float32, torch.float64):
        values = [torch.randn(3, 5, dtype=dtype), torch.randn(3, 5, dtype=dtype),
                  torch.tensor(.7, dtype=dtype), torch.tensor(-.8, dtype=dtype)]
        before = [x.clone() for x in values]
        logits = impl.pairwise_logits(*values)
        loss = impl.siglip_loss(logits)
        assert logits.shape == (3, 3) and loss.shape == ()
        assert logits.dtype == loss.dtype == dtype and logits.device == values[0].device
        for x, saved in zip(values, before): close(x, saved)


@pytest.mark.t0
def test_invalid_shapes_and_scalars(impl):
    """The supplied validation rejects nonfinite data, mismatched towers and nonscalar calibration."""
    x = torch.ones(2, 3); s = torch.tensor(0.); b = torch.tensor(-1.)
    for text in (x[:, :2], x.double(), x * float('nan')):
        with pytest.raises(ValueError): impl.pairwise_logits(x, text, s, b)
    for scale in (s[None], torch.tensor(float('nan')), torch.tensor(1000.)):
        with pytest.raises(ValueError): impl.pairwise_logits(x, x, scale, b)
    for logits in (torch.empty(0, 0), torch.zeros(2, 3), torch.ones(2, 2, dtype=torch.long),
                   torch.tensor([[float('inf')]])):
        with pytest.raises(ValueError): impl.siglip_loss(logits)


@pytest.mark.t1
def test_logits_against_scalar_cosines(impl):
    """Independent scalar cosines verify per-example normalization, exponential scale and bias."""
    images = torch.tensor([[3., 4., 0.], [-2., 0., 1.], [0., 0., 0.]], dtype=torch.float64)
    texts = images.new_tensor([[0., 3., 4.], [2., -1., 2.], [1., 1., -1.]])
    expected = []
    for image in images.tolist():
        row = []
        for text in texts.tolist():
            ni = max(math.sqrt(sum(v*v for v in image)), 1e-12)
            nt = max(math.sqrt(sum(v*v for v in text)), 1e-12)
            row.append(2.5 * sum(a*b for a, b in zip(image, text)) / ni / nt - 1.2)
        expected.append(row)
    close(impl.pairwise_logits(images, texts, images.new_tensor(math.log(2.5)), images.new_tensor(-1.2)),
          images.new_tensor(expected), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_loss_against_scalar_logistic_oracle(impl):
    """A stable scalar oracle checks all pairs and division by B instead of B squared."""
    logits = torch.tensor([[2., -1., .4], [.3, -.7, 2.], [-2., 0., .9]], dtype=torch.float64)
    total = 0.
    for i, row in enumerate(logits.tolist()):
        for j, value in enumerate(row):
            z = (-1 if i == j else 1) * value
            total += max(z, 0.) + math.log1p(math.exp(-abs(z)))
    close(impl.siglip_loss(logits), logits.new_tensor(total / 3), atol=1e-12, rtol=1e-12)
    close(impl.siglip_loss(torch.zeros(4, 4, dtype=torch.float64)), logits.new_tensor(4 * math.log(2)))


@pytest.mark.t2
def test_analytic_logit_gradient_and_extreme_stability(impl):
    """Positive logits are pushed upward, negatives downward, including saturated finite logits."""
    logits = torch.tensor([[1000., 1000.], [-1000., -1000.]], dtype=torch.float64, requires_grad=True)
    loss = impl.siglip_loss(logits)
    gradient, = torch.autograd.grad(loss, logits)
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    close(loss, logits.new_tensor(1000.))
    close(gradient, logits.new_tensor([[0., .5], [0., -.5]]), atol=1e-12, rtol=1e-12)
    x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    dx, = torch.autograd.grad(impl.siglip_loss(x), x)
    labels01 = torch.eye(3, dtype=x.dtype)
    close(dx, (torch.sigmoid(x.detach()) - labels01) / 3, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_tower_and_calibration_gradients(impl):
    """Central differences independently validate gradients into both feature towers, log scale and bias."""
    values = [torch.tensor([[1., .3], [-.2, .8]], dtype=torch.float64, requires_grad=True),
              torch.tensor([[.6, -.4], [.2, 1.]], dtype=torch.float64, requires_grad=True),
              torch.tensor(.8, dtype=torch.float64, requires_grad=True),
              torch.tensor(-.6, dtype=torch.float64, requires_grad=True)]
    grads = torch.autograd.grad(impl.siglip_loss(impl.pairwise_logits(*values)), values)
    for index, (value, grad) in enumerate(zip(values, grads)):
        expected = torch.zeros_like(value)
        for coordinate in range(value.numel()):
            plus, minus = [x.detach().clone() for x in values], [x.detach().clone() for x in values]
            plus[index].view(-1)[coordinate] += 1e-6
            minus[index].view(-1)[coordinate] -= 1e-6
            expected.view(-1)[coordinate] = (impl.siglip_loss(impl.pairwise_logits(*plus)) - impl.siglip_loss(impl.pairwise_logits(*minus))) / 2e-6
        close(grad, expected, atol=1e-8, rtol=1e-6)
        assert grad.abs().sum() > 1e-5


@pytest.mark.t2
def test_pair_permutation_and_feature_rescaling(impl):
    """Aligned pair permutations and positive row rescaling preserve the objective; transpose symmetry also holds."""
    image, text = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
    scale, bias = image.new_tensor(.5), image.new_tensor(-.7)
    logits = impl.pairwise_logits(image, text, scale, bias)
    loss = impl.siglip_loss(logits)
    perm = torch.tensor([2, 0, 3, 1])
    close(loss, impl.siglip_loss(logits[perm][:, perm]))
    close(loss, impl.siglip_loss(logits.T))
    close(logits, impl.pairwise_logits(image * image.new_tensor([2., .3, 4., 1.])[:, None], text * 3, scale, bias))


@pytest.mark.t2
def test_single_pair_and_negative_pairs_matter(impl):
    """B=1 remains a positive logistic task, and raising an off-diagonal score increases loss."""
    one = torch.tensor([[2.]], dtype=torch.float64, requires_grad=True)
    loss = impl.siglip_loss(one)
    close(loss, one.new_tensor(math.log1p(math.exp(-2))))
    gradient, = torch.autograd.grad(loss, one)
    assert gradient.item() < 0
    x = torch.eye(3, dtype=torch.float64)
    y = x.clone(); y[0, 2] = 4.
    assert impl.siglip_loss(y) > impl.siglip_loss(x)


@pytest.mark.t3
def test_paired_embedding_training(impl, artifact):
    """The submitted functions align synthetic image-text pairs through real gradient updates."""
    result = run(impl.pairwise_logits, impl.siglip_loss)
    artifact('siglip_pair_training', result)
    m = result['metrics']
    assert m['final_loss'] < .05 * m['initial_loss']
    assert m['final_retrieval_accuracy'] == 1.
    assert m['minimum_positive_logit'] > 2.
    assert m['maximum_negative_logit'] < -2.
