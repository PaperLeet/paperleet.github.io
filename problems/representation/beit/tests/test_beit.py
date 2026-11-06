import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.beit.experiment import run


def oracle(logits, labels, mask):
    losses = []
    for b, image in enumerate(logits):
        for n, row in enumerate(image):
            if mask[b][n]:
                peak = max(row)
                losses.append(peak+math.log(sum(math.exp(v-peak) for v in row))-row[labels[b][n]])
    return sum(losses)/max(len(losses), 1)


def sample():
    x = torch.tensor([[[.2, -.5, 1.4], [2., 1., -.4], [-.8, .3, .7]],
                      [[1., -.3, .2], [-.6, .5, .1], [.9, .3, -.2]]], dtype=torch.float64)
    labels = torch.tensor([[2, 1, 0], [0, 1, 2]], dtype=torch.int64)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    return x, labels, mask


@pytest.mark.t0
def test_dtype_stride_no_mutation_and_scalar(impl):
    """Masked vocabulary logits yield one scalar with unchanged floating dtype and input storage."""
    for dtype in (torch.float32, torch.float64):
        x = (torch.arange(36, dtype=dtype).reshape(2, 3, 6)*.1)[..., ::2]
        _, labels, mask = sample(); originals = x.clone(), labels.clone(), mask.clone()
        loss = impl.masked_token_cross_entropy(x, labels, mask)
        assert loss.shape == () and loss.dtype == dtype and loss.device == x.device
        for after, before in zip((x, labels, mask), originals): close(after, before)


@pytest.mark.t0
def test_provided_validation_and_unselected_sentinels(impl):
    """Only masked IDs are supervised; visible sentinels are ignored but masked IDs must index the vocabulary."""
    x, labels, mask = sample()
    baseline = impl.masked_token_cross_entropy(x, labels, mask)
    for sentinel_values in ((-99, -99, -99), (1_000_000, 3, 999), (-99, 1_000_000, 3)):
        sentinel = labels.clone()
        sentinel[0, 1], sentinel[0, 2], sentinel[1, 2] = sentinel_values
        try:
            actual = impl.masked_token_cross_entropy(x, sentinel, mask)
        except (IndexError, RuntimeError, ValueError) as error:
            assert False, f'Visible token IDs must be ignored before gathering, including positive sentinels: {error}'
        close(actual, baseline, atol=1e-12, rtol=1e-12)
    for bad in (x.long(), x[:, :0], x[..., :0], torch.ones(2, 3), x*float('nan'), x*float('inf')):
        with pytest.raises(ValueError): impl.masked_token_cross_entropy(bad, labels, mask)
    for ids, selected in ((labels.float(), mask), (labels[:1], mask), (labels, mask.float()), (labels, mask[:1])):
        with pytest.raises(ValueError): impl.masked_token_cross_entropy(x, ids, selected)
    for invalid in (-1, 3):
        ids = labels.clone(); ids[0, 0] = invalid
        with pytest.raises(ValueError): impl.masked_token_cross_entropy(x, ids, mask)


@pytest.mark.t1
def test_independent_stable_scalar_cross_entropy(impl, artifact):
    """The BEiT token objective is negative log probability, averaged over the selected positions."""
    x, labels, mask = sample()
    expected = x.new_tensor(oracle(x.tolist(), labels.tolist(), mask.tolist()))
    close(impl.masked_token_cross_entropy(x, labels, mask), expected, atol=1e-12, rtol=1e-12)
    artifact('expected_masked_ce', expected)
    uniform = torch.zeros_like(x)
    close(impl.masked_token_cross_entropy(uniform, labels, mask), x.new_tensor(math.log(3)), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_global_token_weighting_not_equal_image_weighting(impl):
    """Each masked token has equal weight; images with more masked tokens contribute more terms."""
    x = torch.tensor([[[0., 0.], [80., -80.]], [[0., 2.], [1., -1.]]], dtype=torch.float64)
    labels = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64)
    mask = torch.tensor([[True, False], [True, True]])
    expected = (math.log(2)+math.log1p(math.exp(-2))+math.log1p(math.exp(2)))/3
    close(impl.masked_token_cross_entropy(x, labels, mask), x.new_tensor(expected), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_analytic_gradient_and_scalar_finite_difference(impl):
    """A masked logit has gradient (softmax minus one-hot)/M; visible gradients vanish exactly."""
    x, labels, mask = sample(); x.requires_grad_()
    loss = impl.masked_token_cross_entropy(x, labels, mask)
    assert loss.requires_grad, 'Cross entropy must differentiate the logits'
    gradient = torch.autograd.grad(loss, x)[0]
    analytic, finite = torch.zeros_like(x), torch.zeros_like(x); h = 1e-6
    for b in range(2):
        for n in range(3):
            row = x.detach().tolist()[b][n]; peak = max(row)
            exps = [math.exp(v-peak) for v in row]
            for k in range(3):
                if bool(mask[b, n]): analytic[b, n, k] = (exps[k]/sum(exps)-int(k==int(labels[b, n])))/3
                values = x.detach().tolist(); values[b][n][k] += h
                plus = oracle(values, labels.tolist(), mask.tolist()); values[b][n][k] -= 2*h
                finite[b, n, k] = (plus-oracle(values, labels.tolist(), mask.tolist()))/(2*h)
    close(gradient, analytic, atol=1e-12, rtol=1e-12)
    close(gradient, finite, atol=2e-9, rtol=2e-6)
    parent = torch.stack((x.detach(), torch.full_like(x, 13.)), -1).reshape(2, 3, 6).requires_grad_()
    strided = parent[..., ::2]
    assert not strided.is_contiguous()
    strided_loss = impl.masked_token_cross_entropy(strided, labels, mask)
    assert strided_loss.requires_grad, 'Noncontiguous vocabulary scores must retain parent gradients'
    gp = torch.autograd.grad(strided_loss, parent)[0]
    parent_expected = torch.zeros_like(parent); parent_expected[..., ::2] = analytic
    close(gp, parent_expected, atol=1e-12, rtol=1e-12)
    close(gp[..., 1::2], torch.zeros_like(gp[..., 1::2]), atol=0, rtol=0)


@pytest.mark.t2
def test_logit_offsets_vocabulary_relabeling_and_replication(impl):
    """Categorical CE is invariant to per-patch score offsets, consistent class renaming and data replication."""
    x, labels, mask = sample(); base = impl.masked_token_cross_entropy(x, labels, mask)
    offsets = torch.arange(6, dtype=x.dtype).reshape(2, 3, 1)*17
    close(impl.masked_token_cross_entropy(x+offsets, labels, mask), base, atol=1e-12, rtol=1e-12)
    # Reversing the vocabulary maps old class k to K-1-k.
    close(impl.masked_token_cross_entropy(x.flip(-1), 2-labels, mask), base, atol=1e-12, rtol=1e-12)
    close(impl.masked_token_cross_entropy(x.repeat(2, 1, 1), labels.repeat(2, 1), mask.repeat(2, 1)), base)
    changed = x + (~mask).to(x.dtype)[..., None]*x.new_tensor([100., -100., 20.])
    close(impl.masked_token_cross_entropy(changed, labels, mask), base, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_empty_full_masks_single_class_and_extreme_logits(impl):
    """Empty masks give differentiable zero; stable log-sum-exp handles extreme finite scores and K=1."""
    x, labels, mask = sample(); x.requires_grad_()
    loss = impl.masked_token_cross_entropy(x, labels, torch.zeros_like(mask))
    assert loss.requires_grad; close(loss, x.new_tensor(0.))
    close(torch.autograd.grad(loss, x)[0], torch.zeros_like(x))
    one = torch.ones(2, 3, 1, dtype=x.dtype, requires_grad=True)
    single = impl.masked_token_cross_entropy(one, torch.zeros_like(labels), mask)
    close(single, x.new_tensor(0.)); close(torch.autograd.grad(single, one)[0], torch.zeros_like(one))
    huge = torch.tensor([[[1000., -1000., 999.], [-1000., 1000., 999.]]], dtype=x.dtype, requires_grad=True)
    ids = torch.tensor([[1, 1]], dtype=torch.int64); full = torch.ones(1, 2, dtype=torch.bool)
    loss = impl.masked_token_cross_entropy(huge, ids, full)
    close(loss, x.new_tensor(oracle(huge.detach().tolist(), ids.tolist(), full.tolist())), atol=1e-12, rtol=1e-12)
    assert bool(torch.isfinite(torch.autograd.grad(loss, huge)[0]).all())


@pytest.mark.t3
def test_actual_shared_discrete_token_predictor(impl, artifact):
    """The submitted masked CE learns a four-class shared token head in 100 real Adam updates."""
    result = run(impl.masked_token_cross_entropy); artifact('beit_token_training', result)
    m = result['metrics']
    assert m['updates'] == 100 and m['masked_accuracy'] == 1.
    assert m['final_ce'] < .03 and m['final_ce'] < .03*m['initial_ce']
    assert abs(m['final_loss']-m['final_ce']) < 1e-12
