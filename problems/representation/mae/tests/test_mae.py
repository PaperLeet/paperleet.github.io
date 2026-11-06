import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.mae.experiment import run


def normalized(rows, eps):
    result = []
    for image in rows:
        patches = []
        for patch in image:
            mean = sum(patch)/len(patch)
            variance = sum((v-mean)**2 for v in patch)/(len(patch)-1)
            patches.append([(v-mean)/math.sqrt(variance+eps) for v in patch])
        result.append(patches)
    return result


def oracle(prediction, target, mask, norm=False, eps=1e-6):
    target = normalized(target, eps) if norm else target
    values = [sum((x-y)**2 for x, y in zip(prediction[b][n], target[b][n]))/len(target[b][n])
              for b in range(len(target)) for n in range(len(target[b])) if mask[b][n]]
    return sum(values)/max(len(values), 1)


def sample():
    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)*.17-1.
    target = (x*.9).sin() + x.flip(-1)*.2
    mask = torch.tensor([[True, False, False], [True, True, False]])
    return x, target, mask


@pytest.mark.t0
def test_shape_dtype_strides_and_storage(impl):
    """Patch normalization preserves [B,N,P], while the masked objective is one scalar."""
    for dtype in (torch.float32, torch.float64):
        x = torch.arange(48, dtype=dtype).reshape(2, 3, 8)[..., ::2]
        y = x.flip(1) + .3
        mask = torch.tensor([[True, False, False], [True, True, False]])
        original = x.clone(), y.clone(), mask.clone()
        result = impl.normalize_patch_targets(y)
        loss = impl.masked_reconstruction_loss(x, y, mask, True)
        assert result.shape == x.shape and loss.shape == ()
        assert result.dtype == loss.dtype == dtype and result.device == loss.device == x.device
        close(x, original[0]); close(y, original[1]); close(mask, original[2])


@pytest.mark.t0
def test_provided_validation(impl):
    """The contract rejects nonfinite pixels, invalid masks and nonpositive stabilization."""
    x, y, mask = sample()
    for bad in (torch.ones(2), torch.ones(0, 3, 4), torch.ones(2, 3, 0), x.long(), x*float('nan'), x*float('inf')):
        with pytest.raises(ValueError): impl.normalize_patch_targets(bad)
    for eps in (0., -1., float('nan'), float('inf'), True, torch.tensor(.1)):
        with pytest.raises(ValueError): impl.normalize_patch_targets(x, eps)
    for target, selected in ((y.float(), mask), (y[:, :2], mask), (y, mask.float()), (y, mask[:1])):
        with pytest.raises(ValueError): impl.masked_reconstruction_loss(x, target, selected)
    with pytest.raises(ValueError): impl.masked_reconstruction_loss(x, y, mask, 1)
    with pytest.raises(ValueError): impl.masked_reconstruction_loss(x, y, mask, False, 0.)


@pytest.mark.t1
def test_independent_scalar_normalization_and_masked_loss(impl, artifact):
    """Official MAE uses sample patch variance and global masked-patch mean of pixel MSE."""
    x, y, mask = sample()
    for eps in (1e-6, .13):
        expected = y.new_tensor(normalized(y.tolist(), eps))
        close(impl.normalize_patch_targets(y, eps), expected, atol=1e-12, rtol=1e-12)
        for norm in (False, True):
            close(impl.masked_reconstruction_loss(x, y, mask, norm, eps),
                  x.new_tensor(oracle(x.tolist(), y.tolist(), mask.tolist(), norm, eps)), atol=1e-12, rtol=1e-12)
    artifact('normalized_targets', expected)


@pytest.mark.t1
def test_unequal_image_mask_counts_and_patch_pixel_mean(impl):
    """Each removed patch has equal weight even when images have different mask counts."""
    prediction = torch.tensor([[[1., 3.], [90., 90.]], [[2., 4.], [3., 5.]]], dtype=torch.float64)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False], [True, True]])
    close(impl.masked_reconstruction_loss(prediction, target, mask), prediction.new_tensor((5.+10.+17.)/3))
    # Two pixels [-1,+1] have sample variance 2, not population variance 1.
    close(impl.normalize_patch_targets(prediction.new_tensor([[[-1., 1.]]]), .5),
          prediction.new_tensor([[[-1/math.sqrt(2.5), 1/math.sqrt(2.5)]]]), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_prediction_gradient_and_fixed_target_boundary(impl):
    """Only selected predicted pixels differentiate the loss; normalized targets remain fixed data."""
    for norm in (False, True):
        x, y, mask = sample(); x.requires_grad_(); y.requires_grad_()
        loss = impl.masked_reconstruction_loss(x, y, mask, norm, .03)
        assert loss.requires_grad, 'The prediction gradient must remain connected'
        gx, gy = torch.autograd.grad(loss, (x, y), allow_unused=True)
        assert gx is not None and gy is None, 'Targets are fixed data inside the loss'
        target = normalized(y.detach().tolist(), .03) if norm else y.detach().tolist()
        expected = torch.zeros_like(x)
        for b in range(2):
            for n in range(3):
                if bool(mask[b, n]):
                    for p in range(4): expected[b, n, p] = 2*(float(x[b, n, p].detach())-target[b][n][p])/(3*4)
        close(gx, expected, atol=1e-12, rtol=1e-12)
        parent = torch.stack((x.detach(), torch.full_like(x, 7.)), -1).reshape(2, 3, 8).requires_grad_()
        target_parent = torch.stack((y.detach(), torch.full_like(y, -3.)), -1).reshape(2, 3, 8).requires_grad_()
        strided = parent[..., ::2]
        assert not strided.is_contiguous()
        strided_loss = impl.masked_reconstruction_loss(strided, target_parent[..., ::2], mask, norm, .03)
        assert strided_loss.requires_grad, 'Noncontiguous predictions must preserve the parent gradient'
        gp, gt = torch.autograd.grad(strided_loss, (parent, target_parent), allow_unused=True)
        assert gp is not None and gt is None, 'Strided targets remain fixed data'
        parent_expected = torch.zeros_like(parent); parent_expected[..., ::2] = expected
        close(gp, parent_expected, atol=1e-12, rtol=1e-12)
        close(gp[..., 1::2], torch.zeros_like(gp[..., 1::2]), atol=0, rtol=0)


@pytest.mark.t2
def test_normalizer_vjp_against_scalar_finite_difference(impl):
    """The standalone normalizer is differentiable, including its mean and sample variance."""
    _, y, _ = sample(); y.requires_grad_()
    weights = (torch.arange(24, dtype=y.dtype).reshape(y.shape)*.37).sin()
    result = (impl.normalize_patch_targets(y, .07)*weights).sum()
    assert result.requires_grad
    gradient = torch.autograd.grad(result, y)[0]
    expected = torch.zeros_like(y); h = 1e-6
    for b in range(2):
        for n in range(3):
            for p in range(4):
                values = y.detach().tolist(); values[b][n][p] += h
                plus = normalized(values, .07)[b][n]
                values[b][n][p] -= 2*h
                minus = normalized(values, .07)[b][n]
                expected[b, n, p] = sum(float(weights[b, n, k])*(plus[k]-minus[k]) for k in range(4))/(2*h)
    close(gradient, expected, atol=2e-9, rtol=2e-6)
    close(gradient.sum(-1), torch.zeros(2, 3, dtype=y.dtype), atol=1e-12, rtol=1e-12)
    parent = torch.stack((y.detach(), torch.full_like(y, 11.)), -1).reshape(2, 3, 8).requires_grad_()
    strided = parent[..., ::2]
    assert not strided.is_contiguous()
    weighted = (impl.normalize_patch_targets(strided, .07)*weights).sum()
    assert weighted.requires_grad, 'Standalone normalization must retain strided parent gradients'
    gp = torch.autograd.grad(weighted, parent)[0]
    parent_expected = torch.zeros_like(parent); parent_expected[..., ::2] = expected
    close(gp, parent_expected, atol=2e-9, rtol=2e-6)
    close(gp[..., 1::2], torch.zeros_like(gp[..., 1::2]), atol=0, rtol=0)


@pytest.mark.t2
def test_per_patch_offsets_scaling_and_constant_targets(impl):
    """Normalization removes each patch offset; nonzero scaling requires epsilon to scale quadratically."""
    _, y, _ = sample()
    base = impl.normalize_patch_targets(y, .02)
    offset = torch.arange(6, dtype=y.dtype).reshape(2, 3, 1)*4
    close(impl.normalize_patch_targets(y+offset, .02), base, atol=1e-12, rtol=1e-12)
    close(impl.normalize_patch_targets(y*3, .18), base, atol=1e-12, rtol=1e-12)
    close(impl.normalize_patch_targets(-y, .02), -base, atol=1e-12, rtol=1e-12)
    close(impl.normalize_patch_targets(torch.ones_like(y)*7), torch.zeros_like(y))


@pytest.mark.t2
def test_visible_pixels_batch_replication_and_patch_permutation(impl):
    """Visible pixels contribute nothing; duplicating examples or jointly permuting patches preserves the mean."""
    x, y, mask = sample()
    loss = impl.masked_reconstruction_loss(x, y, mask, True)
    visible_offset = (~mask).to(x.dtype)[..., None]*100
    close(impl.masked_reconstruction_loss(x+visible_offset, y-visible_offset, mask, True), loss, atol=1e-12, rtol=1e-12)
    close(impl.masked_reconstruction_loss(x.repeat(2, 1, 1), y.repeat(2, 1, 1), mask.repeat(2, 1), True), loss)
    close(impl.masked_reconstruction_loss(x.flip(1), y.flip(1), mask.flip(1), True), loss)


@pytest.mark.t2
def test_empty_full_masks_and_single_pixel_boundary(impl):
    """Empty masks extend the training loss to differentiable zero; sample variance requires P>=2."""
    x, y, mask = sample(); x.requires_grad_()
    loss = impl.masked_reconstruction_loss(x, y, torch.zeros_like(mask), True)
    assert loss.requires_grad
    close(loss, x.new_tensor(0.))
    close(torch.autograd.grad(loss, x)[0], torch.zeros_like(x))
    full = torch.ones_like(mask)
    close(impl.masked_reconstruction_loss(x, y, full), x.new_tensor(oracle(x.detach().tolist(), y.tolist(), full.tolist())))
    one = x[..., :1]
    close(impl.masked_reconstruction_loss(one, y[..., :1], mask),
          x.new_tensor(oracle(one.detach().tolist(), y[..., :1].tolist(), mask.tolist())))
    with pytest.raises(ValueError): impl.normalize_patch_targets(one)
    with pytest.raises(ValueError): impl.masked_reconstruction_loss(one, y[..., :1], mask, True)


@pytest.mark.t3
def test_actual_shared_pixel_decoder_training(impl, artifact):
    """The submitted masked loss learns a shared synthetic pixel decoder while fixed targets cannot move."""
    result = run(impl.masked_reconstruction_loss); artifact('mae_decoder_training', result)
    m = result['metrics']
    assert m['updates'] == 120 and m['target_drift'] == 0
    assert m['final_mse'] < 1e-4 and m['final_mse'] < .001*m['initial_mse']
    assert abs(m['final_loss']-m['final_mse']) < 1e-12
