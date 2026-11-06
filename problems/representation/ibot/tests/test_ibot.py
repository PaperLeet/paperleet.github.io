import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.ibot.experiment import run


def probability(row):
    peak = max(row); values = [math.exp(v-peak) for v in row]
    return [v/sum(values) for v in values]


def target_oracle(teacher, center, temperature):
    return [[probability([(v-center[k])/temperature for k, v in enumerate(patch)])
             for patch in image] for image in teacher]


def loss_oracle(student, teacher, mask, center, student_temp, teacher_temp):
    targets = target_oracle(teacher, center, teacher_temp); result = 0.
    for b, image in enumerate(student):
        count = max(sum(mask[b]), 1)
        for n, patch in enumerate(image):
            if mask[b][n]:
                scaled = [v/student_temp for v in patch]; peak = max(scaled)
                logz = peak + math.log(sum(math.exp(v-peak) for v in scaled))
                result -= sum(targets[b][n][k]*(v-logz) for k, v in enumerate(scaled))/count/len(student)
    return result


def sample():
    index = torch.arange(27, dtype=torch.float64).reshape(3, 3, 3)
    student, teacher = (index*.37).sin(), (index*.71).cos()*.6
    center = torch.tensor([.2, -.3, .05], dtype=student.dtype)
    mask = torch.tensor([[True, False, False], [True, True, False], [False, False, False]])
    return student, teacher, mask, center


@pytest.mark.t0
def test_shapes_dtypes_strides_and_no_mutation(impl):
    """Teacher targets preserve the patch layout; the image-balanced objective is a scalar."""
    for dtype in (torch.float32, torch.float64):
        s = (torch.arange(54, dtype=dtype).reshape(3, 3, 6)*.03)[..., ::2]
        t = s.flip(1)+.1; _, _, mask, center = sample(); center = center.to(dtype)
        originals = [x.clone() for x in (s, t, mask, center)]
        targets = impl.teacher_patch_targets(t, center, .4)
        loss = impl.ibot_patch_loss(s, t, mask, center, .7, .4)
        assert targets.shape == s.shape and loss.shape == ()
        assert targets.dtype == loss.dtype == dtype and targets.device == loss.device == s.device
        for after, before in zip((s, t, mask, center), originals): close(after, before)


@pytest.mark.t0
def test_provided_validation(impl):
    """Matched finite floating logits, a prototype center and positive finite temperatures define the core."""
    s, t, mask, center = sample()
    for bad in (t.long(), t[:0], t[..., :0], torch.ones(3, 3), t*float('nan'), t*float('inf')):
        with pytest.raises(ValueError): impl.teacher_patch_targets(bad, center)
    for c in (center.float(), center[:2], center[None], center*float('nan'), center*float('inf')):
        with pytest.raises(ValueError): impl.teacher_patch_targets(t, c)
    for temp in (0., -1., float('nan'), float('inf'), True, torch.tensor(.1)):
        with pytest.raises(ValueError): impl.teacher_patch_targets(t, center, temp)
        with pytest.raises(ValueError): impl.ibot_patch_loss(s, t, mask, center, temp)
    for student, selected in ((s.float(), mask), (s[:2], mask), (s, mask.float()), (s, mask[:2])):
        with pytest.raises(ValueError): impl.ibot_patch_loss(student, t, selected, center)


@pytest.mark.t1
def test_independent_centered_soft_targets_and_cross_entropy(impl, artifact):
    """iBOT centers teacher scores before temperature-softmax and predicts the full soft distribution."""
    s, t, mask, center = sample()
    for student_temp, teacher_temp in ((.1, .04), (.7, .4)):
        targets = s.new_tensor(target_oracle(t.tolist(), center.tolist(), teacher_temp))
        close(impl.teacher_patch_targets(t, center, teacher_temp), targets, atol=1e-12, rtol=1e-12)
        expected = loss_oracle(s.tolist(), t.tolist(), mask.tolist(), center.tolist(), student_temp, teacher_temp)
        close(impl.ibot_patch_loss(s, t, mask, center, student_temp, teacher_temp), s.new_tensor(expected), atol=1e-12, rtol=1e-12)
    artifact('centered_teacher_targets', targets)


@pytest.mark.t1
def test_image_balanced_masked_mean_including_empty_images(impl):
    """Official patch loss normalizes each image separately, and an empty image still counts in B."""
    s, t, mask, center = sample()
    losses = []
    for b in range(3):
        losses.append(loss_oracle(s[b:b+1].tolist(), t[b:b+1].tolist(), mask[b:b+1].tolist(), center.tolist(), .7, .4))
    assert losses[2] == 0 and abs(losses[0]-losses[1]) > .01
    close(impl.ibot_patch_loss(s, t, mask, center, .7, .4), s.new_tensor(sum(losses)/3), atol=1e-12, rtol=1e-12)
    global_mean = (losses[0]+2*losses[1])/3
    assert abs(global_mean-sum(losses)/3) > .1, 'Fixture must distinguish global-token and per-image reductions'


@pytest.mark.t2
def test_gradient_destination_and_analytic_student_gradient(impl):
    """Teacher logits and center are stopped; masked student gradients are (p-q)/(B*count*student_temp)."""
    s, t, mask, center = sample(); s.requires_grad_(); t.requires_grad_(); center.requires_grad_()
    targets = impl.teacher_patch_targets(t, center, .4)
    assert not targets.requires_grad, 'The entire teacher target, including its center, must be detached'
    loss = impl.ibot_patch_loss(s, t, mask, center, .7, .4)
    assert loss.requires_grad
    gs, gt, gc = torch.autograd.grad(loss, (s, t, center), allow_unused=True)
    assert gs is not None and gt is None and gc is None
    expected = torch.zeros_like(s)
    q = target_oracle(t.detach().tolist(), center.detach().tolist(), .4)
    for b in range(3):
        for n in range(3):
            p = probability([v/.7 for v in s.detach().tolist()[b][n]])
            if bool(mask[b, n]):
                for k in range(3): expected[b, n, k] = (p[k]-q[b][n][k])/(3*int(mask[b].sum())*.7)
    close(gs, expected, atol=1e-12, rtol=1e-12)
    parent = torch.stack((s.detach(), torch.full_like(s, 17.)), -1).reshape(3, 3, 6).requires_grad_()
    teacher_parent = torch.stack((t.detach(), torch.full_like(t, -7.)), -1).reshape(3, 3, 6).requires_grad_()
    center_parent = torch.stack((center.detach(), torch.full_like(center, 9.)), -1).reshape(6).requires_grad_()
    strided, teacher_view, center_view = parent[..., ::2], teacher_parent[..., ::2], center_parent[::2]
    assert not strided.is_contiguous() and not teacher_view.is_contiguous() and not center_view.is_contiguous()
    targets = impl.teacher_patch_targets(teacher_view, center_view, .4)
    assert not targets.requires_grad, 'A strided teacher and center must also be stopped'
    strided_loss = impl.ibot_patch_loss(strided, teacher_view, mask, center_view, .7, .4)
    assert strided_loss.requires_grad, 'Noncontiguous students must retain their parent gradients'
    gp, gtp, gcp = torch.autograd.grad(strided_loss, (parent, teacher_parent, center_parent), allow_unused=True)
    assert gp is not None and gtp is None and gcp is None
    parent_expected = torch.zeros_like(parent); parent_expected[..., ::2] = expected
    close(gp, parent_expected, atol=1e-12, rtol=1e-12)
    close(gp[..., 1::2], torch.zeros_like(gp[..., 1::2]), atol=0, rtol=0)


@pytest.mark.t2
def test_student_gradient_scalar_finite_difference(impl):
    """Differentiating the independently written scalar objective validates both temperature and reduction factors."""
    s, t, mask, center = sample(); s.requires_grad_()
    loss = impl.ibot_patch_loss(s, t, mask, center, .7, .4)
    assert loss.requires_grad
    gradient = torch.autograd.grad(loss, s)[0]; expected = torch.zeros_like(s); h = 1e-6
    for b in range(3):
        for n in range(3):
            for k in range(3):
                values = s.detach().tolist(); values[b][n][k] += h
                plus = loss_oracle(values, t.tolist(), mask.tolist(), center.tolist(), .7, .4)
                values[b][n][k] -= 2*h
                expected[b, n, k] = (plus-loss_oracle(values, t.tolist(), mask.tolist(), center.tolist(), .7, .4))/(2*h)
    close(gradient, expected, atol=2e-9, rtol=2e-6)


@pytest.mark.t2
def test_offsets_center_compensation_and_joint_permutations(impl):
    """Softmax ignores scalar offsets; shifting teacher and center together and permuting matched views preserves the loss."""
    s, t, mask, center = sample(); base = impl.ibot_patch_loss(s, t, mask, center, .7, .4)
    offsets = torch.arange(9, dtype=s.dtype).reshape(3, 3, 1)*3
    close(impl.ibot_patch_loss(s+offsets, t-offsets, mask, center, .7, .4), base, atol=1e-12, rtol=1e-12)
    shift = s.new_tensor([.4, -.8, 1.3])
    close(impl.teacher_patch_targets(t+shift, center+shift, .4), impl.teacher_patch_targets(t, center, .4), atol=1e-12, rtol=1e-12)
    close(impl.ibot_patch_loss(s.flip(0).flip(1), t.flip(0).flip(1), mask.flip(0).flip(1), center, .7, .4), base)
    close(impl.ibot_patch_loss(s.flip(-1), t.flip(-1), mask, center.flip(0), .7, .4), base)
    changed = s + (~mask).to(s.dtype)[..., None]*s.new_tensor([10., -20., 30.])
    close(impl.ibot_patch_loss(changed, t, mask, center, .7, .4), base)


@pytest.mark.t2
def test_soft_teacher_entropy_floor(impl):
    """Matching the teacher distribution minimizes CE at its entropy, not at zero or a hard-label loss."""
    _, t, mask, center = sample()
    s = (t-center)*(.7/.4)
    q = target_oracle(t.tolist(), center.tolist(), .4)
    entropy = 0.
    for b in range(3):
        for n in range(3):
            if bool(mask[b, n]): entropy -= sum(v*math.log(v) for v in q[b][n])/int(mask[b].sum())/3
    loss = impl.ibot_patch_loss(s, t, mask, center, .7, .4)
    close(loss, s.new_tensor(entropy), atol=1e-12, rtol=1e-12)
    assert entropy > .2
    s = s.detach().requires_grad_()
    close(torch.autograd.grad(impl.ibot_patch_loss(s, t, mask, center, .7, .4), s)[0], torch.zeros_like(s), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_empty_masks_single_prototype_and_large_logits(impl):
    """Empty images contribute zero; one prototype has zero CE and stable log-softmax avoids overflow."""
    s, t, mask, center = sample(); s.requires_grad_()
    loss = impl.ibot_patch_loss(s, t, torch.zeros_like(mask), center)
    assert loss.requires_grad; close(loss, s.new_tensor(0.))
    close(torch.autograd.grad(loss, s)[0], torch.zeros_like(s))
    one = s[..., :1].detach().requires_grad_()
    single = impl.ibot_patch_loss(one, t[..., :1], mask, center[:1])
    close(single, s.new_tensor(0.)); close(torch.autograd.grad(single, one)[0], torch.zeros_like(one))
    huge = (s.detach()*1000).requires_grad_()
    loss = impl.ibot_patch_loss(huge, -t*1000, torch.ones_like(mask), center, .7, .4)
    expected = loss_oracle(huge.detach().tolist(), (-t*1000).tolist(), torch.ones_like(mask).tolist(), center.tolist(), .7, .4)
    close(loss, s.new_tensor(expected), atol=1e-10, rtol=1e-12)
    assert bool(torch.isfinite(torch.autograd.grad(loss, huge)[0]).all())


@pytest.mark.t3
def test_actual_soft_patch_teacher_matching(impl, artifact):
    """Real Adam updates reduce independent KL without moving teacher, center, or visible student patches."""
    result = run(impl.teacher_patch_targets, impl.ibot_patch_loss); artifact('ibot_patch_training', result)
    m = result['metrics']
    assert m['updates'] == 120 and m['teacher_drift'] == m['center_drift'] == m['visible_student_drift'] == 0
    assert m['final_kl'] < 1e-5 and m['final_kl'] < .001*m['initial_kl']
    assert abs(m['final_loss']-m['oracle_ce']) < 1e-12 and m['teacher_entropy'] > .2
