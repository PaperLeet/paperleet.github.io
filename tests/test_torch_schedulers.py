"""Independent native-reference schedules, optimizer updates and saved training state."""
import copy
import gc
import warnings
import weakref

import numpy as np
import pytest
import torch as native

from paperleet._vendor.torch import UnsupportedOperation, torch as lite


NAMES = ('step', 'multi', 'exponential', 'cosine', 'linear', 'constant',
         'polynomial', 'lambda', 'multiplicative', 'sequential', 'chained', 'plateau')


def make(api, name, *, tensor_lr=False, dtype='float64', last_epoch=-1):
    dtype = getattr(api, dtype)
    params = [api.nn.Parameter(api.ones(2, dtype=dtype)) for _ in range(2)]
    rates = [api.tensor(value, dtype=dtype) if tensor_lr else value for value in (.2, .05)]
    optimizer = api.optim.SGD([{'params': [p], 'lr': lr} for p, lr in zip(params, rates)], momentum=.8)
    if last_epoch != -1:
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr'].clone() if tensor_lr else group['lr']
    scheduler = schedule(api, optimizer, name, last_epoch)
    return params, optimizer, scheduler


def schedule(api, optimizer, name, last_epoch=-1):
    S = api.optim.lr_scheduler
    options = {'last_epoch': last_epoch}
    if name == 'step':return S.StepLR(optimizer, 3, .5, **options)
    if name == 'multi':return S.MultiStepLR(optimizer, [0, 1, 3, 3, 7], .5, **options)
    if name == 'exponential':return S.ExponentialLR(optimizer, .8, **options)
    if name == 'cosine':return S.CosineAnnealingLR(optimizer, 4, .01, **options)
    if name == 'linear':return S.LinearLR(optimizer, .2, 1., 3, **options)
    if name == 'constant':return S.ConstantLR(optimizer, .25, 3, **options)
    if name == 'polynomial':return S.PolynomialLR(optimizer, 4, 1.7, **options)
    if name == 'lambda':return S.LambdaLR(optimizer, [lambda n:1/(n+1), lambda n:.9**n], **options)
    if name == 'multiplicative':return S.MultiplicativeLR(optimizer, [lambda n:.8, lambda n:.9], **options)
    if name == 'sequential':return S.SequentialLR(optimizer, [S.LinearLR(optimizer, .2, total_iters=3),
                                                              S.CosineAnnealingLR(optimizer, 4, .01)], [3], **options)
    if name == 'chained':return S.ChainedScheduler([S.ConstantLR(optimizer, .25, 3), S.ExponentialLR(optimizer, .8)])
    return S.ReduceLROnPlateau(optimizer, factor=.5, patience=1, cooldown=1, min_lr=[.012, .003])


def number(value):
    return value.item() if hasattr(value, 'item') else value


def plain(value):
    if isinstance(value, (lite.Tensor, native.Tensor)):
        return ('tensor', str(value.dtype).split('.')[-1], value.detach().numpy().tolist())
    if isinstance(value, dict):return {key:plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):return [plain(item) for item in value]
    return value


def advance(api, params, optimizer, scheduler, step, name):
    for p in params:p.grad = api.full_like(p, .3 + step * .01)
    optimizer.step()
    if step == 4:
        for group in optimizer.param_groups:
            if isinstance(group['lr'], api.Tensor):group['lr'].fill_(.037)
            else:group['lr'] = .037
    if name == 'plateau':scheduler.step([1., 1.1, 1.2, .9, .9, .9, float('nan'), 1., 1.][step % 9])
    else:scheduler.step()


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('tensor_lr', [False, True])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_grouped_trajectories_manual_rate_edits_and_training_updates_match_native(name, tensor_lr, dtype):
    histories = []
    for api in (lite, native):
        params, optimizer, scheduler = make(api, name, tensor_lr=tensor_lr, dtype=dtype)
        identities = [group['lr'] for group in optimizer.param_groups]
        rows = []
        for step in range(13):
            rows.append([*[number(v) for v in scheduler.get_last_lr()], *params[0].detach().numpy()])
            advance(api, params, optimizer, scheduler, step, name)
            if tensor_lr:
                assert all(group['lr'] is old for group, old in zip(optimizer.param_groups, identities))
                assert all(a is not group['lr'] for a, group in zip(scheduler.get_last_lr(), optimizer.param_groups))
        histories.append(rows)
    np.testing.assert_allclose(histories[0], histories[1], rtol=4e-6 if dtype=='float32' else 1e-12, atol=1e-12)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('tensor_lr', [False, True])
def test_checkpoint_rebuild_load_and_continue_matches_native_without_replacing_optimizer(name, tensor_lr):
    endings = []
    for api in (lite, native):
        params, optimizer, scheduler = make(api, name, tensor_lr=tensor_lr)
        for step in range(5):advance(api, params, optimizer, scheduler, step, name)
        checkpoint = copy.deepcopy({'params':[p.detach() for p in params], 'optimizer':optimizer.state_dict(),
                                    'scheduler':scheduler.state_dict()})
        before = plain(checkpoint)
        restored, ro, rs = make(api, name, tensor_lr=tensor_lr)
        rs.load_state_dict(checkpoint['scheduler']);ro.load_state_dict(checkpoint['optimizer'])
        with api.no_grad():
            for p, saved in zip(restored, checkpoint['params']):p[:] = saved
        assert rs.optimizer is ro and plain(checkpoint) == before
        for step in range(5,13):
            advance(api, params, optimizer, scheduler, step, name)
            advance(api, restored, ro, rs, step, name)
            np.testing.assert_array_equal(restored[0].detach().numpy(), params[0].detach().numpy())
            np.testing.assert_allclose([number(v) for v in rs.get_last_lr()], [number(v) for v in scheduler.get_last_lr()])
        endings.append((restored[0].detach().numpy(), set(scheduler.state_dict())))
    np.testing.assert_allclose(endings[0][0], endings[1][0], rtol=1e-12, atol=1e-12)
    assert endings[0][1] == endings[1][1]


@pytest.mark.parametrize('name', NAMES[:9])
@pytest.mark.parametrize('epoch', [0, 2, 6, 10])
def test_explicit_epoch_closed_form_and_resume_constructor_match_native(name, epoch):
    results = []
    for api in (lite, native):
        _, optimizer, scheduler = make(api, name, last_epoch=epoch)
        before = [number(v) for v in scheduler.get_last_lr()]
        optimizer.step()
        with pytest.warns(UserWarning, match='epoch'):scheduler.step(epoch)
        after = [number(v) for v in scheduler.get_last_lr()]
        results.append([before, after])
    np.testing.assert_allclose(results[0], results[1], rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize('mode', ['min', 'max'])
@pytest.mark.parametrize('threshold_mode', ['rel', 'abs'])
@pytest.mark.parametrize('patience,cooldown', [(0,0), (1,2), (2,0)])
def test_plateau_best_bad_epochs_cooldown_floors_nan_and_infinity_match_native(mode, threshold_mode, patience, cooldown):
    histories = []
    for api in (lite, native):
        _, optimizer, _ = make(api, 'step')
        s = api.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, .5, patience, .1,
                                                  threshold_mode, cooldown, [.026,.007], 1e-5)
        rows = []
        for value in [1., 1., 1.01, .9, .89, .8, float('nan'), float('inf'), -float('inf'), -.3, -.3]*2:
            s.step(api.tensor(value, dtype=api.float64))
            rows.append([s.best, s.num_bad_epochs, s.cooldown_counter, *s.get_last_lr()])
        histories.append(rows)
    np.testing.assert_allclose(histories[0], histories[1], rtol=1e-14, atol=1e-15, equal_nan=True)


@pytest.mark.parametrize('name', ['lambda','multiplicative'])
def test_callable_objects_save_attributes_but_functions_are_not_serialized(name):
    class Decay:
        def __init__(self, scale):self.scale = scale;self.calls = 0
        def __call__(self, epoch):self.calls += 1;return self.scale ** epoch
    results = []
    for api in (lite, native):
        _, optimizer, _ = make(api, 'step')
        cls = api.optim.lr_scheduler.LambdaLR if name=='lambda' else api.optim.lr_scheduler.MultiplicativeLR
        fn = Decay(.9);s = cls(optimizer, [fn, lambda n:.8])
        optimizer.step();s.step();saved = copy.deepcopy(s.state_dict())
        new_fn = Decay(.3);restored = cls(optimizer, [new_fn, lambda n:.8]);before = plain(saved)
        restored.load_state_dict(saved)
        assert plain(saved) == before and restored.lr_lambdas[0] is new_fn
        assert saved['lr_lambdas'][1] is None and new_fn.__dict__ == fn.__dict__
        results.append(saved['lr_lambdas'])
    assert results[0] == results[1]


@pytest.mark.parametrize('milestones', [[0,3], [2,2], [2,5]])
def test_sequential_zero_duplicate_milestones_and_nested_composition_match_native(milestones):
    histories = []
    for api in (lite, native):
        _, o, _ = make(api, 'step');S = api.optim.lr_scheduler
        nested = S.ChainedScheduler([S.ConstantLR(o, .5, 2), S.ExponentialLR(o, .9)])
        with warnings.catch_warnings(record=True) as initialization:
            warnings.simplefilter('always')
            scheduler = S.SequentialLR(o, [S.LinearLR(o,.2,total_iters=2), nested, S.CosineAnnealingLR(o,4)], milestones)
        assert bool(initialization) == (milestones[0] == 0)
        rows = []
        failure = None
        for _ in range(10):
            rows.append([number(v) for v in scheduler.get_last_lr()]);o.step()
            try:scheduler.step()
            except NotImplementedError:
                # Native ChainedScheduler has no closed-form rate query for
                # entering it through SequentialLR's epoch-zero transition.
                failure = scheduler.last_epoch
                break
        histories.append((rows, failure))
    np.testing.assert_allclose(histories[0][0], histories[1][0], rtol=1e-13, atol=1e-15)
    assert histories[0][1] == histories[1][1]


def test_step_order_override_warnings_and_custom_scheduler():
    for api in (lite, native):
        _, o, s = make(api, 'step')
        with pytest.warns(UserWarning, match='before'):s.step()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');s.step()
        assert not caught
        _, o, s = make(api, 'step');o.step = lambda:None
        with pytest.warns(UserWarning, match='overridden'):s.step()
        with pytest.warns(UserWarning, match='get_last_lr'):s.get_lr()
        class Custom(api.optim.lr_scheduler.LRScheduler):
            def get_lr(self):return (base/(1+self.last_epoch) for base in self.base_lrs)
        _, o, _ = make(api, 'step');custom = Custom(o)
        o.step();custom.step();np.testing.assert_allclose(custom.get_last_lr(), [.1,.025])


@pytest.mark.parametrize('name', ['step','multi','exponential','linear','constant','plateau'])
def test_added_groups_are_used_by_current_rate_schedules(name):
    histories = []
    for api in (lite, native):
        _, o, s = make(api, name)
        if name=='plateau':s = api.optim.lr_scheduler.ReduceLROnPlateau(o, patience=0, factor=.5, min_lr=.001)
        o.add_param_group({'params':[api.nn.Parameter(api.ones(1))], 'lr':.12})
        rows = []
        for _ in range(7):
            o.step();s.step(1.) if name=='plateau' else s.step()
            rows.append([number(v) for v in s.get_last_lr()])
        histories.append(rows)
    np.testing.assert_allclose(histories[0], histories[1], rtol=1e-14, atol=1e-15)


def test_tracking_wrapper_does_not_keep_discarded_optimizer_alive():
    for api in (lite, native):
        _, optimizer, scheduler = make(api, 'step')
        reference = weakref.ref(optimizer);step = optimizer.step
        del scheduler, optimizer;gc.collect()
        assert reference() is None
        assert getattr(step, '_wrapped_by_lr_sched', False)


@pytest.mark.parametrize('case', ['missing_initial','lambda_count','linear_start','linear_end',
                                  'constant_factor','plateau_factor','plateau_mode','plateau_threshold',
                                  'plateau_floors','empty_sequence','wrong_count','wrong_optimizer','metric_child'])
def test_invalid_configuration_errors_match_native(case):
    for api in (lite, native):
        _, o, _ = make(api, 'step');S = api.optim.lr_scheduler
        with pytest.raises((ValueError,KeyError)):
            if case=='missing_initial':
                for g in o.param_groups:g.pop('initial_lr')
                S.StepLR(o,3,last_epoch=2)
            elif case=='lambda_count':S.LambdaLR(o,[lambda n:1])
            elif case=='linear_start':S.LinearLR(o,start_factor=0)
            elif case=='linear_end':S.LinearLR(o,end_factor=1.1)
            elif case=='constant_factor':S.ConstantLR(o,factor=1.1)
            elif case=='plateau_factor':S.ReduceLROnPlateau(o,factor=1.)
            elif case=='plateau_mode':S.ReduceLROnPlateau(o,mode='other')
            elif case=='plateau_threshold':S.ReduceLROnPlateau(o,threshold_mode='other')
            elif case=='plateau_floors':S.ReduceLROnPlateau(o,min_lr=[.1])
            elif case=='empty_sequence':S.SequentialLR(o,[],[])
            elif case=='wrong_count':S.SequentialLR(o,[S.StepLR(o,2)],[2])
            elif case=='wrong_optimizer':
                _, other, _ = make(api,'step');S.ChainedScheduler([S.StepLR(o,2),S.StepLR(other,2)])
            else:S.SequentialLR(o,[S.ReduceLROnPlateau(o)],[])


def test_plateau_explicit_group_floors_must_expand_after_dynamic_addition():
    for api in (lite,native):
        _,o,_=make(api,'step');s=api.optim.lr_scheduler.ReduceLROnPlateau(o,patience=0,min_lr=[.01,.01])
        s.step(1.);o.add_param_group({'params':[api.nn.Parameter(api.ones(1))],'lr':.2})
        with pytest.raises(RuntimeError,match='min_lr'):s.step(1.)
        s.min_lrs.append(.02);s.step(1.)
        np.testing.assert_allclose(s.get_last_lr(),[.02,.01,.02])


@pytest.mark.parametrize('name',['step','exponential','cosine','linear','constant','polynomial','multiplicative'])
@pytest.mark.parametrize('initial',[.123456789,1e-38,1e-44])
@pytest.mark.parametrize('shape',[(),(1,)])
def test_float32_rate_rounding_matches_native_exactly_at_small_and_subnormal_values(name,initial,shape):
    histories=[]
    for api in (lite,native):
        p=api.nn.Parameter(api.ones(1))
        rate=api.tensor(np.full(shape,initial),dtype=api.float32)
        o=api.optim.SGD([{'params':[p],'lr':rate},{'params':[],'lr':rate.clone()}])
        s=schedule(api,o,name);o.step();rows=[]
        for _ in range(12):
            rows.append(s.get_last_lr()[0].detach().numpy().copy());s.step()
        histories.append(rows)
    np.testing.assert_array_equal(histories[0],histories[1])


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_shared_rate_groups_read_before_filling_and_invalidate_saved_graphs(dtype):
    histories=[]
    for api in (lite,native):
        rate=api.tensor(.2,dtype=getattr(api,dtype))
        params=[api.nn.Parameter(api.ones(1)) for _ in range(2)]
        o=api.optim.SGD([{'params':[p],'lr':rate} for p in params])
        s=api.optim.lr_scheduler.LambdaLR(o,[lambda n:.8**n,lambda n:.5**n])
        x=api.ones(1,dtype=getattr(api,dtype),requires_grad=True);loss=(x*rate).sum()
        before=rate._version;o.step();s.step()
        assert rate._version>before and all(g['lr'] is rate for g in o.param_groups)
        assert all(v is not rate for v in s.get_last_lr())
        with pytest.raises((RuntimeError,UnsupportedOperation)):loss.backward()
        histories.append([v.item() for v in s.get_last_lr()])
    np.testing.assert_array_equal(histories[0],histories[1])
