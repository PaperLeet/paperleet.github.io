"""Independent NumPy-only scheduler workflows; also executed on real WASM."""
import copy

import numpy as np
import pytest

from torchwasm import UnsupportedOperation, as_torch, torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_scheduled_grouped_sgd_updates_agree_with_independent_numpy(dtype):
    parameters = [torch.nn.Parameter(torch.tensor([1.,2.],dtype=dtype)) for _ in range(2)]
    optimizer = torch.optim.SGD([{'params':[parameters[0]],'lr':.2},
                                 {'params':[parameters[1]],'lr':.05}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=.5)
    expected = [np.array([1.,2.],dtype=dtype) for _ in range(2)]
    for epoch in range(11):
        rates = [.2*.5**(epoch//3), .05*.5**(epoch//3)]
        np.testing.assert_allclose(scheduler.get_last_lr(),rates,rtol=1e-12)
        for p, values, rate in zip(parameters, expected, rates):
            p.grad = torch.tensor([.3,-.2],dtype=dtype)
            values -= rate*np.array([.3,-.2],dtype=dtype)
        optimizer.step();scheduler.step()
        for p, values in zip(parameters,expected):np.testing.assert_allclose(p.numpy(),values,rtol=2e-6)


def test_linear_warmup_cosine_transition_and_complete_checkpoint_continuation():
    def create():
        model = torch.nn.Linear(2,1,dtype=torch.float64)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.2,weight_decay=.03)
        S = torch.optim.lr_scheduler
        scheduler = S.SequentialLR(optimizer,[S.LinearLR(optimizer,.2,total_iters=3),
                                             S.CosineAnnealingLR(optimizer,4,eta_min=.01)],[3])
        return model,optimizer,scheduler
    model,optimizer,scheduler = create()
    expected = [.04,.09333333333333334,.14666666666666667,.2,
                .172175144212722,.10500000000000001,.03782485578727799,.01]
    x = torch.ones(3,2,dtype=torch.float64)
    for epoch in range(5):
        np.testing.assert_allclose(scheduler.get_last_lr(),[expected[epoch]],rtol=1e-13)
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step();scheduler.step()
    checkpoint = copy.deepcopy({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                                'scheduler':scheduler.state_dict()})
    restored,ro,rs = create();restored.load_state_dict(checkpoint['model'])
    rs.load_state_dict(checkpoint['scheduler']);ro.load_state_dict(checkpoint['optimizer'])
    for epoch in range(5,8):
        np.testing.assert_allclose(rs.get_last_lr(),[expected[epoch]],rtol=1e-13)
        for m,o,s in ((model,optimizer,scheduler),(restored,ro,rs)):
            o.zero_grad();m(x).square().mean().backward();o.step();s.step()
        for a,b in zip(model.parameters(),restored.parameters()):np.testing.assert_array_equal(a.numpy(),b.numpy())


def test_plateau_patience_cooldown_minimum_and_resumption():
    def create():
        p=torch.nn.Parameter(torch.ones(1));o=torch.optim.SGD([p],lr=.2)
        return o,torch.optim.lr_scheduler.ReduceLROnPlateau(o,patience=1,cooldown=1,factor=.5,min_lr=.04)
    optimizer,scheduler=create();metrics=[1.,1.,1.,1.,1.,1.,1.,1.,1.]
    expected=[.2,.2,.1,.1,.1,.05,.05,.05,.04]
    saved=None
    for i,metric in enumerate(metrics):
        scheduler.step(metric);np.testing.assert_allclose(scheduler.get_last_lr(),[expected[i]])
        if i==4:saved=copy.deepcopy((optimizer.state_dict(),scheduler.state_dict()))
    ro,rs=create();rs.load_state_dict(saved[1]);ro.load_state_dict(saved[0])
    for i in range(5,len(metrics)):
        rs.step(metrics[i]);np.testing.assert_allclose(rs.get_last_lr(),[expected[i]])
    assert rs.num_bad_epochs==scheduler.num_bad_epochs and rs.best==1.


def test_scalar_tensor_rates_preserve_group_identity_and_snapshot_independence():
    rate=torch.tensor(.2,dtype=torch.float64);p=torch.nn.Parameter(torch.ones(1))
    o=torch.optim.SGD([p],lr=rate);s=torch.optim.lr_scheduler.ExponentialLR(o,.5)
    initial=o.param_groups[0]['initial_lr'];base=s.base_lrs[0];last=s.get_last_lr()[0]
    assert initial is not rate and base is not initial and last is not rate
    o.step();s.step();assert o.param_groups[0]['lr'] is rate
    np.testing.assert_allclose(rate.item(),.1)
    np.testing.assert_allclose([initial.item(),base.item(),last.item()],[.2,.2,.2])
    s.get_last_lr()[0].fill_(77.);assert rate.item()==.1


def test_lambda_and_chained_schedules_preserve_plain_torch_imports():
    with as_torch():
        from torch import optim
        from torch.optim.lr_scheduler import LambdaLR,ChainedScheduler,ConstantLR,ExponentialLR
        p=torch.nn.Parameter(torch.ones(1));o=optim.SGD([p],lr=.2)
        schedule=LambdaLR(o,lambda epoch:1/(epoch+1))
        for epoch in range(4):
            np.testing.assert_allclose(schedule.get_last_lr(),[.2/(epoch+1)])
            o.step();schedule.step()
        o=optim.SGD([p],lr=.2)
        chained=ChainedScheduler([ConstantLR(o,.5,2),ExponentialLR(o,.5)])
        expected=[.1,.05,.05,.025]
        for rate in expected:
            np.testing.assert_allclose(chained.get_last_lr(),[rate]);o.step();chained.step()


def test_rate_inspection_and_first_step_order_warnings():
    p=torch.nn.Parameter(torch.ones(1));o=torch.optim.SGD([p],lr=.1)
    scheduler=torch.optim.lr_scheduler.StepLR(o,2)
    with pytest.warns(UserWarning,match='before'):scheduler.step()
    with pytest.warns(UserWarning,match='get_last_lr'):scheduler.get_lr()
    with pytest.warns(UserWarning,match='epoch'):scheduler.step(4)
    np.testing.assert_allclose(scheduler.get_last_lr(),[.001])


def test_scheduler_hyperparameter_forward_differentiation_fails_explicitly():
    p=torch.nn.Parameter(torch.ones(1))
    def construct(rate):
        o=torch.optim.SGD([p],lr=rate)
        return torch.optim.lr_scheduler.StepLR(o,2).get_last_lr()[0]
    with pytest.raises(UnsupportedOperation,match='Forward differentiation'):
        torch.func.jvp(construct,(torch.tensor(.1),),(torch.tensor(1.),))


def test_shared_group_rate_updates_keep_backward_mutation_guards():
    rate=torch.tensor(.2,dtype=torch.float64)
    params=[torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    o=torch.optim.SGD([{'params':[p],'lr':rate} for p in params])
    s=torch.optim.lr_scheduler.LambdaLR(o,[lambda n:.8**n,lambda n:.5**n])
    x=torch.ones(1,dtype=torch.float64,requires_grad=True);loss=(x*rate).sum()
    o.step();s.step()
    np.testing.assert_allclose([v.item() for v in s.get_last_lr()],[.1,.1])
    assert all(g['lr'] is rate for g in o.param_groups)
    with pytest.raises(UnsupportedOperation,match='modified in-place'):loss.backward()
