import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.supermimic.experiment import run


def scalar_return(rewards, termination, bootstrap, gamma):
    total = 0.
    for b in range(len(rewards)):
        weight = 1.; value = 0.
        for t in range(len(rewards[b])):
            value += weight * rewards[b][t]
            weight *= gamma * (1-termination[b][t])
        total += value + weight * bootstrap[b]
    return total / len(rewards)


@pytest.mark.t0
def test_contract_and_immutability(impl):
    """Actor return is scalar; critic targets retain batch and horizon axes."""
    for dtype in (torch.float32, torch.float64):
        r = torch.arange(6,dtype=dtype).reshape(2,3); d = torch.full_like(r,.1)
        v = torch.arange(8,dtype=dtype).reshape(2,4); before = [x.clone() for x in (r,d,v)]
        objective = impl.imagined_return(r,d,v[:,-1],.9)
        targets = impl.lambda_return_targets(r,d,v,.9,.8)
        assert objective.shape == () and targets.shape == r.shape
        assert objective.dtype == targets.dtype == dtype and targets.device == r.device
        for x,y in zip((r,d,v),before): close(x,y)


@pytest.mark.t0
def test_invalid_rollouts(impl):
    """Reject inconsistent horizons, invalid probabilities and invalid discounts."""
    r = torch.ones(2,3); d = torch.zeros_like(r); v = torch.zeros(2,4)
    for gamma in (-.1,1.,float('nan')):
        with pytest.raises(ValueError): impl.imagined_return(r,d,v[:,-1],gamma)
    for bad in (d[:1], d+1.1, d-.1, d*float('nan')):
        with pytest.raises(ValueError): impl.imagined_return(r,bad,v[:,-1],.9)
    with pytest.raises(ValueError): impl.imagined_return(r,d,v,.9)
    with pytest.raises(ValueError): impl.lambda_return_targets(r,d,v[:,:3],.9,.8)
    for lam in (-.1,1.1,float('inf')):
        with pytest.raises(ValueError): impl.lambda_return_targets(r,d,v,.9,lam)


@pytest.mark.t1
def test_soft_survival_scalar_oracle(impl):
    """Reward from a transition is counted before that transition's termination probability."""
    r = torch.tensor([[1.,2.,4.],[-2.,3.,1.]],dtype=torch.float64)
    d = r.new_tensor([[.2,.5,.1],[0.,.4,1.]])
    b = r.new_tensor([5.,-3.])
    expected = scalar_return(r.tolist(),d.tolist(),b.tolist(),.8)
    close(impl.imagined_return(r,d,b,.8),r.new_tensor(expected))


@pytest.mark.t1
def test_lambda_recursion_independent_oracle(impl):
    """Lambda targets use the next-state critic and recurse independently for every batch row."""
    r = torch.tensor([[1.,2.,3.],[4.,-1.,2.]],dtype=torch.float64)
    d = r.new_tensor([[0.,.25,.5],[1.,.1,0.]])
    v = r.new_tensor([[99.,.5,2.,4.],[-50.,2.,1.,3.]])
    expected = []
    for rewards,done,values in zip(r.tolist(),d.tolist(),v.tolist()):
        row = [0.]*3; tail = values[-1]
        for t in (2,1,0):
            tail = rewards[t] + .9*(1-done[t])*(.3*values[t+1]+.7*tail)
            row[t] = tail
        expected.append(row)
    close(impl.lambda_return_targets(r,d,v,.9,.7),r.new_tensor(expected))


@pytest.mark.t2
def test_terminal_transition_keeps_reward(impl):
    """Hard termination preserves its arrival reward and blocks every later reward and bootstrap."""
    r = torch.tensor([[2.,3.,100.]],dtype=torch.float64,requires_grad=True)
    d = r.new_tensor([[0.,1.,0.]])
    b = r.new_tensor([1000.],requires_grad=True)
    out = impl.imagined_return(r,d,b,.5)
    close(out,r.new_tensor(3.5))
    gr,gb = torch.autograd.grad(out,(r,b))
    close(gr,r.new_tensor([[1.,.5,0.]])); close(gb,torch.zeros_like(b))


@pytest.mark.t2
def test_actor_gradients_include_soft_survival_and_bootstrap(impl):
    """Finite differences check all reward, survival and terminal-value gradient paths."""
    r = torch.tensor([[.5,1.,2.],[-1.,3.,.2]],dtype=torch.float64,requires_grad=True)
    d = torch.tensor([[.2,.3,.4],[.1,.6,.2]],dtype=r.dtype,requires_grad=True)
    b = torch.tensor([2.,-1.],dtype=r.dtype,requires_grad=True)
    grads = torch.autograd.grad(impl.imagined_return(r,d,b,.91),(r,d,b))
    arrays = [r.detach(),d.detach(),b.detach()]
    eps = 1e-6
    for which, (value,gradient) in enumerate(zip(arrays,grads)):
        numerical = torch.zeros_like(value)
        for flat in range(value.numel()):
            plus = [x.clone() for x in arrays]; minus = [x.clone() for x in arrays]
            plus[which].reshape(-1)[flat] += eps; minus[which].reshape(-1)[flat] -= eps
            upper = scalar_return(plus[0].tolist(),plus[1].tolist(),plus[2].tolist(),.91)
            lower = scalar_return(minus[0].tolist(),minus[1].tolist(),minus[2].tolist(),.91)
            numerical.reshape(-1)[flat] = (upper-lower)/(2*eps)
        close(gradient,numerical,atol=1e-8,rtol=1e-7)


@pytest.mark.t2
def test_lambda_endpoints_and_detachment(impl):
    """Lambda zero gives one-step targets; lambda one gives the full bootstrapped rollout; targets are frozen."""
    r = torch.tensor([[1.,2.,3.]],dtype=torch.float64,requires_grad=True)
    d = torch.tensor([[.1,.2,.3]],dtype=r.dtype,requires_grad=True)
    v = torch.tensor([[7.,8.,9.,10.]],dtype=r.dtype,requires_grad=True)
    one_step = impl.lambda_return_targets(r,d,v,.9,0.)
    close(one_step,r.detach()+.9*(1-d.detach())*v.detach()[:,1:])
    full = impl.lambda_return_targets(r,d,v,.9,1.)
    close(full[:,0].mean(),impl.imagined_return(r,d,v[:,-1],.9).detach())
    assert not one_step.requires_grad and not full.requires_grad
    altered = v.detach().clone(); altered[:,0] = -1000
    close(impl.lambda_return_targets(r,d,altered,.9,.6),impl.lambda_return_targets(r,d,v,.9,.6))


@pytest.mark.t2
def test_zero_discount_single_step_and_batch_repetition(impl):
    """Zero discount isolates the first reward; a one-step objective still bootstraps after survival."""
    r = torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64); d = torch.full_like(r,.2); b = r.new_tensor([5.,6.])
    close(impl.imagined_return(r,d,b,0.),r[:,0].mean())
    close(impl.imagined_return(r[:,:1],d[:,:1],b,.9),(r[:,0]+.9*.8*b).mean())
    close(impl.imagined_return(r.repeat(3,1),d.repeat(3,1),b.repeat(3),.9),impl.imagined_return(r,d,b,.9))
    v = torch.ones(2,3,dtype=r.dtype)
    close(impl.lambda_return_targets(r,d,v,0.,.7),r)


@pytest.mark.t3
def test_world_model_policy_improvement(impl, artifact):
    """The submitted return drives policy gradients through fixed dynamics and supplies consistent critic targets."""
    result = run(impl.imagined_return,impl.lambda_return_targets)
    artifact('supermimic_imagined_control',result)
    m = result['metrics']
    assert m['final_return'] > m['initial_return'] + 1.
    assert m['final_tracking_error'] < .2 * m['initial_tracking_error']
    assert m['lambda_one_consistency_error'] < 1e-10
