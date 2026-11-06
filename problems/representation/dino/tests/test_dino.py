import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from ..toy import fixture,probabilities,scalar_loss
from ..experiment import run


@pytest.mark.t0
def test_shape_dtype_strides_and_no_mutation(impl):
    """Teacher probabilities retain [2,B,K], loss is scalar, and all floating strided inputs stay intact."""
    for dtype in (torch.float32,torch.float64):
        s,t,c=fixture(dtype);s=s[:,:,::2];t=t[:,:,::2];c=c[::2]
        before=[v.clone() for v in (s,t,c)]
        q=impl.teacher_probabilities(t,c,.2);loss=impl.dino_loss(s,t,c,.3,.2)
        assert q.shape==t.shape and q.dtype==dtype and not q.requires_grad
        assert loss.shape==() and loss.dtype==dtype and loss.device==s.device
        close(q.sum(-1),torch.ones((2,4),dtype=dtype),atol=2e-6,rtol=2e-6)
        for a,b in zip((s,t,c),before):close(a,b)


@pytest.mark.t0
def test_invalid_input_contract(impl):
    """Exactly two global teacher views align with the first two of at least two student views."""
    s,t,c=fixture()
    for bad in (t[:1],t[:,:0],t.flatten(),t.float(),t*float('nan')):
        with pytest.raises(ValueError):impl.teacher_probabilities(bad,c)
    for bad in (s[:1],s[:,:2],s.long(),s*float('inf')):
        with pytest.raises(ValueError):impl.dino_loss(bad,t,c)
    for bad in (c[None],c[:1],c.float()):
        with pytest.raises(ValueError):impl.update_center(bad,t)
    for scalar in (0,-1,True,float('nan'),torch.tensor(.1)):
        with pytest.raises(ValueError):impl.teacher_probabilities(t,c,scalar)
        with pytest.raises(ValueError):impl.dino_loss(s,t,c,scalar)
    for scalar in (-.1,1.1,True,float('inf')):
        with pytest.raises(ValueError):impl.update_center(c,t,scalar)


@pytest.mark.t1
def test_scalar_teacher_and_all_unequal_view_terms(impl):
    """DINO centers only the teacher and averages the 2*(V-1) nonmatching-view cross-entropies."""
    s,t,c=fixture()
    extra=(torch.arange(24,dtype=s.dtype).reshape(2,4,3)*.19+.4).cos()*.35
    s=torch.cat((s,extra),dim=0)
    for st,tt in ((.1,.04),(.4,.3)):
        close(impl.teacher_probabilities(t,c,tt),torch.tensor(probabilities(t.tolist(),c.tolist(),tt),dtype=t.dtype),atol=1e-12,rtol=1e-11)
        for v in (2,3,5):
            expected=scalar_loss(s[:v].tolist(),t.tolist(),c.tolist(),st,tt)
            close(impl.dino_loss(s[:v],t,c,st,tt),torch.tensor(expected,dtype=t.dtype),atol=1e-12,rtol=1e-11)
    uniform=torch.zeros_like(t);zero=torch.zeros_like(c)
    close(impl.dino_loss(torch.zeros_like(s),uniform,zero),torch.tensor(math.log(3),dtype=t.dtype),atol=1e-12,rtol=0)


@pytest.mark.t2
def test_independent_student_gradients_and_stopped_teacher_center(impl):
    """A student view receives the mean teacher distribution from all allowed global targets only."""
    s,t,c=fixture()
    extra=(torch.arange(24,dtype=s.dtype).reshape(2,4,3)*.19+.4).cos()*.35
    s=torch.cat((s,extra),dim=0).requires_grad_();t.requires_grad_();c.requires_grad_()
    q=impl.teacher_probabilities(t,c,.2);assert not q.requires_grad
    loss=impl.dino_loss(s,t,c,.3,.2)
    ds,dt,dc=torch.autograd.grad(loss,(s,t,c),allow_unused=True)
    assert ds is not None and dt is None and dc is None
    teacher=probabilities(t.detach().tolist(),c.detach().tolist(),.2);expected=[]
    for v,view in enumerate(s.detach().tolist()):
        allowed=[i for i in range(2) if i!=v];out=[]
        for i,row in enumerate(view):
            peak=max(row);exp=[math.exp((x-peak)/.3) for x in row];total=sum(exp)
            out.append([(len(allowed)*w/total-sum(teacher[a][i][k] for a in allowed))/(2*(s.shape[0]-1)*s.shape[1]*.3) for k,w in enumerate(exp)])
        expected.append(out)
    close(ds,torch.tensor(expected,dtype=s.dtype),atol=1e-12,rtol=1e-10)
    parent=torch.stack((s.detach(),torch.zeros_like(s)),dim=-1).reshape(5,4,6).requires_grad_()
    strided_loss=impl.dino_loss(parent[:,:,::2],t.detach(),c.detach(),.3,.2)
    assert strided_loss.requires_grad, 'strided student views must retain their gradient paths'
    gp=torch.autograd.grad(strided_loss,parent)[0]
    close(gp[:,:,::2],ds,atol=1e-12,rtol=1e-11)
    close(gp[:,:,1::2],torch.zeros_like(ds),atol=0,rtol=0)


@pytest.mark.t2
def test_center_sharpening_shift_and_large_logit_stability(impl):
    """A common teacher/center feature shift cancels; sharpening changes probabilities and large logits stay finite."""
    s,t,c=fixture();q=impl.teacher_probabilities(t,c,.3)
    offset=torch.tensor([4.,-2.,7.],dtype=c.dtype)
    close(impl.teacher_probabilities(t+offset,c+offset,.3),q,atol=1e-12,rtol=1e-11)
    sharp=impl.teacher_probabilities(t,c,.03)
    assert float(sharp.max(-1).values.mean())>float(q.max(-1).values.mean())
    large=t*10000;s=s*10000
    assert bool(torch.isfinite(impl.teacher_probabilities(large,c)).all())
    assert bool(torch.isfinite(impl.dino_loss(s,large,c)))
    close(impl.dino_loss(s,t,c,.3,.2),impl.dino_loss(s+1000,t,c,.3,.2),atol=1e-9,rtol=1e-11)


@pytest.mark.t2
def test_center_ema_raw_statistics_and_independent_storage(impl):
    """The center tracks raw logits over both global views and examples, not teacher probabilities."""
    _,t,c=fixture();t.requires_grad_();c.requires_grad_()
    rows=t.detach().tolist();batch=[sum(rows[v][i][k] for v in range(2) for i in range(4))/8 for k in range(3)]
    for m in (0.,.2,.9,1.):
        got=impl.update_center(c,t,m);expected=[m*x+(1-m)*y for x,y in zip(c.detach().tolist(),batch)]
        close(got,torch.tensor(expected,dtype=c.dtype),atol=1e-12,rtol=1e-12);assert not got.requires_grad
        before=c.detach().clone();got[0]+=1;close(c,before)


@pytest.mark.t3
def test_real_student_training_and_center_tracking(impl,artifact):
    """Eighty real SGD steps fit global/local student views while raw-logit center EMA follows its closed form."""
    result=run(impl.teacher_probabilities,impl.dino_loss,impl.update_center);artifact('dino_training',result);m=result['metrics']
    assert m['updates']==80 and m['trainable_parameters']==54 and m['training_accuracy']==1
    assert m['scalar_error']<1e-11 and m['center_error']<1e-12 and m['probability_error']<1e-12
    assert m['teacher_stopped'] and m['center_detached']
    assert m['final_loss']<.08 and m['final_loss']<.1*m['initial_loss']
