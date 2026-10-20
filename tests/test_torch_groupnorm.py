"""Native-reference GroupNorm values, tapes, selective guards and continuation."""
import copy
import numpy as np
import pytest
import torch
from torchwasm import torch as light


def close(a,b,dtype):
    actual=a.detach().numpy() if hasattr(a,'detach') else np.asarray(a)
    expected=b.detach().numpy() if hasattr(b,'detach') else np.asarray(b)
    tolerance=1e-5 if dtype==np.float32 else 8e-12
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape,groups',[((3,6),2),((2,6,5),1),((2,6,3,4),3),((2,6,2,3,2),6),((2,6),2),((0,6,4),3)])
@pytest.mark.parametrize('layout',['contiguous','fortran','stride'])
@pytest.mark.parametrize('affine',[False,True])
def test_values_gradients_and_retained_backward(dtype,shape,groups,layout,affine):
    rng=np.random.default_rng(1712);raw=rng.normal(size=shape).astype(dtype)
    if layout=='fortran':raw=np.asfortranarray(raw)
    if layout=='stride':raw=np.repeat(raw,2,axis=-1)[...,::2]
    arrays=[raw,rng.normal(size=6).astype(dtype),rng.normal(size=6).astype(dtype)]
    a=[light.from_numpy(v).requires_grad_() for v in arrays];b=[torch.from_numpy(v).requires_grad_() for v in arrays]
    y=light.nn.functional.group_norm(a[0],groups,*a[1:] if affine else (None,None))
    yy=torch.nn.functional.group_norm(b[0],groups,*b[1:] if affine else (None,None))
    close(y,yy,dtype);assert y.shape==shape and y.dtype==np.dtype(dtype)
    for step in range(2):
        seed=rng.normal(size=shape).astype(dtype)
        y.backward(light.tensor(seed),retain_graph=step==0);yy.backward(torch.tensor(seed),retain_graph=step==0)
        for x,xx in zip(a if affine else a[:1],b if affine else b[:1]):close(x.grad,xx.grad,dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape,groups',[((2,6),3),((2,6,4),3),((2,6,2,3),1),((2,6,2,3,2),6),((0,6,4),3)])
@pytest.mark.parametrize('active',['all','input','weight','bias'])
def test_jvp_and_reverse_over_forward_statistics_remain_differentiable(dtype,shape,groups,active):
    rng=np.random.default_rng(1713);arrays=[rng.normal(size=shape).astype(dtype),rng.normal(size=6).astype(dtype),rng.normal(size=6).astype(dtype)]
    arrays += [rng.normal(size=v.shape).astype(dtype) for v in arrays]
    records=[]
    for api in [light,torch]:
        v=[api.tensor(a,requires_grad=True) for a in arrays]
        if active=='all':fn=lambda x,w,b:api.nn.functional.group_norm(x,groups,w,b);p,t=api.func.jvp(fn,tuple(v[:3]),tuple(v[3:]))
        else:
            i=['input','weight','bias'].index(active)
            def fn(z):
                args=v[:3].copy();args[i]=z
                return api.nn.functional.group_norm(args[0],groups,*args[1:])
            p,t=api.func.jvp(fn,(v[i],),(v[i+3],))
        (p.square().sum()+t.square().sum()).backward()
        records.append([p,t,*[z.grad for z in v if z.grad is not None]])
    assert len(records[0])==len(records[1])
    for x,xx in zip(*records):close(x,xx,dtype)


@pytest.mark.parametrize('needs',[(True,True,True),(True,False,False),(False,True,False),(False,False,True)])
@pytest.mark.parametrize('changed',['x','w','b','output'])
def test_saved_input_weight_but_not_bias_output_version_guards(needs,changed):
    outcomes=[]
    for api in [light,torch]:
        v=[api.ones(2,6,4,requires_grad=needs[0]),api.ones(6,requires_grad=needs[1]),api.zeros(6,requires_grad=needs[2])]
        y=api.nn.functional.group_norm(v[0],3,*v[1:]);target=y if changed=='output' else v[['x','w','b'].index(changed)]
        target=target.detach()
        with api.no_grad():target.add_(1)
        try:y.sum().backward();outcomes.append('ok')
        except RuntimeError:outcomes.append('RuntimeError')
    assert outcomes[0]==outcomes[1]


@pytest.mark.parametrize('shape,groups,error',[((1,6),6,ValueError),((2,6,0),3,RuntimeError),((2,0,4),1,RuntimeError),((2,6,4),0,ZeroDivisionError),((2,6,4),4,RuntimeError),((2,6,4),True,TypeError),((6,),3,RuntimeError)])
def test_invalid_input_boundaries(shape,groups,error):
    for api in [light,torch]:
        with pytest.raises(error):api.nn.functional.group_norm(api.ones(shape),groups)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_adam_model_checkpoint_resume_and_train_eval_identity(dtype):
    rng=np.random.default_rng(1714);tdtype=torch.float32 if dtype==np.float32 else torch.float64
    models=[api.nn.Sequential(api.nn.Linear(6,6,dtype=dt),api.nn.GroupNorm(3,6,dtype=dt),api.nn.SiLU(),api.nn.Linear(6,2,dtype=dt)) for api,dt in [(light,np.dtype(dtype)),(torch,tdtype)]]
    for params in zip(*[list(m.parameters()) for m in models]):
        raw=(rng.normal(size=tuple(params[0].shape))*.3).astype(dtype)
        for p,api in zip(params,[light,torch]):
            with api.no_grad():p.copy_(api.tensor(raw))
    optimizers=[api.optim.Adam(m.parameters(),lr=.003) for api,m in zip([light,torch],models)]
    for step in range(7):
        raw=rng.normal(size=(5,6)).astype(dtype);target=rng.normal(size=(5,2)).astype(dtype);losses=[]
        for api,m,opt in zip([light,torch],models,optimizers):
            opt.zero_grad();loss=(m(api.tensor(raw))-api.tensor(target)).square().mean();loss.backward();opt.step();losses.append(loss)
        close(*losses,dtype)
        for x,xx in zip(models[0].parameters(),models[1].parameters()):close(x,xx,dtype)
        if step==2:
            for api,i,dt in [(light,0,np.dtype(dtype)),(torch,1,tdtype)]:
                state=copy.deepcopy(models[i].state_dict());optstate=copy.deepcopy(optimizers[i].state_dict())
                models[i]=api.nn.Sequential(api.nn.Linear(6,6,dtype=dt),api.nn.GroupNorm(3,6,dtype=dt),api.nn.SiLU(),api.nn.Linear(6,2,dtype=dt));models[i].load_state_dict(state)
                optimizers[i]=api.optim.Adam(models[i].parameters(),lr=99);optimizers[i].load_state_dict(optstate)
    for api,m in zip([light,torch],models):
        x=api.tensor(raw);m.train();before=m(x);m.eval();after=m(x);close(before,after,dtype)
        norm=m[1];assert list(norm.state_dict())==['weight','bias'];assert not list(norm.buffers())
        norm.reset_parameters();assert norm.weight.tolist()==[1.]*6 and norm.bias.tolist()==[0.]*6
