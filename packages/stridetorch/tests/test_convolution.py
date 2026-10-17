"""Independent scalar convolution values/VJPs and bilinear finite differences."""
import copy,itertools,math,weakref,gc
import numpy as np
import pytest
from stridetorch import torch,as_torch,UnsupportedOperation


@pytest.fixture(params=['direct','matrix','auto'],autouse=True)
def convolution_algorithm(request,monkeypatch):
    from stridetorch import _convolution
    monkeypatch.setattr(_convolution,'_ALGORITHM',request.param)


def oracle(x,w,b,stride,padding,dilation,groups,seed=None):
    dim=w.ndim-2;n,channels,*spatial=x.shape;outputs=w.shape[0];kernel=w.shape[2:]
    effective=[d*(k-1)+1 for d,k in zip(dilation,kernel)]
    pairs=tuple((d*(k-1)//2,d*(k-1)-d*(k-1)//2) for d,k in zip(dilation,kernel)) if padding=='same' else tuple((p,p) for p in padding)
    out=tuple((size+a+bb-e)//s+1 for size,(a,bb),e,s in zip(spatial,pairs,effective,stride))
    y=np.zeros((n,outputs,*out),dtype=np.float64);dx=np.zeros(x.shape,dtype=np.float64);dw=np.zeros(w.shape,dtype=np.float64);db=np.zeros(outputs,dtype=np.float64)
    for batch in range(n):
        for o in range(outputs):
            group=o//(outputs//groups)
            for pos in np.ndindex(out):
                key=(batch,o,*pos);value=float(b[o]) if b is not None else 0.
                if seed is not None:db[o]+=float(seed[key])
                for i in range(channels//groups):
                    channel=group*(channels//groups)+i
                    for k in np.ndindex(kernel):
                        source=tuple(p*s+kk*d-pair[0] for p,s,kk,d,pair in zip(pos,stride,k,dilation,pairs))
                        if any(v<0 or v>=size for v,size in zip(source,spatial)):continue
                        xx=(batch,channel,*source);ww=(o,i,*k);value+=float(x[xx])*float(w[ww])
                        if seed is not None:dx[xx]+=float(seed[key])*float(w[ww]);dw[ww]+=float(seed[key])*float(x[xx])
                y[key]=value
    return y,dx,dw,db


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('groups',[1,2])
@pytest.mark.parametrize('case',['valid','strided','same'])
def test_values_and_all_pullbacks_match_independent_scalar_loops(dim,dtype,groups,case):
    rng=np.random.default_rng(1905);shape=(2,2,*([4]*dim));arrays=[rng.normal(size=shape).astype(dtype),rng.normal(size=(4,2//groups,*([2]*dim))).astype(dtype),rng.normal(size=4).astype(dtype)]
    stride=(2 if case=='strided' else 1,)*dim;dilation=(2 if case=='strided' else 1,)*dim;padding='same' if case=='same' else (1 if case=='strided' else 0,)*dim
    x,w,b=[torch.from_numpy(np.asfortranarray(a)).requires_grad_() for a in arrays]
    result=getattr(torch.nn.functional,f'conv{dim}d')(x,w,b,stride,padding,dilation,groups);seed=rng.normal(size=result.shape).astype(dtype);result.backward(torch.tensor(seed))
    expected=oracle(*arrays,stride,padding,dilation,groups,seed)
    for actual,value in zip([result,x.grad,w.grad,b.grad],expected):np.testing.assert_allclose(actual.numpy(),value,rtol=2e-5 if dtype==torch.float32 else 1e-11,atol=8e-6 if dtype==torch.float32 else 2e-12)


@pytest.mark.parametrize('dim',[1,2,3])
def test_primal_and_bilinear_jvp_reverse_finite_differences(dim):
    rng=np.random.default_rng(1906);arrays=[rng.normal(size=(1,2,*([3]*dim))),rng.normal(size=(2,1,*([2]*dim))),rng.normal(size=2)];arrays +=[rng.normal(size=a.shape) for a in arrays];v=[torch.tensor(a,requires_grad=True) for a in arrays]
    fn=lambda x,w,b:getattr(torch.nn.functional,f'conv{dim}d')(x,w,b,padding='same',groups=2)
    assert torch.autograd.gradcheck(fn,tuple(v[:3]))
    p,t=torch.func.jvp(fn,tuple(v[:3]),tuple(v[3:]));(p+t).sum().backward()
    def objective(a):
        x,w,b,dx,dw,db=a;stride=(1,)*dim;dilation=stride
        return sum(oracle(xx,ww,bb,stride,'same',dilation,2)[0].sum() for xx,ww,bb in [(x,w,b),(dx,w,None),(x,dw,db)])
    for i,var in enumerate(v):
        expected=np.empty_like(arrays[i])
        for index in np.ndindex(var.shape):
            plus=[a.copy() for a in arrays];minus=[a.copy() for a in arrays];plus[i][index]+=1e-6;minus[i][index]-=1e-6
            expected[index]=(objective(plus)-objective(minus))/2e-6
        np.testing.assert_allclose(var.grad.numpy(),expected,rtol=4e-6,atol=4e-6)


@pytest.mark.parametrize('mode',['reflect','replicate','circular'])
@pytest.mark.parametrize('dim',[1,2,3])
def test_module_padding_adjoint_matches_finite_difference(mode,dim):
    rng=np.random.default_rng(1907);model=getattr(torch.nn,f'Conv{dim}d')(2,2,2,padding='same',groups=2,padding_mode=mode,dtype=torch.float64);raw=rng.normal(size=(1,2,*([3]*dim)));x=torch.tensor(raw,requires_grad=True)
    model(x).square().sum().backward();expected=np.empty_like(raw)
    for index in np.ndindex(raw.shape):
        plus=raw.copy();minus=raw.copy();plus[index]+=1e-6;minus[index]-=1e-6
        expected[index]=(model(torch.tensor(plus)).square().sum().item()-model(torch.tensor(minus)).square().sum().item())/2e-6
    np.testing.assert_allclose(x.grad.numpy(),expected,rtol=4e-6,atol=4e-6)


def test_standard_aliases_module_parameters_rng_reset_and_dtype():
    with as_torch():
        from torch.nn import Conv1d,Conv2d,Conv3d
        from torch.nn.modules.conv import Conv1d as Alias,_ConvNd
        assert Alias is Conv1d
        for dim,cls in enumerate([Conv1d,Conv2d,Conv3d],1):
            torch.manual_seed(1908);a=cls(2,4,2,groups=2,bias=False,dtype=torch.float64);torch.manual_seed(1908);b=cls(2,4,2,groups=2,bias=False,dtype=torch.float64)
            assert isinstance(a,_ConvNd) and list(a.state_dict())==['weight']
            np.testing.assert_array_equal(a.weight.numpy(),b.weight.numpy());state=copy.deepcopy(a.state_dict());a.float();assert a.weight.dtype==torch.float32;a.double();a.load_state_dict(state)
            raw=torch.ones((1,2,*([3]*dim)),dtype=torch.float64);out=a(raw);a.eval();np.testing.assert_array_equal(a(raw).numpy(),out.numpy())
            a.reset_parameters();assert np.max(np.abs(a.weight.numpy()))<=(2**dim)**-.5


def test_bias_only_tape_releases_unneeded_tensor_objects_but_guards_saved_versions():
    x=torch.ones(2,2,4);w=torch.ones(3,2,2);b=torch.zeros(3,requires_grad=True);refs=[weakref.ref(x),weakref.ref(w)];out=torch.nn.functional.conv1d(x,w,b);del x,w;gc.collect();assert all(r() is None for r in refs);out.sum().backward();assert b.grad.tolist()==[6.]*3
    x=torch.ones(2,2,4);w=torch.ones(3,2,2);out=torch.nn.functional.conv1d(x,w,b)
    with torch.no_grad():x.add_(1)
    with pytest.raises(RuntimeError):out.sum().backward()


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('empty',['batch','channels'])
def test_empty_values_gradients_and_jvp(dim,empty):
    shape=(0,2,*([3]*dim)) if empty=='batch' else (2,0,*([3]*dim));x=torch.ones(shape,requires_grad=True);w=torch.ones((4,shape[1],*([1]*dim)),requires_grad=True);b=torch.zeros(4,requires_grad=True)
    p,t=torch.func.jvp(lambda x,w,b:getattr(torch.nn.functional,f'conv{dim}d')(x,w,b),(x,w,b),(torch.ones_like(x),torch.ones_like(w),torch.ones_like(b)));(p.sum()+t.sum()).backward()
    assert p.shape==(shape[0],0 if not shape[1] else 4,*([3]*dim));assert x.grad.shape==x.shape and w.grad.shape==w.shape;np.testing.assert_array_equal(w.grad.numpy(),np.zeros(w.shape));assert b.grad.tolist()==[0.]*4
    expected=tuple(math.prod(max(1,n) for n in p.shape[i+1:]) for i in range(p.ndim))
    assert p.stride()==t.stride()==expected


@pytest.mark.parametrize('value',[float('inf'),float('nan')])
def test_explicit_zero_tangent_terms_preserve_native_nonfinite_classes(value):
    x=torch.full((1,1,3),value,dtype=torch.float64,requires_grad=True);w=torch.ones(1,1,1,dtype=torch.float64,requires_grad=True);b=torch.ones(1,dtype=torch.float64)
    p,t=torch.func.jvp(lambda bias:torch.nn.functional.conv1d(x,w,bias),(b,),(torch.ones_like(b),));assert np.isnan(t.numpy()).all();t.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),0);np.testing.assert_array_equal(w.grad.numpy(),0)


@pytest.mark.parametrize('kwargs',[{'stride':0},{'dilation':0},{'padding':-1},{'groups':0},{'groups':3},{'padding':'same','stride':2}])
def test_invalid_geometry_is_explicit(kwargs):
    with pytest.raises((RuntimeError,ValueError)):torch.nn.functional.conv2d(torch.ones(1,2,4,4),torch.ones(4,2,2,2),**kwargs)


@pytest.mark.parametrize('mode',['reflect','circular'])
def test_nonzero_padding_limits(mode):
    model=torch.nn.Conv1d(1,1,1,padding=3 if mode=='reflect' else 4,padding_mode=mode)
    with pytest.raises(RuntimeError):model(torch.ones(1,1,3))


def test_unimplemented_dtype_and_unknown_argument_fail_closed():
    with pytest.raises(UnsupportedOperation):torch.nn.functional.conv1d(torch.ones(1,1,3,dtype=torch.float16),torch.ones(1,1,1,dtype=torch.float16))
    with pytest.raises(TypeError):torch.nn.functional.conv2d(torch.ones(1,1,3,3),torch.ones(1,1,1,1),output_padding=1)


def test_complete_cnn_checkpoint_resume_matches_uninterrupted_updates():
    rng=np.random.default_rng(1910)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.conv=torch.nn.Conv2d(2,4,3,padding=1,dtype=torch.float64);self.norm=torch.nn.GroupNorm(2,4,dtype=torch.float64);self.act=torch.nn.SiLU();self.project=torch.nn.Conv2d(4,3,1,dtype=torch.float64)
        def forward(self,x):return self.project(self.act(self.norm(self.conv(x)))).mean((2,3))
    a=Model();b=Model();b.load_state_dict(copy.deepcopy(a.state_dict()));oa=torch.optim.AdamW(a.parameters(),lr=.003);ob=torch.optim.AdamW(b.parameters(),lr=.003)
    for step in range(6):
        x=torch.tensor(rng.normal(size=(4,2,4,4)));labels=torch.tensor(rng.integers(0,3,size=4,dtype=np.int64))
        for model,opt in [(a,oa),(b,ob)]:opt.zero_grad();torch.nn.functional.cross_entropy(model(x),labels,label_smoothing=.1).backward();opt.step()
        for p,q in zip(a.parameters(),b.parameters()):np.testing.assert_allclose(p.numpy(),q.numpy(),rtol=1e-11,atol=1e-12)
        if step==2:
            state=copy.deepcopy(b.state_dict());os=copy.deepcopy(ob.state_dict());b=Model();b.load_state_dict(state);ob=torch.optim.AdamW(b.parameters(),lr=99);ob.load_state_dict(os)


def test_workspace_budget_falls_back_without_calling_matrix_helpers(monkeypatch):
    from stridetorch import _convolution as c
    monkeypatch.setattr(c,'_MATRIX_WORKSPACE_BYTES',1)
    def forbidden(*args,**kwargs):raise AssertionError('Budgeted operation must not materialize matrix workspace')
    monkeypatch.setattr(c,'_matrix_forward',forbidden);monkeypatch.setattr(c,'_matrix_input_gradient',forbidden);monkeypatch.setattr(c,'_matrix_weight_gradient',forbidden)
    x=torch.ones(2,4,8,8,requires_grad=True);w=torch.ones(8,4,3,3,requires_grad=True);out=torch.nn.functional.conv2d(x,w,padding=1);out.sum().backward()
    assert out.shape==(2,8,8,8) and x.grad.shape==x.shape and w.grad.shape==w.shape
