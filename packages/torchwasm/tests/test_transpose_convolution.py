"""Scalar scatter oracle, adjoint identities and decoder-training workflows."""
import copy
import gc
import io
import itertools
import math
import weakref
import numpy as np
import pytest
from torchwasm import as_torch,torch,UnsupportedOperation


@pytest.fixture(params=['direct','matrix','auto','direct_no_pad','matrix_no_pad'],autouse=True)
def algorithm(request,monkeypatch):
    from torchwasm import _transpose_convolution as tc
    monkeypatch.setattr(tc,'_ALGORITHM',request.param.split('_')[0])
    if request.param.endswith('_no_pad'):monkeypatch.setattr(tc,'_PAD_WORKSPACE_BYTES',0)


def oracle(x,w,b,stride,padding,output_padding,dilation,groups,seed=None):
    dim=w.ndim-2;n,ci,*spatial=x.shape;co=w.shape[1]*groups;kernel=w.shape[2:]
    out=tuple((size-1)*s-2*p+d*(k-1)+1+op for size,s,p,d,k,op in zip(spatial,stride,padding,dilation,kernel,output_padding))
    y=np.zeros((n,co,*out));dx=np.zeros(x.shape);dw=np.zeros(w.shape);db=np.zeros(co)
    if b is not None:y+=b.reshape(1,co,*([1]*dim))
    for batch in range(n):
        for i in range(ci):
            group=i//(ci//groups)
            for loc in np.ndindex(tuple(spatial)):
                for o in range(co//groups):
                    for tap in np.ndindex(kernel):
                        pos=tuple(a*s-p+k*d for a,s,p,k,d in zip(loc,stride,padding,tap,dilation))
                        if any(p<0 or p>=v for p,v in zip(pos,out)):continue
                        xx=(batch,i,*loc);ww=(i,o,*tap);yy=(batch,group*(co//groups)+o,*pos)
                        y[yy]+=float(x[xx])*float(w[ww])
                        if seed is not None:
                            dx[xx]+=float(seed[yy])*float(w[ww]);dw[ww]+=float(seed[yy])*float(x[xx])
    if seed is not None:db=seed.astype(np.float64).sum(axis=(0,*range(2,dim+2)))
    return y,dx,dw,db


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('groups',[1,2])
@pytest.mark.parametrize('case',['regular','stride','beyond_stride','cropped','anisotropic'])
def test_values_and_all_gradients_against_scalar_scatter(dim,dtype,groups,case):
    rng=np.random.default_rng(4811)
    arrays=[rng.normal(size=(1,2,*([3]*dim))).astype(dtype),rng.normal(size=(2,4//groups,*([2]*dim))).astype(dtype),rng.normal(size=4).astype(dtype)]
    values=[torch.from_numpy(np.asfortranarray(a)).requires_grad_() for a in arrays]
    s=(2 if case=='stride' else 1,)*dim;p=(1 if case=='cropped' else 0,)*dim;op=(2 if case=='beyond_stride' else 1 if case=='stride' else 0,)*dim;d=(3 if case=='beyond_stride' else 1,)*dim
    if case=='anisotropic':s,p,op,d=(2,1,3)[:dim],(1,0,2)[:dim],(1,2,0)[:dim],(1,3,1)[:dim]
    y=getattr(torch.nn.functional,f'conv_transpose{dim}d')(*values,stride=s,padding=p,output_padding=op,dilation=d,groups=groups)
    seed=rng.normal(size=y.shape).astype(dtype);y.backward(torch.tensor(seed))
    for actual,expected in zip([y,*[v.grad for v in values]],oracle(*arrays,s,p,op,d,groups,seed)):
        np.testing.assert_allclose(actual.numpy(),expected,rtol=2e-5 if dtype==torch.float32 else 4e-12,atol=8e-6 if dtype==torch.float32 else 4e-12)


@pytest.mark.parametrize('dim',[1,2,3])
def test_adjoint_identity_with_original_convolution(dim):
    rng=np.random.default_rng(4812)
    x=torch.tensor(rng.normal(size=(2,4,*([5]*dim))),requires_grad=True)
    w=torch.tensor(rng.normal(size=(6,2,*([2]*dim))),requires_grad=True)
    conv=getattr(torch.nn.functional,f'conv{dim}d');transpose=getattr(torch.nn.functional,f'conv_transpose{dim}d')
    y=conv(x,w,stride=2,padding=1,groups=2);g=torch.tensor(rng.normal(size=y.shape))
    target=transpose(g,w,stride=2,padding=1,output_padding=1,groups=2)
    assert target.shape==x.shape
    np.testing.assert_allclose((y*g).sum().item(),(x*target).sum().item(),atol=5e-12)
    y.backward(g);np.testing.assert_allclose(x.grad.numpy(),target.numpy(),atol=5e-12)


@pytest.mark.parametrize('dim',[1,2,3])
def test_jvp_reverse_finite_differences_and_gradcheck(dim):
    rng=np.random.default_rng(4813)
    arrays=[rng.normal(size=(1,2,*([2]*dim))),rng.normal(size=(2,1,*([2]*dim))),rng.normal(size=2)]
    arrays += [rng.normal(size=a.shape) for a in arrays]
    values=[torch.tensor(a,requires_grad=True) for a in arrays]
    f=lambda x,w,b:getattr(torch.nn.functional,f'conv_transpose{dim}d')(x,w,b,stride=2,padding=1,output_padding=1,groups=2)
    assert torch.autograd.gradcheck(f,tuple(values[:3]))
    primal,tangent=torch.func.jvp(f,tuple(values[:3]),tuple(values[3:]))
    seed=rng.normal(size=primal.shape);seed2=rng.normal(size=primal.shape)
    (primal*torch.tensor(seed)+tangent*torch.tensor(seed2)).sum().backward()
    def objective(raw):
        x,w,b,dx,dw,db=raw;s=(2,)*dim;p=(1,)*dim;op=p;d=(1,)*dim
        y=oracle(x,w,b,s,p,op,d,2)[0]
        t=oracle(dx,w,None,s,p,op,d,2)[0]+oracle(x,dw,db,s,p,op,d,2)[0]
        return float(np.sum(y*seed+t*seed2))
    for position,value in enumerate(values):
        expected=np.empty_like(arrays[position])
        for index in np.ndindex(expected.shape):
            plus=[a.copy() for a in arrays];minus=[a.copy() for a in arrays]
            plus[position][index]+=1e-6;minus[position][index]-=1e-6
            expected[index]=(objective(plus)-objective(minus))/2e-6
        np.testing.assert_allclose(value.grad.numpy(),expected,rtol=5e-6,atol=5e-6)


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('unbatched',[False,True])
def test_module_output_size_parameter_identity_reset_checkpoint_and_aliases(dim,unbatched):
    with as_torch():
        from torch.nn.modules.conv import _ConvTransposeNd,_ConvNd
        cls=getattr(torch.nn,f'ConvTranspose{dim}d')
        torch.manual_seed(4814);m=cls(4,6,2,stride=2,padding=1,groups=2,dtype=torch.float64)
        assert isinstance(m,_ConvTransposeNd) and isinstance(m,_ConvNd) and m.transposed
        assert m.weight.shape==(4,3,*([2]*dim)) and list(m.state_dict())==['weight','bias']
        bound=(3*2**dim)**-.5
        assert np.max(np.abs(m.weight.numpy()))<=bound and np.max(np.abs(m.bias.numpy()))<=bound
        x=torch.ones((4,*([3]*dim)) if unbatched else (2,4,*([3]*dim)),dtype=torch.float64)
        y=m(x,output_size=[5]*dim);assert y.shape[-dim:]==(5,)*dim
        full=[999]*(1 if unbatched else 2)+[5]*dim
        np.testing.assert_array_equal(m(x,output_size=full).numpy(),y.numpy())
        with pytest.raises(ValueError):m(x,output_size=[6]*dim)
        with pytest.raises(ValueError):m(x,output_size=[1]*(dim+3))
        clone=cls(4,6,2,stride=2,padding=1,groups=2,dtype=torch.float64);clone.load_state_dict(copy.deepcopy(m.state_dict()))
        np.testing.assert_array_equal(clone(x,[5]*dim).numpy(),y.numpy())
        identity=m.weight;m.reset_parameters();assert m.weight is identity
        m.float();assert m.weight.dtype==torch.float32
        m.eval();assert not m.training
        fn=getattr(torch,f'conv_transpose{dim}d')
        np.testing.assert_array_equal(fn(x,clone.weight,clone.bias,stride=2,padding=1,groups=2).numpy(),clone(x).numpy())


@pytest.mark.parametrize('dim',[1,2,3])
def test_empty_batch_and_unbatched_gradients(dim):
    x=torch.empty(0,2,*([3]*dim),dtype=torch.float64,requires_grad=True);w=torch.ones(2,3,*([2]*dim),dtype=torch.float64,requires_grad=True);b=torch.ones(3,dtype=torch.float64,requires_grad=True)
    fn=getattr(torch.nn.functional,f'conv_transpose{dim}d')
    p,t=torch.func.jvp(fn,(x,w,b),(torch.ones_like(x),torch.ones_like(w),torch.ones_like(b)))
    assert p.shape==t.shape==(0,3,*([4]*dim))
    expected=tuple(math.prod(max(1,n) for n in p.shape[i+1:]) for i in range(p.ndim))
    assert p.stride()==t.stride()==expected
    (p+t).sum().backward()
    assert x.grad.shape==x.shape;np.testing.assert_array_equal(w.grad.numpy(),np.zeros(w.shape));np.testing.assert_array_equal(b.grad.numpy(),np.zeros(3))


@pytest.mark.parametrize('mask',range(8))
def test_each_active_gradient_subset_and_retained_graph(mask):
    values=[torch.tensor(np.sin(np.arange(6)).reshape(1,2,3),requires_grad=bool(mask&1)),torch.ones(2,3,2,dtype=torch.float64,requires_grad=bool(mask&2)),torch.zeros(3,dtype=torch.float64,requires_grad=bool(mask&4))]
    y=torch.nn.functional.conv_transpose1d(*values,stride=2,padding=1,output_padding=1)
    assert y.requires_grad==bool(mask)
    if not mask:return
    active=[v for v in values if v.requires_grad];seed=torch.ones_like(y)
    expected=torch.autograd.grad(y,active,seed,retain_graph=True);y.backward(seed)
    for v,g in zip(active,expected):np.testing.assert_array_equal(v.grad.numpy(),g.numpy())


@pytest.mark.parametrize('changed',['input','weight','bias','output'])
def test_native_saved_versions(changed):
    x=torch.ones(1,2,3,requires_grad=True);w=torch.ones(2,3,2,requires_grad=True);b=torch.ones(3,requires_grad=True)
    y=torch.nn.functional.conv_transpose1d(x,w,b)
    target={'input':x,'weight':w,'bias':b,'output':y}[changed];target.detach().add_(1)
    if changed in ['input','weight']:
        with pytest.raises(RuntimeError):y.sum().backward()
    else:y.sum().backward();assert b.grad.tolist()==[4.,4.,4.]


def test_bias_only_tape_releases_input_and_weight_objects():
    x=torch.ones(1,2,3);w=torch.ones(2,3,2);b=torch.ones(3,requires_grad=True)
    refs=[weakref.ref(x),weakref.ref(w)];y=torch.nn.functional.conv_transpose1d(x,w,b);del x,w;gc.collect()
    assert all(ref() is None for ref in refs)
    y.sum().backward();assert b.grad.tolist()==[4.,4.,4.]


@pytest.mark.parametrize('dim',[1,2,3])
def test_huge_dilation_padding_does_not_allocate_padded_activation(monkeypatch,dim):
    from torchwasm import _convolution
    def reject(*args,**kwargs):raise AssertionError('unbounded padded activation allocation')
    monkeypatch.setattr(_convolution,'_pad',reject)
    x=torch.ones(1,2,*([3]*dim),dtype=torch.float64,requires_grad=True);w=torch.ones(2,3,*([3]*dim),dtype=torch.float64,requires_grad=True)
    y=getattr(torch.nn.functional,f'conv_transpose{dim}d')(x,w,dilation=10000,padding=10000)
    y.sum().backward();assert y.shape==(1,3,*([3]*dim))
    np.testing.assert_array_equal(y.numpy(),np.full(y.shape,2.));np.testing.assert_array_equal(x.grad.numpy(),np.full(x.shape,3.))
    expected=np.zeros(w.shape);expected[(slice(None),slice(None),*([1]*dim))]=3**dim
    np.testing.assert_array_equal(w.grad.numpy(),expected)


@pytest.mark.parametrize('which',['input','weight'])
def test_huge_padding_preserves_implicit_zero_times_inf_backward(which):
    x=torch.ones(1,1,3,dtype=torch.float64,requires_grad=True);w=torch.ones(1,1,3,dtype=torch.float64,requires_grad=True)
    if which=='input':x.detach()[0,0,0]=float('inf')
    else:w.detach()[0,0,0]=float('inf')
    y=torch.nn.functional.conv_transpose1d(x,w,dilation=10000000,padding=10000000);y.sum().backward()
    if which=='input':
        np.testing.assert_array_equal(y.numpy(),[[[np.inf,1.,1.]]]);assert np.isnan(w.grad.numpy()[0,0,[0,2]]).all();assert np.isinf(w.grad.numpy()[0,0,1])
    else:np.testing.assert_array_equal(y.numpy(),[[[1.,1.,1.]]]);assert np.isnan(x.grad.numpy()).all()


@pytest.mark.parametrize('channels',[1,4])
@pytest.mark.parametrize('which',['input','weight'])
@pytest.mark.parametrize('unbatched',[False,True])
def test_channels_last_2d_output_layout_and_leaf_gradient_layout(channels,which,unbatched):
    raw=np.ones((2,3,3,channels),dtype=np.float64).transpose(0,3,1,2) if which=='input' else np.ones((2,channels,3,3),dtype=np.float64)
    if unbatched:raw=raw[0]
    weights=np.ones((channels,2,2,3),dtype=np.float64).transpose(0,3,1,2) if which=='weight' else np.ones((channels,3,2,2),dtype=np.float64)
    x=torch.from_numpy(raw).requires_grad_();w=torch.from_numpy(weights).requires_grad_()
    y=torch.nn.functional.conv_transpose2d(x,w,stride=2)
    expected=(1,18,3) if unbatched and which=='weight' else (36,6,1) if unbatched else (108,1,18,3)
    assert y.stride()==expected
    y.sum().backward()
    for size,a,b in zip(x.shape,x.grad.stride(),x.stride()):
        if size>1:assert a==b


def test_invalid_options_and_explicit_dtype_boundary():
    x=torch.ones(1,2,3);w=torch.ones(2,3,2)
    for kwargs in [dict(output_padding=1),dict(output_padding=-1),dict(stride=0),dict(dilation=0),dict(padding=-1),dict(groups=3),dict(padding=10)]:
        with pytest.raises((RuntimeError,ValueError)):torch.nn.functional.conv_transpose1d(x,w,**kwargs)
    with pytest.raises(RuntimeError):torch.nn.functional.conv_transpose1d(torch.ones(1,0,3),torch.ones(0,3,2))
    with pytest.raises(UnsupportedOperation):torch.nn.functional.conv_transpose1d(x.half(),w.half())
    with pytest.raises(ValueError):torch.nn.ConvTranspose1d(2,3,2,padding_mode='reflect')
    m=torch.nn.ConvTranspose1d(2,3,2,padding='same');assert m.padding==tuple('same')
    with pytest.raises(TypeError):m(x)


def test_decoder_adam_checkpoint_resume():
    def make():return torch.nn.Sequential(torch.nn.ConvTranspose2d(2,4,3,stride=2,padding=1,output_padding=1,dtype=torch.float64),torch.nn.InstanceNorm2d(4,affine=True,track_running_stats=True,dtype=torch.float64),torch.nn.GELU(),torch.nn.Conv2d(4,2,1,dtype=torch.float64))
    torch.manual_seed(4815);model=make();optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    x=torch.tensor(np.sin(np.arange(36)).reshape(1,2,3,6));target=torch.tensor(np.cos(np.arange(144)).reshape(1,2,6,12))
    def step(m,o):o.zero_grad();loss=(m(x)-target).square().mean();loss.backward();o.step();return loss.item()
    step(model,optimizer);stream=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},stream);stream.seek(0);state=torch.load(stream)
    restored=make();restored.load_state_dict(state['model']);other=torch.optim.AdamW(restored.parameters());other.load_state_dict(state['optimizer'])
    np.testing.assert_allclose(step(model,optimizer),step(restored,other),atol=1e-12)
    for a,b in zip(model.parameters(),restored.parameters()):np.testing.assert_allclose(a.numpy(),b.numpy(),atol=1e-12)
