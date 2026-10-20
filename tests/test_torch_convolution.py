"""Native-reference dense grouped convolutions, padding adjoints and dual tapes."""
import copy
import numpy as np
import pytest
import torch
from torchwasm import torch as lite


@pytest.fixture(params=['direct','matrix','auto'],autouse=True)
def convolution_algorithm(request,monkeypatch):
    from torchwasm import _convolution
    monkeypatch.setattr(_convolution,'_ALGORITHM',request.param)


def close(a,b,dtype=np.float64):
    aa=a.detach().numpy() if hasattr(a,'detach') else np.asarray(a)
    bb=b.detach().numpy() if hasattr(b,'detach') else np.asarray(b)
    np.testing.assert_allclose(aa,bb,rtol=4e-5 if dtype==np.float32 else 3e-11,atol=2e-5 if dtype==np.float32 else 3e-12,equal_nan=True)


def layout(array,kind):
    if kind=='fortran':return np.asfortranarray(array)
    if kind=='stride':return np.repeat(array,2,axis=-1)[...,::2]
    return array


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('case',['pointwise','valid','same','strided_dilated','grouped','depthwise','unbatched','empty_batch','empty_channels'])
@pytest.mark.parametrize('kind',['contiguous','fortran','stride'])
@pytest.mark.parametrize('bias',[False,True])
def test_values_all_pullbacks_layouts_and_retained_backward(dim,dtype,case,kind,bias):
    rng=np.random.default_rng(1902);spatial=tuple(5+i%2 for i in range(dim));groups=2 if case=='grouped' else 4 if case=='depthwise' else 1
    channels=0 if case=='empty_channels' else 4;outputs=8 if case=='depthwise' else 6;kernel=(1 if case=='pointwise' else 3,)*dim
    shape=(0 if case=='empty_batch' else 2,channels,*spatial)
    if case=='unbatched':shape=shape[1:]
    kw=dict(groups=groups,padding='same' if case=='same' else 'valid' if case=='valid' else 1 if case=='strided_dilated' else 0,stride=2 if case=='strided_dilated' else 1,dilation=2 if case=='strided_dilated' else 1)
    arrays=[layout(rng.normal(size=shape).astype(dtype),kind),layout(rng.normal(size=(outputs,channels//groups,*kernel)).astype(dtype),kind),rng.normal(size=outputs).astype(dtype)]
    records=[]
    for api in [lite,torch]:
        v=[api.from_numpy(a).requires_grad_() for a in arrays];out=getattr(api.nn.functional,f'conv{dim}d')(v[0],v[1],v[2] if bias else None,**kw)
        seed=rng.normal(size=out.shape).astype(dtype) if api is lite else seed
        for step in range(2):out.backward(api.tensor(seed),retain_graph=step==0)
        records.append([out,v[0].grad,v[1].grad,*([v[2].grad] if bias else [])])
    for a,b in zip(*records):close(a,b,dtype)
    assert records[0][0].dtype==np.dtype(dtype)


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
@pytest.mark.parametrize('padding',['same',1])
def test_module_padding_modes_state_dtype_reset_and_native_representation(dim,mode,padding):
    rng=np.random.default_rng(1903);shape=(2,4,*([5]*dim));records=[]
    arrays=[rng.normal(size=(6,2,*([2]*dim))),rng.normal(size=6),rng.normal(size=shape)]
    for api in [lite,torch]:
        m=getattr(api.nn,f'Conv{dim}d')(4,6,2,padding=padding,groups=2,padding_mode=mode,dtype=api.float64)
        with api.no_grad():m.weight.copy_(api.tensor(arrays[0]));m.bias.copy_(api.tensor(arrays[1]))
        x=api.tensor(arrays[2],requires_grad=True);out=m(x);out.square().sum().backward();records.append([out,x.grad,m.weight.grad,m.bias.grad,repr(m),m._reversed_padding_repeated_twice])
        state=copy.deepcopy(m.state_dict());m.float();assert str(m.weight.dtype).endswith('float32');m.double();m.load_state_dict(state)
        m.eval();close(m(api.tensor(arrays[2])),out);assert list(state)==['weight','bias'] and not list(m.buffers())
        m.reset_parameters();bound=(2*2**dim)**-.5
        assert np.max(np.abs(m.weight.detach().numpy()))<=bound and np.max(np.abs(m.bias.detach().numpy()))<=bound
    for a,b in zip(records[0][:4],records[1][:4]):close(a,b)
    assert records[0][4:]==records[1][4:]


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('active',['all','input','weight','bias'])
@pytest.mark.parametrize('mode',['zeros','reflect'])
def test_jvp_and_reverse_over_forward_including_closed_zero_dependencies(dim,active,mode):
    rng=np.random.default_rng(1904);shape=(1,2,*([4]*dim));arrays=[rng.normal(size=shape),rng.normal(size=(4,1,*([2]*dim))),rng.normal(size=4)];arrays += [rng.normal(size=a.shape) for a in arrays];records=[]
    for api in [lite,torch]:
        v=[api.tensor(a,requires_grad=True) for a in arrays]
        if mode=='zeros':
            fn=lambda x,w,b:getattr(api.nn.functional,f'conv{dim}d')(x,w,b,padding='same',groups=2)
        else:
            m=getattr(api.nn,f'Conv{dim}d')(2,4,2,padding='same',groups=2,padding_mode=mode,dtype=api.float64)
            # Functional parameters must remain differentiable leaves; replace
            # the module's registered Parameters with exact Tensor arguments.
            fn=lambda x,w,b:m._forward_test(x,w,b)
            if api is lite:
                from torchwasm._convolution import _conv
                m._forward_test=lambda x,w,b:_conv(x,w,b,1,'same',1,2,dim,mode)
            else:
                p=[0,1]*dim
                m._forward_test=lambda x,w,b:getattr(api.nn.functional,f'conv{dim}d')(api.nn.functional.pad(x,p,mode='reflect'),w,b,groups=2)
        if active=='all':p,t=api.func.jvp(fn,tuple(v[:3]),tuple(v[3:]))
        else:
            i=['input','weight','bias'].index(active)
            def selected(z):
                args=v[:3].copy();args[i]=z;return fn(*args)
            p,t=api.func.jvp(selected,(v[i],),(v[i+3],))
        t.square().sum().backward();records.append([p,t,*[a.grad for a in v]])
    for a,b in zip(*records):
        if a is None or b is None:assert a is b
        else:close(a,b)


@pytest.mark.parametrize('needs',[(True,True,True),(True,False,False),(False,True,False),(False,False,True)])
@pytest.mark.parametrize('changed',['input','weight','bias','output'])
def test_native_saved_versions_and_bias_output_mutation(needs,changed):
    outcomes=[]
    for api in [lite,torch]:
        x=api.ones(2,4,5,requires_grad=needs[0]);w=api.ones(6,2,3,requires_grad=needs[1]);b=api.ones(6,requires_grad=needs[2]);out=api.nn.functional.conv1d(x,w,b,groups=2);value=dict(input=x,weight=w,bias=b,output=out)[changed].detach()
        with api.no_grad():value.add_(1)
        try:out.sum().backward();outcomes.append('ok')
        except RuntimeError:outcomes.append('RuntimeError')
    assert outcomes[0]==outcomes[1]


def test_complete_cnn_groupnorm_classification_adamw_scheduler_checkpoint_resume():
    rng=np.random.default_rng(1909)
    def make(api):
        class Model(api.nn.Module):
            def __init__(self):
                super().__init__();self.conv=api.nn.Conv2d(2,4,3,padding=1,dtype=api.float64);self.norm=api.nn.GroupNorm(2,4,dtype=api.float64);self.activation=api.nn.SiLU();self.project=api.nn.Conv2d(4,3,1,dtype=api.float64)
            def forward(self,x):return self.project(self.activation(self.norm(self.conv(x)))).mean((2,3))
        return Model()
    models=[make(api) for api in [lite,torch]]
    for a,b in zip(models[0].parameters(),models[1].parameters()):
        raw=rng.normal(size=a.shape)*.2
        for api,p in [(lite,a),(torch,b)]:
            with api.no_grad():p.copy_(api.tensor(raw))
    opts=[api.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01) for api,m in zip([lite,torch],models)];schedulers=[api.optim.lr_scheduler.StepLR(o,step_size=2,gamma=.8) for api,o in zip([lite,torch],opts)]
    for step in range(7):
        raw=rng.normal(size=(4,2,5,5));labels=rng.integers(0,3,size=4,dtype=np.int64);labels[0]=-100;values=[]
        for api,m,o,s in zip([lite,torch],models,opts,schedulers):
            o.zero_grad();out=m(api.tensor(raw));loss=api.nn.functional.cross_entropy(out,api.tensor(labels),api.tensor([.3,1.2,.8],dtype=api.float64),label_smoothing=.15);loss.backward();o.step();s.step();values.append(loss)
        close(*values)
        for a,b in zip(models[0].parameters(),models[1].parameters()):close(a,b)
        if step==2:
            for api,i in [(lite,0),(torch,1)]:
                state=copy.deepcopy(models[i].state_dict());os=copy.deepcopy(opts[i].state_dict());ss=copy.deepcopy(schedulers[i].state_dict());models[i]=make(api);models[i].load_state_dict(state);opts[i]=api.optim.AdamW(models[i].parameters(),lr=99);schedulers[i]=api.optim.lr_scheduler.StepLR(opts[i],step_size=2,gamma=.8);opts[i].load_state_dict(os);schedulers[i].load_state_dict(ss)


@pytest.mark.parametrize('dim',[1,2,3])
def test_float32_jvp_all_inputs_and_reverse_tape(dim):
    rng=np.random.default_rng(1912);arrays=[rng.normal(size=(1,4,*([5]*dim))).astype(np.float32),(rng.normal(size=(6,2,*([3]*dim)))*.2).astype(np.float32),(rng.normal(size=6)*.1).astype(np.float32)];arrays += [(rng.normal(size=a.shape)*.1).astype(np.float32) for a in arrays];records=[]
    for api in [lite,torch]:
        v=[api.tensor(a,requires_grad=True) for a in arrays];p,t=api.func.jvp(lambda x,w,b:getattr(api.nn.functional,f'conv{dim}d')(x,w,b,padding=1,groups=2),tuple(v[:3]),tuple(v[3:]));(p+t).square().sum().backward();records.append([p,t,*[a.grad for a in v]])
    for a,b in zip(*records):close(a,b,np.float32)


def test_default_contiguous_outputs_for_fortran_inputs():
    raw=np.asfortranarray(np.arange(2*4*8*8,dtype=np.float64).reshape(2,4,8,8)/100);records=[]
    for api in [lite,torch]:
        value=api.nn.functional.conv2d(api.from_numpy(raw),api.ones(8,4,3,3,dtype=api.float64),padding=1);assert value.is_contiguous();records.append(value)
    close(*records);array=records[0].numpy();assert tuple(v//array.itemsize for v in array.strides)==records[1].stride()


def test_nonfinite_backward_seeds_use_explicit_products_and_native_nan_classes():
    records=[]
    for api in [lite,torch]:
        x=api.ones(2,4,8,8,dtype=api.float64,requires_grad=True);w=api.ones(8,4,3,3,dtype=api.float64,requires_grad=True);out=api.nn.functional.conv2d(x,w,padding=1);out.backward(api.full(out.shape,float('inf'),dtype=api.float64));records.append([x.grad,w.grad])
    for a,b in zip(*records):close(a,b)
