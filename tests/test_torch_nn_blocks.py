"""Native-reference network layers, functional derivatives and training workflows."""
import copy

import numpy as np
import pytest
import torch as native

from paperleet._vendor.torch import UnsupportedOperation, torch as lite


def tensor(api,data,dtype='float64',grad=True):
    return api.tensor(np.asarray(data),dtype=getattr(api,dtype),requires_grad=grad)


def compare(a,b,dtype='float64',**kwargs):
    np.testing.assert_allclose(a.detach().numpy(),b.detach().numpy(),
                              rtol=3e-5 if dtype=='float32' else 2e-11,
                              atol=3e-6 if dtype=='float32' else 2e-12,**kwargs)


@pytest.mark.parametrize('kind',['LayerNorm','RMSNorm'])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('shape,normalized', [((5,),(5,)),((2,3,5),(5,)),((2,3,4),(3,4)),
                                           ((0,5),(5,)),((2,0),(0,)),((2,1),(1,))])
@pytest.mark.parametrize('affine',[False,True])
def test_norm_modules_values_vjps_parameters_and_reset_match_native(kind,dtype,shape,normalized,affine):
    rng=np.random.default_rng(270);data=rng.normal(size=shape);upstream=rng.normal(size=shape)
    results=[]
    for api in (lite,native):
        model=getattr(api.nn,kind)(normalized,eps=1e-4,elementwise_affine=affine,dtype=getattr(api,dtype))
        if affine:
            with api.no_grad():
                model.weight[:]=tensor(api,np.linspace(.5,1.5,int(np.prod(normalized))).reshape(normalized),dtype,False)
                if kind=='LayerNorm':model.bias[:]=.1
        x=tensor(api,data,dtype);y=model(x);(y*tensor(api,upstream,dtype,False)).sum().backward()
        assert y.dtype==getattr(api,dtype) and tuple(model.normalized_shape)==normalized
        parameters=list(model.parameters());identities=[id(p) for p in parameters]
        results.append((y,x.grad,[p.grad for p in parameters],list(model.state_dict()),repr(model)))
        model.reset_parameters();assert [id(p) for p in model.parameters()]==identities
        for name,p in model.named_parameters():np.testing.assert_array_equal(p.detach().numpy(),0 if name=='bias' else 1)
    compare(results[0][0],results[1][0],dtype);compare(results[0][1],results[1][1],dtype)
    for a,b in zip(results[0][2],results[1][2]):compare(a,b,dtype)
    assert results[0][3:]==results[1][3:]


@pytest.mark.parametrize('kind',['layer_norm','rms_norm'])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('strided',[False,True])
def test_norm_jvp_and_reverse_over_forward_match_native(kind,dtype,strided):
    rng=np.random.default_rng(107);shape=(2,3,4);raw=rng.normal(size=shape)
    data=raw.transpose(1,0,2) if strided else raw
    inputs=[data,rng.normal(size=4)]
    if kind=='layer_norm':inputs.append(rng.normal(size=4))
    tangents=[rng.normal(size=np.shape(v)) for v in inputs];results=[]
    for api in (lite,native):
        xs=tuple(tensor(api,v,dtype) for v in inputs);ts=tuple(tensor(api,v,dtype) for v in tangents)
        fn=lambda x,*params:getattr(api.nn.functional,kind)(x,(4,),*params,eps=1e-4)
        y,dy=api.func.jvp(fn,xs,ts);dy.square().sum().backward()
        results.append((y,dy,[x.grad for x in (*xs,*ts)]))
    compare(results[0][0],results[1][0],dtype);compare(results[0][1],results[1][1],dtype)
    for a,b in zip(results[0][2],results[1][2]):
        if a is None or b is None:assert a is b
        else:compare(a,b,dtype)


def test_layer_norm_sensitive_float32_rows_follow_float64_formula_without_large_offset_cancellation():
    data=np.array([[99999984.,99999992.,100000000.,100000008.],
                   [2.,3.,4.,7.],[1e20]*4],dtype=np.float32)
    x=tensor(lite,data,'float32');y=lite.nn.LayerNorm(4,elementwise_affine=False)(x)
    expected=(data.astype(np.float64)-data.astype(np.float64).mean(axis=1,keepdims=True))
    expected/=np.sqrt((expected**2).mean(axis=1,keepdims=True)+1e-5)
    np.testing.assert_allclose(y.numpy(),expected.astype(np.float32),rtol=2e-6,atol=2e-7)
    y.sum().backward();np.testing.assert_allclose(x.grad.numpy(),0.,atol=3e-7)


@pytest.mark.parametrize('approximate',['none','tanh'])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('shape', [(),(0,),(2,4)])
def test_gelu_module_values_vjp_jvp_and_curvature_match_native(approximate,dtype,shape):
    rng=np.random.default_rng(281);data=rng.normal(size=shape)*2;direction=rng.normal(size=shape)
    results=[]
    for api in (lite,native):
        x=tensor(api,data,dtype);dx=tensor(api,direction,dtype)
        module=api.nn.GELU(approximate)
        y,dy=api.func.jvp(module,(x,),(dx,));dy.sum().backward()
        results.append((y,dy,x.grad,dx.grad,repr(module)))
    for a,b in zip(results[0][:4],results[1][:4]):compare(a,b,dtype)
    assert results[0][4]==results[1][4]


@pytest.mark.parametrize('approximate',['none','tanh'])
def test_gelu_nonfinite_and_tail_values_follow_native(approximate):
    data=[-np.inf,np.inf,np.nan,-10.,-7.,0.,7.,10.];results=[]
    for api in (lite,native):
        x=tensor(api,data);y=api.nn.GELU(approximate)(x);y.sum().backward();results.append((y,x.grad))
    for a,b in zip(*results):compare(a,b,equal_nan=True)


@pytest.mark.parametrize('p',[0.,1.])
@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_dropout_boundary_modes_identity_values_and_gradients_match_native(p,training,dtype):
    results=[]
    for api in (lite,native):
        x=tensor(api,[np.inf,-np.inf,np.nan,-0.,1.,-2.],dtype)
        layer=api.nn.Dropout(p);layer.train(training);y=layer(x);y.backward(api.ones_like(y))
        results.append((y,x.grad,y is x,repr(layer)))
    compare(results[0][0],results[1][0],dtype,equal_nan=True);compare(results[0][1],results[1][1],dtype)
    assert results[0][2:]==results[1][2:]


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_dropout_random_masks_share_vjp_jvp_scaling_and_do_not_draw_when_disabled(dtype):
    for api in (lite,native):
        x=api.ones(10000,dtype=getattr(api,dtype),requires_grad=True)
        api.manual_seed(374);before=api.rand(3)
        api.manual_seed(374)
        assert api.nn.Dropout(.4).eval()(x) is x
        assert api.nn.Dropout(0)(x) is x
        api.nn.Dropout(1)(x)
        np.testing.assert_array_equal(api.rand(3).numpy(),before.numpy())
        api.manual_seed(441);layer=api.nn.Dropout(.4)
        y,dy=api.func.jvp(layer,(x,),(api.ones_like(x),))
        np.testing.assert_array_equal(y.detach().numpy(),dy.detach().numpy())
        assert .37<float((y.detach().numpy()==0).mean())<.43
        y.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),y.detach().numpy())
        np.testing.assert_allclose(y.detach().numpy()[y.detach().numpy()!=0],1/.6,rtol=2e-7)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('padding',[None,0,-1])
@pytest.mark.parametrize('frequency',[False,True])
def test_embedding_initialization_padding_vjp_and_reset_match_native(dtype,padding,frequency):
    results=[]
    for api in (lite,native):
        api.manual_seed(338);model=api.nn.Embedding(7,4,padding_idx=padding,scale_grad_by_freq=frequency,dtype=getattr(api,dtype))
        x=api.tensor([[0,1,1],[6,2,1]],dtype=api.int64);y=model(x);y.square().sum().backward()
        results.append((model.weight.detach().clone(),y,model.weight.grad,repr(model)))
        weight,grad=model.weight,model.weight.grad;model.reset_parameters()
        assert model.weight is weight and model.weight.grad is grad
        if padding is not None:np.testing.assert_array_equal(model.weight.detach().numpy()[padding],0.)
    for a,b in zip(results[0][:3],results[1][:3]):compare(a,b,dtype)
    assert results[0][3]==results[1][3]


@pytest.mark.parametrize('norm',[0.,1.,2.,-2.,np.inf,-np.inf])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_embedding_max_norm_values_gradients_and_counter_updates_match_native(norm,dtype):
    data=np.array([[.2,.4,.1],[2.,3.,4.],[1.,.4,.2],[2.,-3.,4.]])
    results=[]
    for api in (lite,native):
        w=tensor(api,data,dtype);old=w._version
        y=api.nn.functional.embedding(api.tensor([1,1,3],dtype=api.int64),w,max_norm=.7,norm_type=norm,scale_grad_by_freq=True)
        y.square().sum().backward();assert w._version==old+1
        results.append((w,y,w.grad))
    for a,b in zip(*results):compare(a,b,dtype)


def test_embedding_pretrained_sharing_freezing_and_forward_derivative_match_native():
    results=[]
    for api in (lite,native):
        source=tensor(api,np.arange(12).reshape(4,3),grad=False)
        frozen=api.nn.Embedding.from_pretrained(source,padding_idx=0)
        assert not frozen.weight.requires_grad and frozen.weight.data_ptr()==source.data_ptr()
        assert frozen(api.tensor([0],dtype=api.int64)).numpy().tolist()==[[0.,1.,2.]]
        with api.no_grad():source[1]=7
        assert frozen.weight.detach().numpy()[1].tolist()==[7.,7.,7.]
        w=tensor(api,np.arange(12).reshape(4,3)/7)
        direction=tensor(api,np.ones((4,3))*2)
        fn=lambda v:api.nn.functional.embedding(api.tensor([0,1,1],dtype=api.int64),v,padding_idx=0,scale_grad_by_freq=True)
        y,dy=api.func.jvp(fn,(w,),(direction,));dy.sum().backward()
        results.append((y,dy,direction.grad))
    for a,b in zip(*results):compare(a,b)


@pytest.mark.parametrize('kind',['MSELoss','SmoothL1Loss','CrossEntropyLoss'])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
def test_loss_modules_and_shape_activation_modules_match_native(kind,reduction):
    rng=np.random.default_rng(819);data=rng.normal(size=(3,4));target=rng.normal(size=(3,4));results=[]
    for api in (lite,native):
        x=tensor(api,data)
        y=api.nn.Identity('ignored',anything=True)(x);assert y is x
        y=api.nn.Sigmoid()(api.nn.Tanh()(api.nn.Flatten(0)(x))).reshape(3,4)
        labels=api.tensor([0,2,-100],dtype=api.int64) if kind=='CrossEntropyLoss' else tensor(api,target)
        loss=getattr(api.nn,kind)(reduction=reduction)(y,labels);loss.sum().backward()
        results.append((loss,x.grad))
    for a,b in zip(*results):compare(a,b)


def test_new_layer_boundaries_fail_explicitly_instead_of_silently_changing_execution():
    for constructor,kwargs in [(lite.nn.LayerNorm,{'normalized_shape':3,'device':'cuda'}),
                                (lite.nn.Embedding,{'num_embeddings':3,'embedding_dim':2,'sparse':True})]:
        with pytest.raises(UnsupportedOperation):constructor(**kwargs)
    with pytest.raises(UnsupportedOperation):lite.nn.CrossEntropyLoss(size_average=False)
    with pytest.raises(UnsupportedOperation):lite.nn.Dropout(.2,inplace=True)(lite.ones(3,requires_grad=True))
    with pytest.raises(RuntimeError):lite.nn.LayerNorm(3)(lite.ones(2,4))
    with pytest.raises(RuntimeError):lite.nn.functional.layer_norm(lite.ones(2,3),(3,),weight=lite.ones(1))


def encoder(api,dtype):
    class Encoder(api.nn.Module):
        def __init__(self):
            super().__init__();N=api.nn
            options={'dtype':getattr(api,dtype)}
            self.embedding=N.Embedding(11,8,padding_idx=0,**options)
            self.norm1=N.LayerNorm(8,**options);self.qkv=N.Linear(8,24,**options)
            self.projection=N.Linear(8,8,**options);self.norm2=N.RMSNorm(8,eps=1e-5,**options)
            self.ff=N.Sequential(N.Linear(8,16,**options),N.GELU('tanh'),N.Dropout(0.),N.Linear(16,8,**options))
            self.head=N.Linear(8,5,**options)
        def forward(self,ids):
            x=self.embedding(ids);batch,length=ids.shape
            q,k,v=self.qkv(self.norm1(x)).reshape(batch,length,3,2,4).permute(2,0,3,1,4)
            attention=api.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
            x=x+self.projection(attention.transpose(1,2).reshape(batch,length,8))
            x=x+self.ff(self.norm2(x))
            return self.head(x.mean(1))
    model=Encoder();rng=np.random.default_rng(817)
    with api.no_grad():
        for name,p in model.named_parameters():
            data=rng.normal(size=tuple(p.shape))*.1
            if name.endswith('norm1.weight') or name.endswith('norm2.weight'):data+=1
            if name=='embedding.weight':data[0]=0
            p[:]=api.tensor(data,dtype=getattr(api,dtype))
    return model


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_attention_encoder_training_full_gradients_and_checkpoint_continuation_match_native(dtype):
    histories=[]
    for api in (lite,native):
        model=encoder(api,dtype);o=api.optim.SGD(model.parameters(),lr=.02,momentum=.8)
        scheduler=api.optim.lr_scheduler.LinearLR(o,start_factor=.2,total_iters=3)
        ids=api.tensor([[0,1,2],[2,3,4]],dtype=api.int64);labels=api.tensor([1,4],dtype=api.int64)
        rows=[]
        for step in range(6):
            o.zero_grad();logits=model(ids);loss=api.nn.CrossEntropyLoss()(logits,labels);loss.backward()
            rows.append((loss.detach().numpy(),[p.grad.detach().numpy().copy() for p in model.parameters()]))
            o.step();scheduler.step()
            if step==2:
                checkpoint=copy.deepcopy({'model':model.state_dict(),'optimizer':o.state_dict(),'scheduler':scheduler.state_dict()})
                restored=encoder(api,dtype);ro=api.optim.SGD(restored.parameters(),lr=.02,momentum=.8)
                rs=api.optim.lr_scheduler.LinearLR(ro,start_factor=.2,total_iters=3)
                restored.load_state_dict(checkpoint['model']);rs.load_state_dict(checkpoint['scheduler']);ro.load_state_dict(checkpoint['optimizer'])
            if step>2:
                ro.zero_grad();api.nn.CrossEntropyLoss()(restored(ids),labels).backward();ro.step();rs.step()
                for a,b in zip(model.parameters(),restored.parameters()):np.testing.assert_array_equal(a.detach().numpy(),b.detach().numpy())
        model.eval();assert all(not child.training for child in model.modules())
        with api.no_grad():assert not model(ids).requires_grad
        histories.append(rows)
    for first,second in zip(*histories):
        np.testing.assert_allclose(first[0],second[0],rtol=2e-5 if dtype=='float32' else 1e-11,atol=1e-7)
        for a,b in zip(first[1],second[1]):np.testing.assert_allclose(a,b,rtol=8e-5 if dtype=='float32' else 1e-10,atol=2e-6 if dtype=='float32' else 1e-12)


@pytest.mark.parametrize('kind',['layer_norm','rms_norm'])
@pytest.mark.parametrize('shape',[(2,0),(0,4)])
def test_empty_norm_jvp_and_reverse_over_forward_match_native(kind,shape):
    outcomes=[]
    for api in (lite,native):
        x=api.ones(shape,dtype=api.float64,requires_grad=True);dx=api.ones_like(x,requires_grad=True)
        with np.errstate(divide='ignore',invalid='ignore'):
            y,dy=api.func.jvp(lambda v:getattr(api.nn.functional,kind)(v,(shape[-1],)),(x,),(dx,));dy.sum().backward()
        outcomes.append((y,dy,x.grad,dx.grad))
    for a,b in zip(*outcomes):compare(a,b)
