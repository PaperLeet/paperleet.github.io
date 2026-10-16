"""NumPy-only normalization, stochastic layers and complete encoder training."""
import copy
import gc
import math
import weakref

import numpy as np
import pytest

from stridetorch import UnsupportedOperation,as_torch,torch


@pytest.mark.parametrize('kind',['LayerNorm','RMSNorm'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_normalization_values_and_gradients_against_independent_numpy(kind,dtype):
    rng=np.random.default_rng(381);data=rng.normal(size=(2,3)).astype(dtype)
    weights=np.array([.5,1.,1.5],dtype=dtype);bias=np.array([.1,-.2,.3],dtype=dtype)
    module=getattr(torch.nn,kind)(3,eps=1e-4,dtype=dtype)
    with torch.no_grad():
        module.weight[:]=weights
        if kind=='LayerNorm':module.bias[:]=bias
    def oracle(values,w=weights,b=bias):
        source=values-values.mean(axis=-1,keepdims=True) if kind=='LayerNorm' else values
        result=source/np.sqrt((source*source).mean(axis=-1,keepdims=True)+1e-4)*w
        return result+b if kind=='LayerNorm' else result
    x=torch.tensor(data,requires_grad=True);y=module(x);y.square().sum().backward()
    np.testing.assert_allclose(y.numpy(),oracle(data),rtol=3e-6,atol=2e-7)
    if dtype==torch.float64:
        epsilon=1e-6;expected=np.empty_like(data)
        for index in np.ndindex(data.shape):
            plus,minus=data.copy(),data.copy();plus[index]+=epsilon;minus[index]-=epsilon
            expected[index]=((oracle(plus)**2).sum()-(oracle(minus)**2).sum())/(2*epsilon)
        np.testing.assert_allclose(x.grad.numpy(),expected,rtol=2e-5,atol=1e-8)


@pytest.mark.parametrize('approximate',['none','tanh'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('shape',[(),(0,),(2,3)])
def test_gelu_precision_empty_inputs_and_jvp_against_scalar_math(approximate,dtype,shape):
    data=np.linspace(-2,2,int(np.prod(shape)) if shape else 1,dtype=dtype).reshape(shape)
    def oracle(values):
        if approximate=='none':
            flat=[.5*float(x)*(1+math.erf(float(x)/math.sqrt(2))) for x in values.flat]
            return np.array(flat,dtype=dtype).reshape(values.shape)
        return (.5*values*(1+np.tanh(np.sqrt(2/np.pi)*(values+.044715*values**3)))).astype(dtype)
    x=torch.tensor(data,requires_grad=True)
    y,dy=torch.func.jvp(torch.nn.GELU(approximate),(x,),(torch.ones_like(x),))
    assert y.dtype==dtype and dy.dtype==dtype
    np.testing.assert_allclose(y.numpy(),oracle(data),rtol=3e-6,atol=2e-7)
    y.sum().backward();np.testing.assert_allclose(x.grad.numpy(),dy.numpy(),rtol=1e-6,atol=1e-8)
    if dtype==torch.float64:
        expected=(oracle(data+1e-6)-oracle(data-1e-6))/2e-6
        np.testing.assert_allclose(dy.numpy(),expected,rtol=3e-5,atol=1e-9)


def test_dropout_repeatability_scaling_and_eval_keep_imported_torch_code_unchanged():
    with as_torch():
        from torch.nn import Dropout
        import torch.nn.functional as F
        x=torch.ones(2000,requires_grad=True)
        torch.manual_seed(712);y=Dropout(.25)(x)
        torch.manual_seed(712);z=F.dropout(x,p=.25)
        np.testing.assert_array_equal(y.numpy(),z.numpy())
        y.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),y.numpy())
        assert .2<np.mean(y.numpy()==0)<.3
        assert Dropout(.25).eval()(x) is x
        frozen=x.detach().clone();version=frozen._version
        assert Dropout(.25,inplace=True)(frozen) is frozen and frozen._version==version+1


def test_embedding_pretrained_padding_frequency_renormalization_and_mutation_guards():
    source=torch.tensor([[0.,0.],[3.,4.],[1.,2.]],dtype=torch.float64)
    model=torch.nn.Embedding.from_pretrained(source,freeze=False,padding_idx=0,max_norm=1.,scale_grad_by_freq=True)
    assert model.weight.data_ptr()==source.data_ptr()
    old=model.weight.square().sum();result=model(torch.tensor([0,1,1],dtype=torch.int64));result.sum().backward()
    np.testing.assert_allclose(model.weight.detach().numpy()[1],[3/5.0000001,4/5.0000001])
    np.testing.assert_array_equal(model.weight.grad.numpy(),[[0.,0.],[1.,1.],[0.,0.]])
    with pytest.raises(UnsupportedOperation,match='modified in-place'):old.backward()
    model.zero_grad();model.reset_parameters();np.testing.assert_array_equal(model.weight.detach().numpy()[0],0.)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_stochastic_attention_encoder_train_checkpoint_rebuild_load_and_continue(dtype):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();N=torch.nn;opts={'dtype':dtype}
            self.embed=N.Embedding(7,4,padding_idx=0,**opts);self.norm=N.LayerNorm(4,**opts)
            self.qkv=N.Linear(4,12,**opts);self.rms=N.RMSNorm(4,eps=1e-5,**opts)
            self.ff=N.Sequential(N.Linear(4,8,**opts),N.GELU('tanh'),N.Dropout(.25),N.Linear(8,4,**opts))
            self.head=N.Linear(4,3,**opts)
        def forward(self,ids):
            x=self.embed(ids);batch,length=ids.shape
            q,k,v=self.qkv(self.norm(x)).reshape(batch,length,3,1,4).permute(2,0,3,1,4)
            x=x+torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).reshape(batch,length,4)
            return self.head((x+self.ff(self.rms(x))).mean(1))
    def create():
        m=Encoder();o=torch.optim.AdamW(m.parameters(),lr=.005,amsgrad=True)
        return m,o,torch.optim.lr_scheduler.StepLR(o,2,.8)
    torch.manual_seed(182);model,o,s=create()
    ids=torch.tensor([[0,1,2],[2,3,4]],dtype=torch.int64);labels=torch.tensor([0,2],dtype=torch.int64)
    for step in range(3):
        o.zero_grad();torch.nn.CrossEntropyLoss()(model(ids),labels).backward();o.step();s.step()
    checkpoint=copy.deepcopy({'model':model.state_dict(),'optimizer':o.state_dict(),'scheduler':s.state_dict()})
    with torch.random.fork_rng():restored,ro,rs=create()
    restored.load_state_dict(checkpoint['model']);rs.load_state_dict(checkpoint['scheduler']);ro.load_state_dict(checkpoint['optimizer'])
    for step in range(4):
        torch.manual_seed(711+step)
        for m,opt,schedule in ((model,o,s),(restored,ro,rs)):
            with torch.random.fork_rng():
                opt.zero_grad();torch.nn.CrossEntropyLoss()(m(ids),labels).backward();opt.step();schedule.step()
        for a,b in zip(model.parameters(),restored.parameters()):np.testing.assert_array_equal(a.detach().numpy(),b.detach().numpy())
    model.eval();assert all(not child.training for child in model.modules())
    with torch.no_grad():
        first=model(ids);second=model(ids);np.testing.assert_array_equal(first.numpy(),second.numpy());assert not first.requires_grad


def test_sensitive_layer_norm_uses_accurate_centering_and_preserves_float32():
    data=np.array([[99999984.,99999992.,100000000.,100000008.]],dtype=np.float32)
    expected=data.astype(np.float64)-data.astype(np.float64).mean(-1,keepdims=True)
    expected/=np.sqrt((expected*expected).mean(-1,keepdims=True)+1e-5)
    result=torch.nn.LayerNorm(4)(torch.tensor(data))
    assert result.dtype==torch.float32
    np.testing.assert_allclose(result.numpy(),expected,rtol=2e-6,atol=2e-7)


@pytest.mark.parametrize('shape',[(0,4),(2,0)])
def test_empty_norm_jvp_has_no_spurious_nonfinite_warnings(shape):
    for name in ('layer_norm','rms_norm'):
        x=torch.ones(shape,dtype=torch.float64,requires_grad=True)
        dx=torch.ones_like(x,requires_grad=True)
        y,dy=torch.func.jvp(lambda v:getattr(torch.nn.functional,name)(v,(shape[-1],)),(x,),(dx,))
        dy.sum().backward();assert y.shape==dy.shape==x.grad.shape==dx.grad.shape==shape


def test_bias_only_norm_backward_does_not_retain_frozen_inputs():
    x=torch.ones(4,8);weight=torch.ones(8);bias=torch.zeros(8,requires_grad=True)
    refs=[weakref.ref(x),weakref.ref(weight)]
    y=torch.nn.functional.layer_norm(x,(8,),weight,bias)
    del x,weight;gc.collect();assert all(ref() is None for ref in refs)
    y.backward(torch.ones_like(y));np.testing.assert_array_equal(bias.grad.numpy(),[4.]*8)


def test_standard_nn_submodule_imports_share_the_public_classes():
    with as_torch() as namespace:
        from torch.nn.parameter import Parameter
        from torch.nn.modules import LayerNorm
        from torch.nn.modules.flatten import Flatten
        from torch.nn.modules.module import Module
        from torch.nn.modules.normalization import RMSNorm
        from torch.nn.modules.sparse import Embedding
        from torch.nn.modules.dropout import Dropout
        from torch.nn.modules.activation import GELU
        from torch.nn.modules.loss import CrossEntropyLoss
        for name,value in locals().copy().items():
            if name in ('Parameter','LayerNorm','Flatten','Module','RMSNorm','Embedding','Dropout','GELU','CrossEntropyLoss'):
                assert value is getattr(namespace.nn,name)
