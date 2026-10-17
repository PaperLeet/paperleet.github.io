"""Dense Transformer equations, derivatives, cloning and training lifecycle."""
import copy
import math
import pickle

import numpy as np
import pytest

from stridetorch import as_torch, create_torch, torch


KINDS=['encoder_layer','decoder_layer','encoder','decoder','full']


def create(kind,layout,pre,activation='gelu',dtype=torch.float64,bias=True,ff=7,p=0.):
    opts=dict(dim_feedforward=ff,dropout=p,activation=activation,norm_first=pre,
              batch_first=layout=='batch',bias=bias,dtype=dtype)
    norm=lambda:torch.nn.LayerNorm(4,bias=bias,dtype=dtype)
    if kind=='encoder_layer':return torch.nn.TransformerEncoderLayer(4,2,**opts)
    if kind=='decoder_layer':return torch.nn.TransformerDecoderLayer(4,2,**opts)
    if kind=='encoder':return torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(4,2,**opts),2,norm())
    if kind=='decoder':return torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(4,2,**opts),2,norm())
    return torch.nn.Transformer(4,2,2,2,**opts)


def fixture(kind,layout,pre,activation='gelu',dtype=np.float64,bias=True,ff=7,p=0.):
    rng=np.random.default_rng(5301);model=create(kind,layout,pre,activation,dtype,bias,ff,p)
    with torch.no_grad():
        for name,w in model.named_parameters():
            w.copy_(torch.tensor(rng.normal(1 if 'norm' in name and name.endswith('weight') else 0,.2,size=w.shape).astype(dtype)))
    def shape(length):
        return (length,4) if layout=='unbatched' else (2,length,4) if layout=='batch' else (length,2,4)
    values=[rng.normal(0,.5,size=shape(3)).astype(dtype)]
    if kind not in ['encoder_layer','encoder']:values.append(rng.normal(0,.5,size=shape(4)).astype(dtype))
    inputs=[torch.tensor(np.asfortranarray(v),requires_grad=True) for v in values]
    state={n:p.detach().numpy().copy() for n,p in model.named_parameters()}
    return rng,model,inputs,values,state


def reference(values,state,kind,layout,pre,activation='gelu',mask=None,padding=None,p=0.,ff=7):
    def linear(x,prefix):return x@state[prefix+'weight'].T+state.get(prefix+'bias',0)
    def norm(x,prefix):
        z=x-x.mean(-1,keepdims=True)
        return z/np.sqrt((z*z).mean(-1,keepdims=True)+1e-5)*state[prefix+'weight']+state.get(prefix+'bias',0)
    def activate(x):
        if activation=='relu':return np.maximum(x,0)
        if activation=='sin':return np.sin(x)
        erf=np.array([math.erf(float(v)/math.sqrt(2)) for v in x.flat],dtype=x.dtype).reshape(x.shape)
        return x*.5*(1+erf)
    def attention(q,k,v,prefix,attn_mask=None,pad=None):
        embed=q.shape[-1];head=2;width=embed//head;batch=q.shape[0]
        weights=np.split(state[prefix+'in_proj_weight'],3)
        biases=np.split(state[prefix+'in_proj_bias'],3) if prefix+'in_proj_bias' in state else [0,0,0]
        projected=[(x@w.T+b).reshape(batch,x.shape[1],head,width).transpose(0,2,1,3) for x,w,b in zip([q,k,v],weights,biases)]
        q,k,v=projected;scores=q@k.swapaxes(-1,-2)/math.sqrt(width)
        if attn_mask is not None:
            mk=np.where(attn_mask,-np.inf,0.) if attn_mask.dtype.kind=='b' else attn_mask
            scores=scores+(mk.reshape(batch,head,*mk.shape[-2:]) if mk.ndim==3 else mk)
        if pad is not None:
            pd=np.where(pad,-np.inf,0.) if pad.dtype.kind=='b' else pad
            scores=scores+pd.reshape(batch,1,1,-1)
        with np.errstate(all='ignore'):
            maximum=scores.max(-1,keepdims=True);ex=np.exp(scores-np.where(np.isneginf(maximum),0,maximum))
            total=ex.sum(-1,keepdims=True);prob=np.divide(ex,total,out=np.zeros_like(ex),where=total!=0)
        y=(prob@v).transpose(0,2,1,3).reshape(batch,q.shape[-2],embed)
        return linear(y,prefix+'out_proj.')
    def feed(x,prefix):return linear(activate(linear(x,prefix+'linear1.')),prefix+'linear2.')
    def encoder(x,prefix):
        if pre:
            z=norm(x,prefix+'norm1.');x=x+(0 if p==1 else attention(z,z,z,prefix+'self_attn.',mask,padding))
            return x+(0 if p==1 else feed(norm(x,prefix+'norm2.'),prefix))
        x=norm(x+(0 if p==1 else attention(x,x,x,prefix+'self_attn.',mask,padding)),prefix+'norm1.')
        return norm(x+(0 if p==1 else feed(x,prefix)),prefix+'norm2.')
    def decoder(x,memory,prefix):
        if pre:
            z=norm(x,prefix+'norm1.');x=x+(0 if p==1 else attention(z,z,z,prefix+'self_attn.',mask,padding))
            x=x+(0 if p==1 else attention(norm(x,prefix+'norm2.'),memory,memory,prefix+'multihead_attn.'))
            return x+(0 if p==1 else feed(norm(x,prefix+'norm3.'),prefix))
        x=norm(x+(0 if p==1 else attention(x,x,x,prefix+'self_attn.',mask,padding)),prefix+'norm1.')
        x=norm(x+(0 if p==1 else attention(x,memory,memory,prefix+'multihead_attn.')),prefix+'norm2.')
        return norm(x+(0 if p==1 else feed(x,prefix)),prefix+'norm3.')
    def stack(x,memory,prefix,decode):
        for i in range(2):x=decoder(x,memory,prefix+f'layers.{i}.') if decode else encoder(x,prefix+f'layers.{i}.')
        return norm(x,prefix+'norm.')
    def batch(x):return x[None] if layout=='unbatched' else x.swapaxes(0,1) if layout=='sequence' else x
    x=batch(values[0]);memory=None if len(values)==1 else batch(values[1])
    if kind=='encoder_layer':y=encoder(x,'')
    elif kind=='decoder_layer':y=decoder(x,memory,'')
    elif kind=='encoder':y=stack(x,None,'',False)
    elif kind=='decoder':y=stack(x,memory,'',True)
    else:
        # The full module receives source=values[1], target=values[0]. Only
        # the target mask/padding is used by this reference's public call.
        target_mask,target_padding=mask,padding;mask=padding=None
        encoded=stack(memory,None,'encoder.',False)
        mask,padding=target_mask,target_padding;y=stack(x,encoded,'decoder.',True)
    return y[0] if layout=='unbatched' else y.swapaxes(0,1) if layout=='sequence' else y


def call_model(model,kind,inputs,mask=None,padding=None):
    if kind=='encoder_layer':return model(inputs[0],src_mask=mask,src_key_padding_mask=padding)
    if kind=='encoder':return model(inputs[0],mask=mask,src_key_padding_mask=padding)
    if kind=='full':return model(inputs[1],inputs[0],tgt_mask=mask,tgt_key_padding_mask=padding)
    return model(*inputs,tgt_mask=mask,tgt_key_padding_mask=padding)


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('layout',['sequence','batch','unbatched'])
@pytest.mark.parametrize('pre',[False,True])
@pytest.mark.parametrize('activation',['relu','gelu'])
@pytest.mark.parametrize('masked',[False,True])
def test_independent_numpy_values_vjp_and_input_jvp(kind,layout,pre,activation,masked):
    rng,m,inputs,values,state=fixture(kind,layout,pre,activation)
    mask=padding=None
    if masked:
        mask=np.array([[False,True,False],[False,False,True],[False,True,True]])
        padding=np.array([[False,False,True],[False,True,False]])
        if layout=='unbatched':padding=padding[0]
    tm=None if mask is None else torch.tensor(mask);tp=None if padding is None else torch.tensor(padding)
    opts=dict(kind=kind,layout=layout,pre=pre,activation=activation,mask=mask,padding=padding)
    want=reference(values,state,**opts);output=call_model(m,kind,inputs,tm,tp)
    np.testing.assert_allclose(output.numpy(),want,atol=3e-12,rtol=4e-11)
    seed=rng.normal(size=want.shape);active=inputs+list(m.parameters());directions=[rng.normal(size=x.shape) for x in active]
    gradients=torch.autograd.grad((output*torch.tensor(seed)).sum(),active)
    eps=1e-6
    # All parameter and input directions participate, independently of the AD
    # implementation; separate native fixtures also check each full gradient.
    scalar=[]
    for sign in [1,-1]:
        xs=[x+sign*eps*d for x,d in zip(values,directions)]
        ps={name:p+sign*eps*d for (name,p),d in zip(state.items(),directions[len(inputs):])}
        scalar.append((reference(xs,ps,**opts)*seed).sum())
    np.testing.assert_allclose(sum((g.numpy()*d).sum() for g,d in zip(gradients,directions)),(scalar[0]-scalar[1])/(2*eps),atol=3e-6,rtol=8e-6)
    primal,tangent=torch.func.jvp(lambda *x:call_model(m,kind,x,tm,tp),tuple(inputs),tuple(torch.tensor(d) for d in directions[:len(inputs)]))
    high,low=[reference([x+sign*eps*d for x,d in zip(values,directions)],state,**opts) for sign in [1,-1]]
    np.testing.assert_allclose(primal.numpy(),want,atol=3e-12,rtol=4e-11)
    np.testing.assert_allclose(tangent.numpy(),(high-low)/(2*eps),atol=3e-6,rtol=8e-6)


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('pre',[False,True])
def test_floating_mask_vjp_and_mask_jvp(kind,pre):
    rng,m,inputs,values,state=fixture(kind,'batch',pre)
    data=rng.normal(0,.2,size=(3,3));mask=torch.tensor(data,requires_grad=True)
    direction=rng.normal(size=data.shape);seed=rng.normal(size=inputs[0].shape)
    fn=lambda x:call_model(m,kind,inputs,x)
    output=fn(mask);grad,=torch.autograd.grad((output*torch.tensor(seed)).sum(),[mask])
    p,t=torch.func.jvp(fn,(mask,),(torch.tensor(direction),))
    eps=1e-6
    high,low=[reference(values,state,kind,'batch',pre,mask=data+sign*eps*direction) for sign in [1,-1]]
    expected=(high-low)/(2*eps)
    np.testing.assert_allclose(t.numpy(),expected,atol=3e-6,rtol=8e-6)
    np.testing.assert_allclose((grad.numpy()*direction).sum(),(expected*seed).sum(),atol=3e-6,rtol=8e-6)


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('pre',[False,True])
@pytest.mark.parametrize('bias,ff',[(False,7),(True,0)])
def test_float32_biasless_and_zero_hidden_dimension(kind,pre,bias,ff):
    _,m,inputs,values,state=fixture(kind,'batch',pre,dtype=np.float32,bias=bias,ff=ff)
    y=call_model(m,kind,inputs);want=reference(values,state,kind,'batch',pre,ff=ff)
    assert y.dtype==torch.float32
    np.testing.assert_allclose(y.numpy(),want,atol=4e-6,rtol=5e-5)
    y.square().sum().backward();assert all(p.grad is not None for p in m.parameters())


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('pre',[False,True])
def test_dropout_one_eval_and_backward_replay(kind,pre):
    _,m,inputs,values,state=fixture(kind,'batch',pre,p=1.)
    y=call_model(m,kind,inputs);np.testing.assert_allclose(y.numpy(),reference(values,state,kind,'batch',pre,p=1.),atol=1e-12,rtol=1e-11)
    m.eval();a=call_model(m,kind,inputs);b=call_model(m,kind,inputs)
    np.testing.assert_array_equal(a.numpy(),b.numpy())
    np.testing.assert_allclose(a.numpy(),reference(values,state,kind,'batch',pre),atol=1e-12,rtol=1e-11)
    for child in m.modules():
        if isinstance(child,(torch.nn.Dropout,torch.nn.MultiheadAttention)):
            if isinstance(child,torch.nn.Dropout):child.p=.25
            else:child.dropout=.25
    m.train();torch.manual_seed(5302)
    with torch.random.fork_rng():first=call_model(m,kind,inputs)
    second=call_model(m,kind,inputs);np.testing.assert_array_equal(first.numpy(),second.numpy())
    state=np.random.get_state();first.square().sum().backward()
    np.testing.assert_array_equal(np.random.get_state()[1],state[1]);assert np.random.get_state()[2:]==state[2:]


@pytest.mark.parametrize('activation',['relu','gelu',torch.nn.functional.relu,torch.nn.functional.gelu,torch.nn.ReLU(),torch.nn.GELU('tanh')])
@pytest.mark.parametrize('kind',KINDS)
def test_activation_state_pickle_deepcopy_and_dtype_conversion(kind,activation):
    _,m,inputs,_,_=fixture(kind,'batch',False,activation)
    want=call_model(m,kind,inputs).numpy()
    for clone in [copy.deepcopy(m),pickle.loads(pickle.dumps(m))]:
        np.testing.assert_array_equal(call_model(clone,kind,inputs).numpy(),want)
        assert all(a is not b for a,b in zip(m.parameters(),clone.parameters()))
    assert all(p.dtype==torch.float32 for p in copy.deepcopy(m).float().parameters())


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('name',['sin','relu','tanh','sigmoid','nn.functional.silu','nn.functional.softplus'])
@pytest.mark.parametrize('protocol',[2,pickle.HIGHEST_PROTOCOL])
def test_registered_activation_pickle_keeps_values_and_derivatives(kind,name,protocol):
    activation=torch
    for part in name.split('.'):
        activation=getattr(activation,part)
    _,model,inputs,_,_=fixture(kind,'batch',True,activation)
    def evaluate(module):
        def function(*xs):return call_model(module,kind,xs)
        directions=tuple(torch.ones_like(x)*.13 for x in inputs)
        primal,tangent=torch.func.jvp(function,tuple(inputs),directions)
        active=inputs+list(module.parameters())
        first=torch.autograd.grad(primal.square().sum(),active,retain_graph=True)
        mixed=torch.autograd.grad(tangent.square().sum(),active,allow_unused=True)
        mixed=[torch.zeros_like(x) if g is None else g for x,g in zip(active,mixed)]
        return [primal,tangent,*first,*mixed]
    want=evaluate(model)
    shallow=copy.copy(model)
    assert all(a is b for a,b in zip(model.parameters(),shallow.parameters()))
    for actual,expected in zip(evaluate(shallow),want):
        np.testing.assert_array_equal(actual.numpy(),expected.numpy())
    for clone in [copy.deepcopy(model),pickle.loads(pickle.dumps(model,protocol=protocol))]:
        for actual,expected in zip(evaluate(clone),want):
            np.testing.assert_array_equal(actual.numpy(),expected.numpy())
        assert all(a is not b for a,b in zip(model.parameters(),clone.parameters()))
        for layer in clone.modules():
            if isinstance(layer,(torch.nn.TransformerEncoderLayer,torch.nn.TransformerDecoderLayer)):
                assert layer.activation is activation
    # Serialization must not replace the live callable with a proxy.
    for layer in model.modules():
        if isinstance(layer,(torch.nn.TransformerEncoderLayer,torch.nn.TransformerDecoderLayer)):
            assert layer.activation is activation


def test_registered_activation_from_unregistered_namespace_and_alias_scope():
    import sys
    alternative=create_torch(name='isolated_test_namespace')
    model=alternative.nn.TransformerEncoderLayer(4,2,8,dropout=0,activation=alternative.sin)
    original=sys.modules.get('torch')
    data=pickle.dumps(model)
    assert copy.deepcopy(model).activation is alternative.sin
    with as_torch():
        restored=pickle.loads(data)
        assert restored.activation is torch.sin
        assert restored.activation is not sys.modules['torch'].sin
    assert sys.modules.get('torch') is original
    np.testing.assert_array_equal(restored(torch.ones(3,1,4)).numpy(),model(torch.ones(3,1,4)).numpy())


def test_clone_independence_registration_and_causal_hint_detection():
    layer=torch.nn.TransformerEncoderLayer(4,2,8,dropout=0.,batch_first=True,dtype=torch.float64)
    encoder=torch.nn.TransformerEncoder(layer,2,enable_nested_tensor=True)
    assert encoder.enable_nested_tensor and not encoder.use_nested_tensor
    assert encoder.layers[0] is not layer and encoder.layers[0] is not encoder.layers[1]
    for a,b in zip(encoder.layers[0].parameters(),encoder.layers[1].parameters()):
        np.testing.assert_array_equal(a.numpy(),b.numpy());assert a is not b and a._version==b._version==0
    x=torch.tensor(np.random.default_rng(5303).normal(size=(2,3,4)),requires_grad=True)
    causal=torch.nn.Transformer.generate_square_subsequent_mask(3,dtype=torch.float64)
    y=encoder(x,mask=causal);hinted=encoder(x,mask=causal,is_causal=True)
    np.testing.assert_array_equal(y.numpy(),hinted.numpy())
    mask=causal.clone();mask[0,2]=.1
    a=encoder(x,mask=mask);b=encoder(x,mask=mask,is_causal=False)
    np.testing.assert_array_equal(a.numpy(),b.numpy())
    assert not np.allclose(a.numpy(),hinted.numpy())
    with pytest.raises(RuntimeError,match='attn_mask'):encoder(x,is_causal=True)
    assert 'layers.0.self_attn.in_proj_weight' in encoder.state_dict()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_complete_training_optimizer_scheduler_checkpoint_continuation(dtype):
    def build():
        m=torch.nn.Transformer(4,2,1,1,dim_feedforward=8,dropout=.2,batch_first=True,dtype=dtype)
        o=torch.optim.AdamW(m.parameters(),lr=.002,amsgrad=True)
        return m,o,torch.optim.lr_scheduler.StepLR(o,2,.8)
    def step(m,o,s):
        o.zero_grad();y=m(src,tgt,tgt_mask=mask);(y-target).square().mean().backward();o.step();s.step()
    torch.manual_seed(5304);m,o,s=build()
    src=torch.randn(2,4,4,dtype=dtype);tgt=torch.randn(2,3,4,dtype=dtype);target=torch.randn(2,3,4,dtype=dtype)
    mask=torch.nn.Transformer.generate_square_subsequent_mask(3,dtype=dtype)
    for _ in range(2):step(m,o,s)
    checkpoint=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict()))
    with torch.random.fork_rng():clone,co,cs=build()
    clone.load_state_dict(checkpoint['model']);cs.load_state_dict(checkpoint['scheduler']);co.load_state_dict(checkpoint['optimizer'])
    for _ in range(3):
        with torch.random.fork_rng():step(m,o,s)
        step(clone,co,cs)
        for a,b in zip(m.parameters(),clone.parameters()):np.testing.assert_array_equal(a.numpy(),b.numpy())


def test_custom_encoder_decoder_and_native_style_matrix_reset():
    class Encoder(torch.nn.Module):
        def __init__(self):super().__init__();self.proj=torch.nn.Linear(4,4);self.seen=[]
        def forward(self,src,**options):self.seen.append(options);return self.proj(src)
    class Decoder(torch.nn.Module):
        def __init__(self):super().__init__();self.proj=torch.nn.Linear(4,4);self.seen=[]
        def forward(self,tgt,memory,**options):self.seen.append(options);return self.proj(tgt)+memory.mean(1,True)
    enc,dec=Encoder(),Decoder();old=enc.proj.weight.numpy().copy()
    m=torch.nn.Transformer(4,2,custom_encoder=enc,custom_decoder=dec,batch_first=True)
    assert m.encoder is enc and m.decoder is dec and not np.array_equal(old,enc.proj.weight.numpy())
    x=torch.ones(2,4,4);t=torch.ones(2,3,4);y=m(x,t,src_is_causal=False,tgt_is_causal=False)
    assert y.shape==(2,3,4) and enc.seen[-1]['is_causal'] is False and dec.seen[-1]['tgt_is_causal'] is False
    y.square().sum().backward();assert all(p.grad is not None for p in m.parameters())
    with pytest.raises(RuntimeError,match='batch number'):m(x,torch.ones(3,3,4))
    with pytest.raises(RuntimeError,match='feature number'):m(x,torch.ones(2,3,5))


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('pre',[False,True])
def test_custom_activation_callable_math_and_forward_ad(kind,pre):
    rng,m,inputs,values,state=fixture(kind,'batch',pre,activation=lambda x:x.sin())
    directions=[rng.normal(size=x.shape) for x in values]
    p,t=torch.func.jvp(lambda *x:call_model(m,kind,x),tuple(inputs),tuple(torch.tensor(d) for d in directions))
    np.testing.assert_allclose(p.numpy(),reference(values,state,kind,'batch',pre,activation='sin'),atol=3e-12,rtol=4e-11)
    eps=1e-6
    high,low=[reference([v+sign*eps*d for v,d in zip(values,directions)],state,kind,'batch',pre,activation='sin') for sign in [1,-1]]
    np.testing.assert_allclose(t.numpy(),(high-low)/(2*eps),atol=3e-6,rtol=8e-6)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.float16,torch.bool,torch.complex64])
@pytest.mark.parametrize('size',[0,1,4])
def test_causal_mask_generation_and_import_namespace(dtype,size):
    mask=torch.nn.Transformer.generate_square_subsequent_mask(size,dtype=dtype)
    expected=np.empty((size,size),dtype=dtype)
    for row in range(size):
        for column in range(size):expected[row,column]=-np.inf if column>row else 0.
    np.testing.assert_array_equal(mask.numpy(),expected);assert mask.dtype==dtype and not mask.requires_grad
    with as_torch():
        from torch.nn.modules.transformer import TransformerEncoderLayer,TransformerDecoderLayer,TransformerEncoder,TransformerDecoder,Transformer
        assert [TransformerEncoderLayer,TransformerDecoderLayer,TransformerEncoder,TransformerDecoder,Transformer]==[getattr(torch.nn,n) for n in ['TransformerEncoderLayer','TransformerDecoderLayer','TransformerEncoder','TransformerDecoder','Transformer']]


def test_initialization_reset_and_saved_version_guards():
    layer=torch.nn.Linear(3,4,dtype=torch.float64)
    assert layer.weight._version==layer.bias._version==1
    old=layer.weight.numpy().copy();layer.reset_parameters()
    assert layer.weight._version==layer.bias._version==2 and not np.array_equal(old,layer.weight.numpy())
    norm=torch.nn.LayerNorm(4,dtype=torch.float64)
    assert norm.weight._version==norm.bias._version==1
    norm.reset_parameters();assert norm.weight._version==norm.bias._version==2
    _,m,xs,_,_=fixture('full','batch',True)
    y=call_model(m,'full',xs)
    with torch.no_grad():m.decoder.layers[0].linear1.weight.add_(.01)
    with pytest.raises(RuntimeError,match='modified in-place'):y.sum().backward()


def test_float32_mixed_gradient_against_frozen_exact_input_float64_oracle():
    # Native Torch 2.14.1, fastpath disabled / SDPA MATH. All model/input/seed/
    # direction values are first rounded to FP32, then promoted for the oracle.
    rng=np.random.default_rng(530179)
    m=create('full','unbatched',True,dtype=torch.float32)
    with torch.no_grad():
        for name,p in m.named_parameters():
            p.copy_(torch.tensor(rng.normal(1 if 'norm' in name and name.endswith('weight') else 0,.2,size=p.shape).astype(np.float32)))
    values=[rng.normal(0,.5,size=(3,4)).astype(np.float32),rng.normal(0,.5,size=(4,4)).astype(np.float32)]
    seed=rng.normal(size=(3,4)).astype(np.float32)
    directions=[rng.normal(size=x.shape).astype(np.float32) for x in values]
    expected=np.array([-8.470739313069759,8.854863326255671,-4.885289499956057,.03141273055055993])
    for dtype in [torch.float32,torch.float64]:
        model=copy.deepcopy(m).to(dtype=dtype)
        xs=tuple(torch.tensor(np.asfortranarray(v),dtype=dtype,requires_grad=True) for v in values)
        ds=tuple(torch.tensor(v,dtype=dtype,requires_grad=True) for v in directions)
        mask=torch.nn.Transformer.generate_square_subsequent_mask(3,dtype=dtype)
        p,t=torch.func.jvp(lambda q,s:model(s,q,tgt_mask=mask),xs,ds)
        incoming=torch.tensor(seed,dtype=dtype)
        gradient,=torch.autograd.grad((p*incoming+t*incoming).sum(),[model.encoder.layers[0].norm1.bias])
        if dtype==torch.float64:
            np.testing.assert_allclose(gradient.numpy(),expected,atol=3e-11,rtol=3e-10)
        else:
            error=np.max(abs(gradient.numpy()-expected))/max(1.,np.max(abs(expected)))
            assert error<=1e-5,error


@pytest.mark.parametrize('epsilon',[0.,1e-12,1e-5])
def test_layernorm_epsilon_paths_keep_public_dtype_and_first_derivative(epsilon):
    data=np.array([[1.,2.,3.,4.],[-2.,1.,.5,3.]],dtype=np.float32)
    direction=np.array([[.1,-.2,.3,-.4],[.4,.3,-.2,-.1]],dtype=np.float32)
    x=torch.tensor(data,requires_grad=True);norm=torch.nn.LayerNorm(4,eps=epsilon)
    p,t=torch.func.jvp(norm,(x,),(torch.tensor(direction),))
    high=data.astype(np.float64);d=direction.astype(np.float64)
    centered=high-high.mean(-1,keepdims=True);projected=d-d.mean(-1,keepdims=True)
    inverse=1/np.sqrt((centered*centered).mean(-1,keepdims=True)+epsilon)
    expected=projected*inverse-centered*inverse**3*(projected*centered).mean(-1,keepdims=True)
    assert p.dtype==t.dtype==torch.float32
    np.testing.assert_allclose(t.numpy(),expected,atol=2e-7,rtol=3e-6)


@pytest.mark.parametrize('kind',['encoder','decoder'])
def test_zero_depth_matches_dense_native_forward_failure(kind):
    layer=torch.nn.TransformerEncoderLayer(4,2) if kind=='encoder' else torch.nn.TransformerDecoderLayer(4,2)
    stack=torch.nn.TransformerEncoder(layer,0) if kind=='encoder' else torch.nn.TransformerDecoder(layer,0)
    assert len(stack.layers)==0
    x=torch.ones(3,2,4)
    with pytest.raises(IndexError):stack(x) if kind=='encoder' else stack(x,x)


@pytest.mark.parametrize('options,error',[({'norm_first':1},TypeError),({'activation':'bogus'},RuntimeError),({'activation':3},TypeError),({'dim_feedforward':-1},ValueError),({'dim_feedforward':True},TypeError),({'dropout':float('nan')},ValueError)])
def test_constructor_guards(options,error):
    with pytest.raises(error):torch.nn.TransformerEncoderLayer(4,2,**options)
