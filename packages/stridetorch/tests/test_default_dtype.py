"""Default floating policy across factories, operators and model lifecycle."""
import io
import sys

import numpy as np
import pytest

from stridetorch import UnsupportedOperation, as_torch, create_torch, torch


@pytest.fixture(autouse=True)
def restore_default():
    previous=torch.get_default_dtype()
    try:yield
    finally:torch.set_default_dtype(previous)


FACTORIES={
    'scalar':lambda:torch.tensor(.125),
    'small_list':lambda:torch.tensor([.125,.25]),
    'large_list':lambda:torch.tensor([.125]*12),
    'nested':lambda:torch.tensor([[1.,2],[3.,4.]]),
    'empty':lambda:torch.tensor([]),
    'nested_empty':lambda:torch.tensor([[],[]]),
    'as_tensor':lambda:torch.as_tensor([.125,.25]),
    'zeros':lambda:torch.zeros(2,3),
    'ones':lambda:torch.ones((2,3),dtype=None),
    'empty_factory':lambda:torch.empty(0,3),
    'eye':lambda:torch.eye(3,dtype=None),
    'linspace':lambda:torch.linspace(-1,1,5),
    'linspace_numpy':lambda:torch.linspace(np.float32(-1),np.float64(1),5),
    'arange_float':lambda:torch.arange(0.,2.,.5),
    'arange_numpy':lambda:torch.arange(np.float32(2)),
    'arange_tensor':lambda:torch.arange(torch.tensor(2.,dtype=torch.float32)),
    'full':lambda:torch.full((2,3),.125),
    'full_numpy':lambda:torch.full((2,3),np.float32(.125)),
    'full_tensor':lambda:torch.full((2,3),torch.tensor(.125,dtype=torch.float32)),
    'empty_parameter':lambda:torch.nn.Parameter(),
}


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64])
@pytest.mark.parametrize('name',FACTORIES)
def test_default_factory_inference(dtype,name):
    torch.set_default_dtype(dtype)
    value=FACTORIES[name]()
    assert value.dtype==dtype


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64])
def test_integer_boolean_explicit_and_existing_storage_are_independent(dtype):
    source=np.array([.125,.25],dtype=np.float32)
    existing=torch.from_numpy(source)
    torch.set_default_dtype(dtype)
    for fn in [torch.tensor,torch.as_tensor,torch.from_numpy]:assert fn(source).dtype==torch.float32
    assert torch.tensor(np.float64(.125)).dtype==torch.float64
    assert torch.tensor([np.float32(.125)]).dtype==torch.float32
    assert torch.tensor([np.float64(.125),.25]).dtype==torch.float64
    assert torch.tensor([1,2]).dtype==torch.int64 and torch.tensor([True]).dtype==torch.bool
    assert torch.arange(3).dtype==torch.int64 and torch.full((2,),1).dtype==torch.int64
    assert torch.full((2,),True).dtype==torch.bool
    for name in ['zeros','ones','empty','randn']:
        assert getattr(torch,name+'_like')(existing,dtype=None).dtype==torch.float32
    assert torch.full_like(existing,.5).dtype==torch.float32
    assert existing.new_tensor([.5]).dtype==torch.float32
    assert existing.new_ones(2).dtype==torch.float32
    assert torch.tensor([.5],dtype=torch.float64).dtype==torch.float64
    assert torch.zeros(2,dtype=torch.float32).dtype==torch.float32
    assert torch.as_tensor(source) is not existing
    torch.as_tensor(source)[0]=3.;assert existing[0].item()==3.


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_default_complex_inference_and_explicit_numpy_precision(dtype):
    torch.set_default_dtype(dtype);complex_dtype=torch.complex128 if dtype==torch.float64 else torch.complex64
    for fn in [lambda:torch.tensor(1+2j),lambda:torch.tensor([1+2j]*12),lambda:torch.tensor([[1j],[2.]]),
               lambda:torch.full((2,),1j),lambda:torch.linspace(0,1j,3)]:
        assert fn().dtype==complex_dtype
    value=torch.linspace(0,1j,3);np.testing.assert_array_equal(value.numpy(),[0,.5j,1j])
    assert torch.tensor([np.complex64(1j)]).dtype==torch.complex64
    assert torch.tensor(np.complex128(1j)).dtype==torch.complex128
    assert torch.promote_types(torch.float32,torch.complex64)==torch.complex64


@pytest.mark.parametrize('name',['sqrt','rsqrt','exp','log','sin','cos','tanh','sigmoid','acos','reciprocal'])
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64])
def test_integer_transcendentals_use_current_default(dtype,name):
    torch.set_default_dtype(dtype)
    x=torch.tensor([1,1],dtype=torch.int64)
    result=getattr(x,name)()
    assert result.dtype==dtype
    np.testing.assert_allclose(result.numpy(),getattr(x.to(dtype=dtype),name)().numpy(),rtol=0,atol=0)
    assert (x/2).dtype==dtype and (2/x).dtype==dtype
    assert (x**.5).dtype==dtype and x.logsumexp(0).dtype==dtype


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_wrapped_scalars_rank_and_category_promotion(dtype):
    torch.set_default_dtype(dtype)
    for rank in [(),(2,)]:
        integer=torch.ones(rank,dtype=torch.int64);single=torch.ones(rank,dtype=torch.float32)
        boolean=torch.ones(rank,dtype=torch.bool)
        assert (integer+.25).dtype==dtype
        assert (boolean+.25).dtype==dtype
        assert (single+.25).dtype==torch.float32
        assert (single+1j).dtype==torch.complex64
        assert (integer+1j).dtype==(torch.complex128 if dtype==torch.float64 else torch.complex64)
        assert torch.result_type(integer,.25)==dtype
        assert torch.where(boolean,.25,.75).dtype==dtype
    assert (torch.ones(2,dtype=torch.float32)+torch.tensor(.25,dtype=torch.float64)).dtype==torch.float32
    assert (torch.ones(2,dtype=torch.float32)+torch.ones(2,dtype=torch.float64)).dtype==torch.float64


MODELS={
    'Linear':lambda **kw:torch.nn.Linear(3,4,**kw),
    'Embedding':lambda **kw:torch.nn.Embedding(5,4,**kw),
    'LayerNorm':lambda **kw:torch.nn.LayerNorm(4,**kw),
    'RMSNorm':lambda **kw:torch.nn.RMSNorm(4,**kw),
    'GroupNorm':lambda **kw:torch.nn.GroupNorm(2,4,**kw),
    **{name:lambda _name=name,**kw:getattr(torch.nn,_name)(4,affine=True,track_running_stats=True,**kw)
       for name in ['BatchNorm1d','BatchNorm2d','BatchNorm3d','InstanceNorm1d','InstanceNorm2d','InstanceNorm3d']},
    **{name:lambda _name=name,**kw:getattr(torch.nn,_name)(2,4,3,**kw)
       for name in ['Conv1d','Conv2d','Conv3d','ConvTranspose1d','ConvTranspose2d','ConvTranspose3d']},
    'MHA':lambda **kw:torch.nn.MultiheadAttention(4,2,add_bias_kv=True,**kw),
    'Encoder':lambda **kw:torch.nn.TransformerEncoderLayer(4,2,7,dropout=0,**kw),
    'Decoder':lambda **kw:torch.nn.TransformerDecoderLayer(4,2,7,dropout=0,**kw),
    'Transformer':lambda **kw:torch.nn.Transformer(4,2,1,1,dim_feedforward=7,dropout=0,**kw),
}


@pytest.mark.parametrize('name',MODELS)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_model_initialization_matches_explicit_dtype_and_rng(dtype,name):
    torch.set_default_dtype(dtype);torch.manual_seed(5501);implicit=MODELS[name]()
    torch.manual_seed(5501);explicit=MODELS[name](dtype=dtype)
    for key,value in implicit.state_dict().items():
        assert value.dtype==(dtype if value.is_floating_point() else torch.int64)
        np.testing.assert_array_equal(value.numpy(),explicit.state_dict()[key].numpy())
    torch.set_default_dtype(torch.float32 if dtype==torch.float64 else torch.float64)
    assert all(p.dtype==dtype for p in implicit.parameters())


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64])
@pytest.mark.parametrize('factory',['rand','randn'])
def test_random_factories_match_explicit_stream(dtype,factory):
    torch.set_default_dtype(dtype);torch.manual_seed(5502)
    value=getattr(torch,factory)(3,7)
    torch.manual_seed(5502);explicit=getattr(torch,factory)(3,7,dtype=dtype)
    assert value.dtype==dtype;np.testing.assert_array_equal(value.numpy(),explicit.numpy())


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_existing_forward_reverse_and_checkpoint_survive_default_change(dtype):
    torch.set_default_dtype(dtype)
    model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    x=torch.tensor([[.1,.2,.3]],requires_grad=True)
    y,j=torch.func.jvp(model,(x,),(torch.ones_like(x),))
    torch.set_default_dtype(torch.float32 if dtype==torch.float64 else torch.float64)
    (y.square().sum()+j.square().sum()).backward();optimizer.step()
    assert x.grad.dtype==dtype and all(p.dtype==p.grad.dtype==dtype for p in model.parameters())
    expected_step=torch.get_default_dtype()
    assert all(s['step'].dtype==expected_step for s in optimizer.state.values())
    stream=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},stream);stream.seek(0)
    restored=torch.load(stream);target=torch.nn.Linear(3,2,dtype=dtype);opt=torch.optim.AdamW(target.parameters())
    target.load_state_dict(restored['model']);opt.load_state_dict(restored['optimizer'])
    np.testing.assert_array_equal(target(x).numpy(),model(x).numpy())
    assert all(s['step'].dtype==expected_step for s in opt.state.values())


def test_half_storage_limits_and_complex_factory_exception():
    torch.set_default_dtype(torch.float16)
    assert torch.get_default_dtype()==torch.float16
    assert torch.full((2,),1j).dtype==torch.complex64
    assert torch.tensor([1j],dtype=torch.complex64).dtype==torch.complex64
    for dtype in [torch.float32,torch.float64,torch.complex64,torch.complex128]:
        value=torch.ones(2,dtype=dtype)
        assert (value+1j).dtype==(torch.complex128 if dtype in [torch.float64,torch.complex128] else torch.complex64)
    for fn in [lambda:torch.tensor(1j),lambda:torch.linspace(0,1j,3),lambda:torch.nn.Linear(2,2)]:
        with pytest.raises(UnsupportedOperation):fn()
    assert torch.get_default_dtype()==torch.float16


def test_half_normal_tail_keeps_native_cpu_random_position():
    # Frozen PyTorch 2.14.1 CPU oracle: an incomplete vector tail is redrawn
    # as a full block before subsequent float64 scalar/cached normal draws.
    torch.set_default_dtype(torch.float16);torch.manual_seed(5507)
    value=torch.randn(21)
    expected=[-1.3037109375,.8525390625,1.0263671875,-1.712890625,-.9580078125,
              .8486328125,-.384521484375,-.79833984375,-3.373046875,.412841796875,
              .802734375,.180908203125,.01560211181640625,-.53369140625,.07666015625,
              .2939453125,2.26953125,.401123046875,-.8310546875,.470703125,.75732421875]
    np.testing.assert_allclose(value.numpy(),expected,rtol=1e-3,atol=1e-4)
    torch.set_default_dtype(torch.float64)
    np.testing.assert_allclose(torch.randn(3).numpy(),[-.17733059517119773,.14034071792081193,.8665243414169592],rtol=3e-14,atol=3e-14)
    torch.set_default_dtype(torch.float32)
    np.testing.assert_array_equal(torch.rand(4).numpy(),[.042602479457855225,.13620585203170776,.027905285358428955,.29607075452804565])


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.complex64,np.complex128])
@pytest.mark.parametrize('nested',[False,True])
def test_half_complex_inference_promotes_before_storage_validation(dtype,nested):
    torch.set_default_dtype(torch.float16)
    expected_dtype=torch.complex128 if dtype in [np.float64,np.complex128] else torch.complex64
    for scalar in [dtype(1),torch.tensor(dtype(1))]:
        for values in [[scalar,1j],[1j,scalar]]:
            data=[values] if nested else values
            for fn in [torch.tensor,torch.as_tensor]:
                value=fn(data)
                assert value.dtype==expected_dtype
                want=[1,1j] if values[0] is scalar else [1j,1]
                np.testing.assert_array_equal(value.numpy(),[want] if nested else want)


@pytest.mark.parametrize('name',['LayerNorm','RMSNorm','GroupNorm','BatchNorm2d','InstanceNorm2d'])
def test_parameterless_normalization_and_pretrained_weights_ignore_unused_default(name):
    torch.set_default_dtype(torch.float16)
    shape=(2,4,3,3);x=torch.tensor(np.sin(np.arange(np.prod(shape))).reshape(shape).astype('float32'),requires_grad=True)
    if name in ['LayerNorm','RMSNorm']:model=getattr(torch.nn,name)((3,3),elementwise_affine=False)
    elif name=='GroupNorm':model=torch.nn.GroupNorm(2,4,affine=False)
    else:model=getattr(torch.nn,name)(4,affine=False,track_running_stats=False)
    assert not model.state_dict()
    y,j=torch.func.jvp(model,(x,),(torch.ones_like(x)*.13,))
    (y.square().sum()+j.square().sum()).backward()
    assert y.dtype==j.dtype==x.grad.dtype==torch.float32
    weight=torch.ones(3,4,dtype=torch.float32)
    embedded=torch.nn.Embedding.from_pretrained(weight)
    assert embedded.weight.dtype==torch.float32
    np.testing.assert_array_equal(embedded(torch.tensor([0,2])).numpy(),np.ones((2,4)))


@pytest.mark.parametrize('value',[None,True,1,'float64',float,np.float64,np.dtype('int64'),np.dtype('complex64')])
def test_invalid_default_is_atomic(value):
    torch.set_default_dtype(torch.float64)
    with pytest.raises(TypeError):torch.set_default_dtype(value)
    assert torch.get_default_dtype()==torch.float64


def test_alias_namespaces_share_only_the_owning_distribution_policy():
    old_host=sys.modules.get('torch');other=create_torch(name='default_dtype_probe')
    other.set_default_dtype(other.float64)
    assert torch.get_default_dtype()==torch.float64
    with as_torch() as alias:
        assert alias.get_default_dtype()==torch.float64
        alias.set_default_dtype(alias.float32)
    assert other.get_default_dtype()==torch.float32
    assert sys.modules.get('torch') is old_host
