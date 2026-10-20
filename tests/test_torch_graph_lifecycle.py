"""Reverse graph lifetime, multi-root differentiation and retained gradients."""
import gc
import weakref

import numpy as np
import pytest
import torch as native

from torchwasm import torch


def close(actual,expected,dtype='float64'):
    np.testing.assert_allclose(actual.detach().numpy(),expected.detach().numpy(),
                               rtol=4e-6 if dtype=='float32' else 2e-12,atol=2e-6 if dtype=='float32' else 2e-12)


STATELESS={
    'add':lambda t,a:t+2,'subtract':lambda t,a:2-t,'neg':lambda t,a:-t,
    'clone':lambda t,a:t.clone(),'cast':lambda t,a:t.to(dtype=a.float64),
    'sum':lambda t,a:t.sum(1),'mean':lambda t,a:t.mean(1),
    'reshape':lambda t,a:t.reshape(-1),'transpose':lambda t,a:t.T,
    'unsqueeze':lambda t,a:t.unsqueeze(0),'squeeze':lambda t,a:t.unsqueeze(0).squeeze(0),
    'expand':lambda t,a:t.unsqueeze(0).expand(3,2,3),'repeat':lambda t,a:t.repeat(2,1),
    'slice':lambda t,a:t[:,1:],'flip':lambda t,a:t.flip([1]),
    'roll':lambda t,a:t.roll(1,1),'triu':lambda t,a:t.triu(),'tril':lambda t,a:t.tril(),
    'cumsum':lambda t,a:t.cumsum(1),'round':lambda t,a:t.round(),
    'diagonal':lambda t,a:t.diagonal(),'diag':lambda t,a:t.diag(),
    'trace':lambda t,a:t.trace(),'cat':lambda t,a:a.cat([t,t],dim=0),
    'stack':lambda t,a:a.stack([t,t],dim=0),
    'remainder':lambda t,a:t.remainder(2.),
    'repeat_interleave':lambda t,a:t.repeat_interleave(2,dim=1),
    'scalar_tensor_index':lambda t,a:t[a.tensor(1)],
    'trapezoid':lambda t,a:a.trapezoid(t),
}


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('kind',STATELESS)
def test_stateless_paths_allow_repeated_backward(kind,dtype):
    leaves=[];outputs=[]
    for api in (torch,native):
        leaf=api.tensor([[.2,-.5,1.2],[.6,.1,-.4]],dtype=getattr(api,dtype),requires_grad=True)
        output=STATELESS[kind](leaf,api).sum()
        output.backward();output.backward()
        assert not output.is_leaf and output.requires_grad
        leaves.append(leaf);outputs.append(output)
    close(leaves[0].grad,leaves[1].grad,dtype);close(outputs[0],outputs[1],dtype)


SAVED={
    'square':lambda t,a:t.square(),'multiply':lambda t,a:t*2,
    'divide':lambda t,a:t/2,'exp':lambda t,a:t.exp(),'tanh':lambda t,a:t.tanh(),
    'softmax':lambda t,a:t.softmax(-1),'advanced_index':lambda t,a:t[[0,0,1]],
    'layer_norm':lambda t,a:a.nn.functional.layer_norm(t,(3,)),
}


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('kind',SAVED)
def test_saved_paths_honor_retention_and_release(kind,dtype):
    leaves=[]
    for api in (torch,native):
        leaf=api.tensor([[.2,-.5,1.2],[.6,.1,-.4]],dtype=getattr(api,dtype),requires_grad=True)
        output=SAVED[kind](leaf,api).sum()
        output.backward(retain_graph=True);output.backward(retain_graph=True);output.backward()
        assert not output.is_leaf and output.requires_grad
        with pytest.raises(RuntimeError,match='second time|freed'):output.backward()
        leaves.append(leaf)
    close(leaves[0].grad,leaves[1].grad,dtype)


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_multiple_outputs_duplicates_shared_branches_and_seeds(dtype):
    results=[]
    for api in (torch,native):
        x=api.tensor([.2,.5,-.7],dtype=getattr(api,dtype),requires_grad=True)
        intermediate=x.exp();u=intermediate.square();v=intermediate*3
        a=api.tensor([1.,-.2,.3],dtype=getattr(api,dtype));b=api.tensor([-.3,.4,.1],dtype=getattr(api,dtype))
        gradients=api.autograd.grad((u,v,u),(x,intermediate,x),(a,b,b),retain_graph=True)
        assert x.grad is None
        api.autograd.backward((u,v),(a,b))
        results.append((*gradients,x.grad))
    for actual,expected in zip(*results):close(actual,expected,dtype)


@pytest.mark.parametrize('method',['grad','backward'])
def test_intermediate_target_prunes_previously_released_ancestor(method):
    for api in (torch,native):
        x=api.tensor(2.,requires_grad=True);middle=x.square();out=middle.square()
        middle.backward();assert x.grad.item()==4
        if method=='grad':
            assert api.autograd.grad(out,middle)[0].item()==8
            assert middle.grad is None
        else:
            with pytest.raises(RuntimeError,match='second time|freed'):out.backward(inputs=[middle])
            assert middle.grad.item()==8
            assert middle.retains_grad
        assert x.grad.item()==4


def test_retained_nonleaf_gradients_accumulate_under_grad_without_leaf_grad():
    for api in (torch,native):
        x=api.tensor(2.,requires_grad=True);middle=x.square();middle.retain_grad();out=middle.square()
        assert middle.retains_grad and not x.retains_grad
        assert api.autograd.grad(out,x,retain_graph=True)[0].item()==32
        assert middle.grad.item()==8 and x.grad is None
        assert api.autograd.grad(out,x)[0].item()==32
        assert middle.grad.item()==16 and x.grad is None


@pytest.mark.parametrize('materialize',[False,True])
def test_unused_and_duplicate_inputs(materialize):
    for api in (torch,native):
        x=api.tensor(2.,requires_grad=True);unused=api.tensor([3.,4.],requires_grad=True)
        values=api.autograd.grad(x.square(),(x,unused,x),allow_unused=True,materialize_grads=materialize)
        assert values[0].item()==values[2].item()==4
        if materialize:
            assert values[1].tolist()==[0.,0.] and not values[1].requires_grad and values[1].is_leaf
        else:assert values[1] is None


@pytest.mark.parametrize('kind',['shape','frozen_input','seed_count'])
def test_invalid_requests_do_not_consume_saved_graph(kind):
    for api in (torch,native):
        x=api.tensor([2.,3.],requires_grad=True);out=x.square()
        with pytest.raises((RuntimeError,TypeError)):
            if kind=='shape':out.backward(api.ones(3))
            elif kind=='frozen_input':api.autograd.grad(out,api.ones(2),api.ones(2))
            else:api.autograd.grad((out,out),x,(api.ones(2),))
        out.sum().backward();assert x.grad.tolist()==[4.,6.]


@pytest.mark.parametrize('retain',[False,True])
def test_saved_intermediate_storage_is_released_when_requested(retain):
    x=torch.ones(16,32,requires_grad=True)
    middle=x.exp();reference=weakref.ref(middle._data)
    loss=middle.square().sum();del middle;gc.collect()
    assert reference() is not None
    loss.backward(retain_graph=retain);gc.collect()
    assert (reference() is not None)==retain
    if retain:
        loss.backward();gc.collect();assert reference() is None
    assert not loss.is_leaf


def test_leaf_cache_does_not_create_ownership_cycle():
    value=torch.tensor(2.,requires_grad=True);value.backward();reference=weakref.ref(value)
    del value
    assert reference() is None


def test_backward_inputs_executes_and_frees_selected_nonleaf_node():
    for api in (torch,native):
        x=api.tensor(2.,requires_grad=True);middle=x.square();out=middle.square()
        out.backward(inputs=[middle])
        assert middle.grad.item()==8 and middle.retains_grad and x.grad is None
        with pytest.raises(RuntimeError,match='second time|freed'):middle.backward()


@pytest.mark.parametrize('materialize',[False,True])
def test_named_parameter_inputs_preserve_keys_and_unused_values(materialize):
    for api in (torch,native):
        x=api.tensor(2.,requires_grad=True);unused=api.tensor(3.,requires_grad=True)
        values=api.autograd.grad(x.square(),{'weight':x,'frozen_branch':unused,'shared':x},
                                 allow_unused=True,materialize_grads=materialize)
        assert list(values)==['weight','frozen_branch','shared']
        assert values['weight'].item()==values['shared'].item()==4
        if materialize:assert values['frozen_branch'].item()==0
        else:assert values['frozen_branch'] is None


def test_only_inputs_deprecation_and_invalid_materialization():
    for api in (torch,native):
        x=api.tensor(2.,requires_grad=True)
        with pytest.warns(FutureWarning):assert api.autograd.grad(x.square(),x,only_inputs=False)[0].item()==4
        with pytest.raises(ValueError):api.autograd.grad(x.square(),x,allow_unused=False,materialize_grads=True)


@pytest.mark.parametrize('shape,index', [((2,3),(1,2)),((6,),-1),((),())])
def test_basic_scalar_selections_share_storage_and_mutation_versions(shape,index):
    results=[]
    for api in (torch,native):
        value=api.tensor(np.arange(max(1,int(np.prod(shape))))+1.,dtype=api.float64).reshape(shape)
        selected=value[index]
        assert selected.shape==()
        selected.fill_(7)
        results.append(value.numpy().copy())
        assert selected._version==value._version
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize('count',[0,1,2,3])
@pytest.mark.parametrize('spacing',[.7,float('inf'),float('nan'),float(np.nextafter(np.float32(0),np.float32(1)))])
def test_constant_spacing_trapezoid_boundary_and_underflow(count,spacing):
    outputs=[];grads=[]
    for api in (torch,native):
        x=api.ones(count,dtype=api.float32,requires_grad=True);out=api.trapezoid(x,dx=spacing)
        outputs.append(out.detach().numpy().copy())
        if count:
            out.backward();grads.append(x.grad.numpy().copy())
        else:assert not out.requires_grad
    np.testing.assert_allclose(*outputs,rtol=0,atol=0,equal_nan=True)
    if count:np.testing.assert_allclose(*grads,rtol=0,atol=0,equal_nan=True)


@pytest.mark.parametrize('mode,size',[('bicubic',(4,5)),('bicubic',(1,1)),('area',(1,1))])
def test_stateless_interpolation_specializations_repeat(mode,size):
    grads=[]
    for api in (torch,native):
        x=api.tensor(np.arange(24.).reshape(1,2,3,4),dtype=api.float64,requires_grad=True)
        output=api.nn.functional.interpolate(x,size=size,mode=mode)
        output.sum().backward();output.sum().backward();grads.append(x.grad.numpy().copy())
    np.testing.assert_allclose(*grads,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('shape,dim',[((),0),((),-1),((),()),((2,3),()),((2,3),[]),((2,3),-1)])
@pytest.mark.parametrize('operation',['sum','mean'])
def test_reduction_scalar_empty_axes_and_dtype(shape,dim,operation):
    results=[]
    for api in (torch,native):
        x=api.tensor(np.arange(max(1,int(np.prod(shape))))+.3,dtype=api.float32).reshape(shape).detach().requires_grad_()
        out=getattr(x,operation)(dim=dim,dtype=api.float64)
        out.sum().backward();out.sum().backward()
        assert out.dtype==api.float64
        results.append((out,x.grad))
    for actual,expected in zip(*results):close(actual,expected,'float32')


@pytest.mark.parametrize('operation',['sum','mean'])
def test_reduction_rejects_out_of_range_and_duplicate_axes(operation):
    for api in (torch,native):
        x=api.ones(2,3,requires_grad=True)
        with pytest.raises(IndexError):getattr(x,operation)(dim=2)
        with pytest.raises(RuntimeError):getattr(x,operation)(dim=(0,0))


@pytest.mark.parametrize('dtype',['int64','bool'])
def test_integer_sum_dtype_rejects_autograd_but_allows_no_grad(dtype):
    for api in (torch,native):
        x=api.tensor([1.,2.],requires_grad=True)
        with pytest.raises(RuntimeError):x.sum(dtype=getattr(api,dtype))
        with api.no_grad():
            out=x.sum(dtype=getattr(api,dtype));assert not out.requires_grad
