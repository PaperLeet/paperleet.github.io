"""Native gradient layout, retention and user-visible storage ownership."""
import itertools
import warnings
import numpy as np
import pytest
import torch as native
from stridetorch import torch


def layouts(dtype):
    base=np.arange(24,dtype=dtype).reshape(2,3,4)/8
    cases=[base.transpose(p) for p in itertools.permutations(range(3))]
    cases += [base[:,::2,:],base[...,::2],base[1:,:,:],np.broadcast_to(base[:1,:1,:],(2,3,4))]
    for shape,strides in [((),()),((1,1),(0,0)),((2,1,3),(3,99,1)),
                          ((2,1,3),(3,0,1)),((0,3),(3,1)),((2,0,3),(0,3,1)),
                          ((3,0,2),(2,22,1)),((0,),(0,))]:
        cases.append(np.ndarray(shape,dtype=dtype,buffer=base,
                                strides=tuple(s*base.itemsize for s in strides)))
    return cases


@pytest.mark.parametrize('dtype',['float16','float32','float64'])
@pytest.mark.parametrize('case',range(18))
@pytest.mark.parametrize('seed_layout',['C','F','broadcast'])
def test_first_leaf_gradient_native_strides_and_seed_independence(dtype,case,seed_layout):
    data=layouts(dtype)[case];results=[]
    for api in (native,torch):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');x=api.from_numpy(data).requires_grad_()
        values=np.arange(data.size,dtype=dtype).reshape(data.shape)/16+.5
        values=np.array(values,order=seed_layout) if seed_layout in ('C','F') else np.broadcast_to(np.array(.5,dtype=dtype),data.shape)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');seed=api.from_numpy(values)
        before=seed.numpy().copy();x.backward(seed)
        results.append((x.grad.stride(),x.grad.numpy().copy()))
        assert not np.shares_memory(x.grad.numpy(),seed.numpy())
        assert not np.shares_memory(x.grad.numpy(),x.detach().numpy())
        x.grad.add_(1);np.testing.assert_array_equal(seed.numpy(),before)
    assert results[0][0]==results[1][0]
    np.testing.assert_array_equal(results[1][1],results[0][1])


@pytest.mark.parametrize('dtype',['float16','float32','float64'])
@pytest.mark.parametrize('permutation',list(itertools.permutations(range(3))))
@pytest.mark.parametrize('hook',[False,True])
def test_dense_leaf_operator_hook_and_post_hook_contract(dtype,permutation,hook):
    data=np.arange(24,dtype=dtype).reshape(2,3,4).transpose(permutation)/8
    results=[]
    for api in (native,torch):
        x=api.from_numpy(data).requires_grad_();seen=[];replacements=[]
        if hook:
            def replace(g):
                result=g*2;replacements.append(result);return result
            x.register_hook(replace)
        x.register_post_accumulate_grad_hook(lambda p:seen.append(p.grad.stride()))
        x.square().sum().backward();first=x.grad;version=first._version
        x.square().sum().backward()
        assert x.grad is first and first._version==version+1
        assert seen==[x.stride(),x.stride()]
        if hook:
            saved=x.grad.numpy().copy();replacements[0].fill_(99)
            np.testing.assert_array_equal(x.grad.numpy(),saved)
        results.append(x.grad.numpy())
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize('preset',['C','F'])
def test_existing_leaf_gradient_keeps_identity_layout_and_old_dtype(preset):
    for api in (native,torch):
        x=api.from_numpy(np.ones((2,3),dtype='float64',order='F')).requires_grad_()
        x.grad=api.from_numpy(np.full((2,3),7.,order=preset));old=x.grad
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');x.sum().backward()
        assert x.grad is old
        assert x.grad.stride()==((3,1) if preset=='C' else (1,2))
        np.testing.assert_array_equal(x.grad.numpy(),np.full((2,3),8.))
        model=api.nn.Linear(3,2,dtype=api.float32);loss=(model.weight*2).sum()
        model.to(api.float64);loss.backward()
        assert model.weight.grad.dtype==api.float32


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('case',range(18))
def test_retained_nonleaf_gradient_copy_and_replacement(dtype,case):
    data=layouts(dtype)[case];results=[]
    for api in (native,torch):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');x=api.from_numpy(data).requires_grad_()
        middle=x*2;middle.retain_grad()
        seed=api.ones(middle.shape,dtype=getattr(api,dtype));middle.backward(seed,retain_graph=True)
        old=middle.grad;old_version=old._version
        middle.backward(seed)
        assert middle.grad is not old and old._version==old_version
        np.testing.assert_array_equal(old.numpy(),np.ones(data.shape))
        np.testing.assert_array_equal(middle.grad.numpy(),np.full(data.shape,2.))
        assert not np.shares_memory(old.numpy(),seed.numpy())
        results.append((old.stride(),middle.grad.stride()))
    assert results[0]==results[1]
