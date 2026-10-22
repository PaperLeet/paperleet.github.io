"""Consumed joint cotangents must not stay pinned during predecessor work."""
import weakref
import numpy as np
import pytest
from torchwasm import torch
from torchwasm._autograd import joint_result

@pytest.mark.parametrize('method',['backward','targeted_grad'])
@pytest.mark.parametrize('selective',[False,True])
@pytest.mark.parametrize('retained_first',[False,True])
def test_consumed_cotangents_release_before_leaf_hooks(method,selective,retained_first):
    x=torch.ones(32,dtype=torch.float64,requires_grad=True)
    y=torch.ones(32,dtype=torch.float64,requires_grad=True)
    a=x*2;b=y*3;refs={};observed=[]
    def pullback(g,needs):
        first=np.full(32,float(g)*4) if needs[0] else None
        second=np.full(32,float(g)*5) if needs[1] else None
        refs.clear()
        for name,value in [('a',first),('b',second)]:
            if value is not None:refs[name]=weakref.ref(value)
        return first,second
    def inspect(name):
        def hook(g):
            assert refs[name]() is None
            observed.append(name)
        return hook
    x.register_hook(inspect('a'))
    y.register_hook(inspect('b'))
    output=joint_result(np.array(np.sum(4*a.numpy()+5*b.numpy()),dtype=np.float64),(a,b),pullback,prune_inputs=selective)
    for index,seed in enumerate([2.,-3.] if retained_first else [2.]):
        x.grad=y.grad=None;retain=retained_first and index==0;start=len(observed)
        if method=='targeted_grad':
            result=torch.autograd.grad(output,x,torch.tensor(seed,dtype=torch.float64),retain_graph=retain)[0]
            np.testing.assert_array_equal(result.numpy(),np.full(32,8*seed))
            assert x.grad is y.grad is None
        else:
            output.backward(torch.tensor(seed,dtype=torch.float64),retain_graph=retain)
            np.testing.assert_array_equal(x.grad.numpy(),np.full(32,8*seed))
            np.testing.assert_array_equal(y.grad.numpy(),np.full(32,15*seed))
        assert len(observed)>start
        assert all(ref() is None for ref in refs.values())
