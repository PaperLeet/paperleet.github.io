"""Independent linear-polynomial gradients and minimal saved-value guards."""
import numpy as np
import pytest

from stridetorch import torch


SHAPES=[((2,2),(2,2)),((2,),(2,)),((2,),(2,2)),((2,2),(2,)),((2,2,2),(2,2))]
CASES=[(a,b,needs,selected,changed) for a,b in SHAPES
       for needs in [(True,True),(True,False),(False,True)]
       for selected in ['a','b','both']
       if not (selected=='a' and not needs[0] or selected=='b' and not needs[1])
       for changed in ['a','b','output']]


def exact_linear_gradient(a,b,which):
    # A matrix product is linear in each operand independently. Unit steps
    # on these small integer-valued inputs give its exact scalar derivative.
    data=a if which=='a' else b;result=np.empty_like(data);base=float(np.asarray(a@b).sum())
    for index in np.ndindex(data.shape):
        perturbed=data.copy();perturbed[index]+=1
        value=perturbed@b if which=='a' else a@perturbed
        result[index]=float(np.asarray(value).sum())-base
    return result


@pytest.mark.parametrize('shape_a,shape_b,needs,selected,changed',CASES)
def test_matmul_only_guards_operands_needed_by_active_derivatives(shape_a,shape_b,needs,selected,changed):
    data_a=np.arange(np.prod(shape_a),dtype='float64').reshape(shape_a)+1
    data_b=np.arange(np.prod(shape_b),dtype='float64').reshape(shape_b)+1
    a=torch.tensor(data_a,requires_grad=needs[0]);b=torch.tensor(data_b,requires_grad=needs[1]);y=a@b
    with torch.no_grad():{'a':a,'b':b,'output':y}[changed].detach().add_(1)
    guarded=changed=='a' and needs[1] or changed=='b' and needs[0]
    requested=[key for key,flag in zip(['a','b'],needs) if flag] if selected=='both' else [selected]
    inputs=tuple({'a':a,'b':b}[key] for key in requested)
    if guarded:
        with pytest.raises(RuntimeError,match='modified'):torch.autograd.grad(y.sum(),inputs)
    else:
        for key,gradient in zip(requested,torch.autograd.grad(y.sum(),inputs)):
            np.testing.assert_array_equal(gradient.numpy(),exact_linear_gradient(data_a,data_b,key))


@pytest.mark.parametrize('shape',[(2,),(2,2),(2,2,2)])
def test_shared_matmul_operand_keeps_one_correct_storage_guard(shape):
    data=np.arange(np.prod(shape),dtype='float64').reshape(shape)+1;x=torch.tensor(data,requires_grad=True);y=x@x
    expected=exact_linear_gradient(data,data,'a')+exact_linear_gradient(data,data,'b')
    y.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),expected)
    x=torch.tensor(data,requires_grad=True);y=x@x
    with torch.no_grad():x.detach().add_(1)
    with pytest.raises(RuntimeError,match='modified'):y.sum().backward()
