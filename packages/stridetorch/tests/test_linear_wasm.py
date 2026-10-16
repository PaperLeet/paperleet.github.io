"""Independent scalar oracles for the WASM linear pullbacks."""
import math

import numpy as np
import pytest

from stridetorch import UnsupportedOperation, torch
from stridetorch import _nn


@pytest.fixture(autouse=True)
def wasm_linear_kernel(monkeypatch):
    monkeypatch.setattr(_nn, '_WASM_LINEAR_GRADS', True)


def scalar_oracles(x, weight, bias, seed):
    rows = math.prod(x.shape[:-1])
    features, outputs = weight.shape[1], weight.shape[0]
    source, incoming = x.reshape(rows, features), seed.reshape(rows, outputs)
    result = np.empty((rows, outputs), dtype=np.float64)
    dx, dw, db = np.zeros_like(source, dtype=np.float64), np.zeros_like(weight, dtype=np.float64), np.zeros(outputs)
    for row in range(rows):
        for output in range(outputs):
            result[row, output] = math.fsum(float(source[row, feature])*float(weight[output, feature]) for feature in range(features)) + float(bias[output])
        for feature in range(features):
            dx[row, feature] = math.fsum(float(incoming[row, output])*float(weight[output, feature]) for output in range(outputs))
    for output in range(outputs):
        db[output] = math.fsum(float(incoming[row, output]) for row in range(rows))
        for feature in range(features):
            dw[output, feature] = math.fsum(float(incoming[row, output])*float(source[row, feature]) for row in range(rows))
    return result.reshape(x.shape[:-1]+(outputs,)), (dx.reshape(x.shape), dw, db)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape,outputs', [((4,),3), ((5,4),3), ((3,2,4),3), ((0,4),3), ((2,0),3), ((2,4),0)])
@pytest.mark.parametrize('active', [(True,True,True), (False,True,False), (True,False,True), (False,False,True)])
def test_strided_linear_values_and_two_backward_seeds_against_scalar_sums(dtype, shape, outputs, active):
    rng = np.random.default_rng(814)
    x = rng.normal(size=shape[:-1]+(shape[-1]*2,)).astype(dtype)[...,::2]
    weight = rng.normal(size=(outputs,shape[-1]*2)).astype(dtype)[:,::2]
    bias = rng.normal(size=(outputs,)).astype(dtype)
    seed = rng.normal(size=shape[:-1]+(outputs*2,)).astype(dtype)[...,::2]
    tensors = [torch.from_numpy(value).requires_grad_(enabled) for value,enabled in zip((x,weight,bias),active)]
    output = torch.nn.functional.linear(*tensors)
    expected, first = scalar_oracles(x,weight,bias,seed)
    tolerance = 4e-6 if dtype == torch.float32 else 1e-12
    assert output.shape == expected.shape and output.dtype == dtype
    np.testing.assert_allclose(output.numpy(),expected,rtol=tolerance,atol=tolerance)
    output.backward(torch.from_numpy(seed),retain_graph=True)
    for value,enabled,derivative in zip(tensors,active,first):
        if enabled: np.testing.assert_allclose(value.grad.numpy(),derivative,rtol=tolerance,atol=tolerance)
        else: assert value.grad is None
    second_seed = (seed * -.3).astype(dtype)
    _, second = scalar_oracles(x,weight,bias,second_seed)
    output.backward(torch.from_numpy(second_seed))
    for value,enabled,a,b in zip(tensors,active,first,second):
        if enabled: np.testing.assert_allclose(value.grad.numpy(),a+b,rtol=tolerance,atol=tolerance)
        else: assert value.grad is None


def test_linear_wasm_pullbacks_pass_full_jacobian_and_reverse_over_forward():
    x = torch.tensor([[.2,-.4,.1],[.3,.8,-.5]],dtype=torch.float64,requires_grad=True)
    w = torch.tensor([[.7,.2,-.3],[-.1,.4,.9]],dtype=torch.float64,requires_grad=True)
    b = torch.tensor([.1,-.2],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(torch.nn.functional.linear,(x,w,b))
    direction = torch.tensor([[.4,-.2,.3],[.7,.1,-.6]],dtype=torch.float64)
    primal,tangent = torch.func.jvp(lambda value:torch.nn.functional.linear(value,w,b),(x,),(direction,))
    _, expected = scalar_oracles(direction.numpy(),w.numpy(),np.zeros(2),2*tangent.numpy())
    tangent.square().sum().backward()
    np.testing.assert_allclose(w.grad.numpy(),expected[1],rtol=1e-12,atol=1e-12)
    assert primal.shape == tangent.shape == (2,2)


def test_linear_wasm_keeps_saved_weight_mutation_guard():
    x = torch.ones(2,3,requires_grad=True)
    weight = torch.ones(4,3,requires_grad=True)
    value = torch.nn.functional.linear(x,weight)
    with torch.no_grad(): weight.add_(1.)
    with pytest.raises(UnsupportedOperation,match='modified in-place'): value.sum().backward()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape,outputs', [((2,4,128),32), ((128,),256)])
@pytest.mark.parametrize('layout', ['contiguous', 'feature_stride', 'fortran'])
def test_large_linear_values_retained_gradients_and_frozen_inputs(dtype, shape, outputs, layout):
    rng = np.random.default_rng(1519)
    x = rng.normal(size=shape).astype(dtype)
    w = rng.normal(size=(outputs,shape[-1])).astype(dtype)
    if layout == 'feature_stride':
        x = np.repeat(x,2,axis=-1)[...,::2]
        w = np.repeat(w,2,axis=-1)[:,::2]
    elif layout == 'fortran':
        x, w = np.asfortranarray(x), np.asfortranarray(w)
    bias = rng.normal(size=outputs).astype(dtype)
    seed = rng.normal(size=shape[:-1]+(outputs,)).astype(dtype)
    before = [value.copy() for value in (x,w,bias)]
    # The input is frozen while weight and bias train. A retained graph must
    # compute each new incoming seed without caching a previous derivative.
    a = torch.from_numpy(x)
    weight = torch.from_numpy(w).requires_grad_()
    offset = torch.from_numpy(bias).requires_grad_()
    value = torch.nn.functional.linear(a,weight,offset)
    expected, first = scalar_oracles(x,w,bias,seed)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    np.testing.assert_allclose(value.numpy(),expected,rtol=tolerance,atol=tolerance)
    value.backward(torch.from_numpy(seed),retain_graph=True)
    np.testing.assert_allclose(weight.grad.numpy(),first[1],rtol=tolerance,atol=tolerance)
    np.testing.assert_allclose(offset.grad.numpy(),first[2],rtol=tolerance,atol=tolerance)
    second_seed = (seed * -.3).astype(dtype)
    _, second = scalar_oracles(x,w,bias,second_seed)
    value.backward(torch.from_numpy(second_seed))
    np.testing.assert_allclose(weight.grad.numpy(),first[1]+second[1],rtol=tolerance,atol=tolerance)
    np.testing.assert_allclose(offset.grad.numpy(),first[2]+second[2],rtol=tolerance,atol=tolerance)
    assert a.grad is None
    for actual, original in zip((x,w,bias),before): np.testing.assert_array_equal(actual,original)
    with pytest.raises(RuntimeError,match='second time'): value.backward(torch.from_numpy(seed))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_large_linear_reverse_over_forward_against_independent_scalar_sums(dtype):
    rng = np.random.default_rng(1520)
    xraw = rng.normal(size=(2,4,128)).astype(dtype)
    direction = rng.normal(size=xraw.shape).astype(dtype)
    wraw = rng.normal(size=(32,128)).astype(dtype)
    x = torch.tensor(xraw,requires_grad=True)
    tangent = torch.tensor(direction,requires_grad=True)
    weight = torch.tensor(wraw,requires_grad=True)
    primal, derivative = torch.func.jvp(lambda value:torch.nn.functional.linear(value,weight),(x,),(tangent,))
    expected, _ = scalar_oracles(xraw,wraw,np.zeros(32),np.zeros((2,4,32)))
    tangent_expected, _ = scalar_oracles(direction,wraw,np.zeros(32),np.zeros((2,4,32)))
    seed = rng.normal(size=(2,4,32)).astype(dtype)
    _, gradients = scalar_oracles(direction,wraw,np.zeros(32),seed)
    derivative.backward(torch.tensor(seed))
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    for actual, reference in [(primal,expected),(derivative,tangent_expected),(tangent.grad,gradients[0]),(weight.grad,gradients[1])]:
        np.testing.assert_allclose(actual.numpy(),reference,rtol=tolerance,atol=tolerance)
    if x.grad is not None: np.testing.assert_array_equal(x.grad.numpy(),np.zeros_like(xraw))


@pytest.mark.parametrize('changed', ['input','weight','output'])
def test_large_linear_keeps_all_saved_storage_version_guards(changed):
    x = torch.ones(8,128,requires_grad=True)
    weight = torch.ones(32,128,requires_grad=True)
    value = torch.nn.functional.linear(x,weight)
    with torch.no_grad(): {'input':x,'weight':weight,'output':value}[changed].detach().add_(1.)
    with pytest.raises(UnsupportedOperation,match='modified in-place'): value.sum().backward()
