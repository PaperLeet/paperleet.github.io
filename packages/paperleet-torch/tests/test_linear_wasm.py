"""Independent scalar oracles for the WASM linear pullbacks."""
import math

import numpy as np
import pytest

from paperleet_torch import UnsupportedOperation, torch
from paperleet_torch import _nn


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
