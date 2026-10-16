"""Cholesky contracts against native Torch, including reverse through JVP.

Reference: https://docs.pytorch.org/docs/2.14/generated/torch.linalg.cholesky.html
"""
import numpy as np
import pytest
import torch as native
from stridetorch import torch as lite, UnsupportedOperation


def matrix(shape, dtype):
    rng = np.random.default_rng(32)
    data = rng.normal(size=shape)
    if np.dtype(dtype).kind == 'c':
        data = data + 1j * rng.normal(size=shape)
    data = data.astype(dtype)
    return data @ data.conj().swapaxes(-1, -2) + np.eye(shape[-1], dtype=dtype)


def close(actual, expected, dtype):
    single = np.dtype(dtype).itemsize == (8 if np.dtype(dtype).kind == 'c' else 4)
    np.testing.assert_allclose(actual, expected, rtol=8e-5 if single else 4e-11,
                               atol=5e-6 if single else 4e-12, equal_nan=True)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('shape', [(0,0),(1,1),(2,2),(7,7),(2,3,3),(2,1,4,4),(0,3,3),(2,0,0),(2,1,1),(2,65,65)])
@pytest.mark.parametrize('strided', [False, True])
def test_values_strides_vjp_and_repeated_dynamic_seed(dtype, upper, shape, strided):
    data = matrix(shape, dtype)
    if strided:
        backing = np.zeros(shape[:-2] + (shape[-1]*2, shape[-1]*2), dtype=dtype)
        backing[...,::2,::2] = data
        data = backing[...,::2,::2]
    a = lite.from_numpy(data).requires_grad_()
    b = native.from_numpy(data).requires_grad_()
    x = lite.linalg.cholesky(a, upper=upper)
    y = native.linalg.cholesky(b, upper=upper)
    close(x.numpy(), y.detach().numpy(), dtype)
    assert x.stride() == y.stride()
    for scale in (.3, -1.2):
        seed = np.linspace(.2,1.1,x.numel(),dtype=dtype).reshape(shape) * scale
        a.grad = b.grad = None
        x.backward(lite.tensor(seed), retain_graph=True)
        y.backward(native.tensor(seed), retain_graph=True)
        close(a.grad.numpy(), b.grad.numpy(), dtype)
        assert a.grad.is_contiguous() == b.grad.is_contiguous()


@pytest.mark.parametrize('dtype', ['complex64','complex128'])
@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('shape', [(0,0),(1,1),(3,3),(2,4,4),(0,3,3)])
def test_complex_inference(dtype, upper, shape):
    a = matrix(shape, dtype)
    x = lite.linalg.cholesky(lite.tensor(a), upper=upper)
    y = native.linalg.cholesky(native.tensor(a), upper=upper)
    close(x.numpy(),y.numpy(),dtype)
    assert x.stride() == y.stride()
    with pytest.raises(UnsupportedOperation, match='Complex autograd'):
        lite.linalg.cholesky(lite.tensor(a,requires_grad=True),upper=upper)
    with lite.no_grad():
        close(lite.linalg.cholesky(lite.tensor(a),upper=upper).numpy(),y.numpy(),dtype)


@pytest.mark.parametrize('dtype', ['float32','float64'])
@pytest.mark.parametrize('upper', [False,True])
@pytest.mark.parametrize('shape', [(1,1),(2,2),(2,3,3),(2,0,0)])
@pytest.mark.parametrize('symmetric', [False,True])
def test_jvp_and_reverse_through_primal_and_direction(dtype,upper,shape,symmetric):
    data = matrix(shape,dtype)
    direction = np.random.default_rng(320).normal(size=shape).astype(dtype) * .2
    if symmetric:
        direction = (direction + direction.swapaxes(-1,-2)) * .5
    reports=[]
    for lib in (lite,native):
        a,d = (lib.tensor(v,requires_grad=True) for v in (data,direction))
        x,t = lib.func.jvp(lambda v:lib.linalg.cholesky(v,upper=upper),(a,),(d,))
        # Sum handles empty factors and gives both the primal and direction
        # real reverse edges even when the tangent contains no elements.
        loss = .17*x.square().sum()+.23*t.square().sum()
        gradients=lib.autograd.grad(loss,(a,d))
        reports.append((x.detach().numpy(),t.detach().numpy(),x.stride(),t.stride(),[v.detach().numpy() for v in gradients]))
    x,y=reports
    close(x[0],y[0],dtype);close(x[1],y[1],dtype)
    assert x[2:4] == y[2:4]
    for a,b in zip(x[4],y[4]):close(a,b,dtype)


@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('mutated',['a','factor'])
def test_native_mutation_guards(upper,mutated):
    outcomes=[]
    for lib in (lite,native):
        a=lib.tensor([[4.,1.],[1.,3.]],requires_grad=True)
        factor=lib.linalg.cholesky(a,upper=upper)
        with lib.no_grad():
            (a if mutated=='a' else factor).detach().add_(.1)
        try:
            factor.sum().backward();outcomes.append(False)
        except RuntimeError:
            outcomes.append(True)
    assert outcomes == [mutated=='factor']*2


@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('kind',['unused-nan','used-nan','diagonal-nan','infinite-diagonal','negative','zero','batched-negative'])
def test_selected_triangle_and_failures(upper,kind):
    a=matrix((2,3,3) if kind=='batched-negative' else (3,3),'float64')
    row,col=(0,1) if upper else (1,0)
    if kind=='unused-nan':a[col,row]=np.nan
    if kind=='used-nan':a[row,col]=np.nan
    if kind=='diagonal-nan':a[0,0]=np.nan
    if kind=='infinite-diagonal':a[0,0]=np.inf
    if kind=='negative':a[0,0]=-1
    if kind=='zero':a[0,0]=0
    if kind=='batched-negative':a[1,1,1]=-1
    if kind in ('used-nan','diagonal-nan','negative','zero','batched-negative'):
        # Native linalg deliberately makes no cross-backend guarantees for
        # nonfinite inputs. The adapter rejects NaN factors consistently;
        # finite non-positive-definite cases still compare both libraries.
        libraries=(lite,) if kind in ('used-nan','diagonal-nan') else (lite,native)
        for lib in libraries:
            with pytest.raises(RuntimeError,match='positive-definite'):
                lib.linalg.cholesky(lib.tensor(a),upper=upper)
    else:
        x=lite.tensor(a,requires_grad=True);y=native.tensor(a,requires_grad=True)
        lx=lite.linalg.cholesky(x,upper=upper);ly=native.linalg.cholesky(y,upper=upper)
        close(lx.numpy(),ly.detach().numpy(),'float64')
        lx.sum().backward();ly.sum().backward()
        close(x.grad.numpy(),y.grad.numpy(),'float64')
