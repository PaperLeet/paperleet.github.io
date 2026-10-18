"""Affine products: independent polynomial derivatives and native edge controls."""
import numpy as np
import pytest
from stridetorch import torch, as_torch, UnsupportedOperation


def reduce_bias(value,shape):
    while value.ndim>len(shape):value=value.sum(axis=0)
    for axis,size in enumerate(shape):
        if size==1 and value.shape[axis]!=1:value=value.sum(axis=axis,keepdims=True)
    return value.reshape(shape)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('matrix,bias_shape',[(True,()),(True,(2,)),(True,(3,1)),(True,(3,2)),(False,()),(False,(3,))])
@pytest.mark.parametrize('alpha,beta',[(1.,1.),(1.5,-.5),(0.,0.)])
def test_affine_product_polynomial_derivatives(dtype,matrix,bias_shape,alpha,beta):
    rng=np.random.default_rng(7819)
    shapes=(bias_shape,(3,4),(4,2) if matrix else (4,))
    values=tuple(np.asarray(rng.integers(-4,5,size=s)/8,dtype=dtype) for s in shapes)
    directions=tuple(np.asarray(rng.integers(-3,4,size=s)/8,dtype=dtype) for s in shapes)
    c,a,b=values;dc,da,db=directions
    fn=lambda c,a,b:getattr(torch,'addmm' if matrix else 'addmv')(c,a,b,alpha=alpha,beta=beta)
    inputs=tuple(torch.from_numpy(x).requires_grad_() for x in values)
    vectors=tuple(torch.from_numpy(x) for x in directions)
    y=alpha*(a@b)+beta*c;dy=alpha*(da@b+a@db)+beta*dc
    seed=np.asarray(rng.integers(-4,5,size=y.shape)/8,dtype=dtype)
    output=fn(*inputs)
    gradients=torch.autograd.grad(output,inputs,torch.from_numpy(seed))
    expected=(beta*reduce_bias(seed,c.shape),alpha*(seed@b.T if matrix else np.outer(seed,b)),alpha*(a.T@seed))
    for actual,target in zip((output,*gradients),(y,*expected)):
        np.testing.assert_allclose(actual.numpy(),target,rtol=1e-6,atol=1e-6)
    for jvp in (torch.func.jvp,torch.autograd.functional.jvp):
        primal,tangent=jvp(fn,inputs,vectors)
        np.testing.assert_allclose(primal.numpy(),y,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(tangent.numpy(),dy,rtol=1e-6,atol=1e-6)
    loss,hessian=torch.autograd.functional.hvp(lambda *x:fn(*x).square().sum()*.5,inputs,vectors)
    ha=alpha*(dy@b.T+y@db.T) if matrix else alpha*(np.outer(dy,b)+np.outer(y,db))
    hb=alpha*(da.T@y+a.T@dy)
    for actual,target in zip(hessian,(beta*reduce_bias(dy,c.shape),ha,hb)):
        np.testing.assert_allclose(actual.numpy(),target,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(loss.numpy(),(y*y).sum()*.5,rtol=1e-6,atol=1e-6)
    assert not any(np.shares_memory(output.numpy(),x) for x in values)


@pytest.mark.parametrize('matrix',[False,True])
def test_zero_beta_ignores_nan_input(matrix):
    a=torch.tensor([[1.,2.],[3.,4.]])
    b=torch.tensor([[2.,0.],[0.,2.]]) if matrix else torch.tensor([2.,2.])
    c=torch.tensor([float('nan')]*2)
    y=getattr(torch,'addmm' if matrix else 'addmv')(c,a,b,beta=0)
    np.testing.assert_array_equal(y.numpy(),a.numpy()@b.numpy())


@pytest.mark.parametrize('matrix',[False,True])
def test_empty_factor_gradient_uses_unchecked_arithmetic_scale(matrix):
    a=torch.zeros((2,0),requires_grad=True)
    b=torch.zeros((0,3) if matrix else (0,),requires_grad=True)
    c=torch.ones((3,) if matrix else (2,),requires_grad=True)
    y=getattr(torch,'addmm' if matrix else 'addmv')(c,a,b,alpha=1e40)
    gradients=torch.autograd.grad(y,(a,b),torch.ones_like(y))
    assert [g.shape for g in gradients]==[a.shape,b.shape]
    assert all(g.numel()==0 for g in gradients)


@pytest.mark.parametrize('matrix',[False,True])
@pytest.mark.parametrize('which',['alpha','beta'])
def test_complex_scale_cannot_deliver_a_real_reverse_gradient(matrix,which):
    a=torch.ones((2,3),requires_grad=which=='alpha')
    b=torch.ones((3,2) if matrix else (3,))
    c=torch.ones((2,),requires_grad=which=='beta')
    y=getattr(torch,'addmm' if matrix else 'addmv')(c,a,b,**{which:1+0j})
    with pytest.raises(RuntimeError,match='complex'):
        torch.autograd.grad(y,a if which=='alpha' else c,torch.ones_like(y))


@pytest.mark.parametrize('matrix',[False,True])
def test_only_saved_opposite_factor_is_guarded(matrix):
    operation=getattr(torch,'addmm' if matrix else 'addmv')
    for target,raises in [('input',False),('self',False),('other',True),('output',False)]:
        a=torch.ones((2,3),requires_grad=True)
        b=torch.ones((3,2) if matrix else (3,))
        c=torch.ones((2,));y=operation(c,a,b,alpha=1.5)
        changed={'input':c,'self':a,'other':b,'output':y}[target]
        with torch.no_grad():changed.detach().add_(1)
        if raises:
            with pytest.raises(UnsupportedOperation,match='modified in-place'):
                torch.autograd.grad(y,a,torch.ones_like(y))
        else:
            gradient=torch.autograd.grad(y,a,torch.ones_like(y))[0]
            np.testing.assert_array_equal(gradient.numpy(),np.full((2,3),3. if matrix else 1.5))


def test_mixed_addmv_tangent_keeps_precision_through_scalar_addition():
    c=torch.tensor([1.,2.],dtype=torch.float64)
    a=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float32)
    b=torch.tensor([1.,2.],dtype=torch.float32)
    fn=lambda c,a,b:torch.addmv(c,a,b)+.125
    primal,tangent=torch.func.jvp(fn,(c,a,b),tuple(torch.ones_like(x) for x in (c,a,b)))
    assert primal.dtype==torch.float32 and tangent.dtype==torch.float64
    np.testing.assert_array_equal(tangent.numpy(),[7.,11.])


@pytest.mark.parametrize('value',[2.,float('inf'),float('nan')])
@pytest.mark.parametrize('divide',[False,True])
def test_direct_scalar_jvp_has_zero_reverse_edge_without_nan(value,divide):
    x=torch.tensor([value],requires_grad=True)
    fn=(lambda x:x/1.25) if divide else (lambda x:x*1.25)
    _,tangent=torch.func.jvp(fn,(x,),(torch.ones_like(x),))
    assert tangent.requires_grad
    np.testing.assert_allclose(tangent.numpy(),[.8 if divide else 1.25])
    gradient=torch.autograd.grad(tangent,x,torch.ones_like(tangent))[0]
    np.testing.assert_array_equal(gradient.numpy(),[0.])


@pytest.mark.parametrize('operation',['softmax','log_softmax'])
@pytest.mark.parametrize('shape,axis',[((0,),0),((2,0),1),((0,3),0)])
def test_empty_normalized_exponential_gradients(operation,shape,axis):
    x=torch.zeros(shape,requires_grad=True)
    fn=lambda x:getattr(x,operation)(axis)
    y=fn(x);assert y.shape==shape
    gradient=torch.autograd.grad(y,x,torch.ones_like(y))[0]
    assert gradient.shape==shape and gradient.numel()==0
    p,t=torch.func.jvp(fn,(x,),(torch.ones_like(x),))
    assert p.shape==t.shape==shape
    with pytest.raises(IndexError):getattr(x,operation)(len(shape))


def test_empty_mean_reverse_jvp_is_zero():
    x=torch.zeros((0,),requires_grad=True)
    p,t=torch.autograd.functional.jvp(lambda x:x.mean(),x,torch.ones_like(x),create_graph=True)
    assert np.isnan(p.item()) and t.item()==0 and t.requires_grad is False


def test_empty_addmv_preserves_native_legacy_bias_shape():
    a=torch.zeros((2,0),requires_grad=True);b=torch.zeros((0,));c=torch.tensor(3.)
    y=torch.addmv(c,a,b,beta=2)
    assert y.shape==() and y.item()==6
    with pytest.raises(RuntimeError):torch.autograd.grad(y,a)
    with pytest.raises(RuntimeError):torch.func.jvp(lambda a:torch.addmv(c,a,b),(a,),(torch.zeros_like(a),))


def test_affine_products_use_the_torch_alias_and_explicit_out_boundary():
    c=torch.ones((2,));a=torch.ones((2,3));b=torch.ones((3,2))
    with as_torch():
        import torch as alias
        assert alias.Tensor is torch.Tensor
        np.testing.assert_array_equal(alias.addmm(c,a,b).numpy(),np.full((2,2),4.))
        np.testing.assert_array_equal(alias.addmv(c,a,torch.ones((3,))).numpy(),[4.,4.])
    np.testing.assert_array_equal(c.addmm(a,b).numpy(),torch.addmm(c,a,b,out=None).numpy())
    with pytest.raises(TypeError):torch.addmm(c,a,b,out=1)
    with pytest.raises(UnsupportedOperation):torch.addmm(c,a,b,out=torch.zeros((2,2)))


@pytest.mark.parametrize('matrix',[False,True])
def test_output_option_values_are_validated_on_every_call(matrix):
    c=torch.ones((2,));a=torch.ones((2,3));b=torch.ones((3,2) if matrix else (3,))
    operation=getattr(torch,'addmm' if matrix else 'addmv')
    for out in (None,1,None,'buffer',torch.zeros((2,2) if matrix else (2,)),None):
        if out is None:assert operation(c,a,b,out=out).shape==((2,2) if matrix else (2,))
        elif isinstance(out,torch.Tensor):
            with pytest.raises(UnsupportedOperation):operation(c,a,b,out=out)
        else:
            with pytest.raises(TypeError):operation(c,a,b,out=out)


def test_empty_addmm_bias_gradient_can_overflow_without_blas_conversion_error():
    c=torch.ones((3,),requires_grad=True);a=torch.zeros((2,0));b=torch.zeros((0,3))
    y=torch.addmm(c,a,b,beta=1e40)
    gradient=torch.autograd.grad(y,c,torch.ones_like(y))[0]
    assert np.isposinf(y.numpy()).all() and np.isposinf(gradient.numpy()).all()


@pytest.mark.parametrize('matrix',[False,True])
def test_half_zero_alpha_preserves_native_nonfinite_product(matrix):
    a=torch.full((2,3),float('nan'),dtype=torch.float16)
    b=torch.ones((3,2) if matrix else (3,),dtype=torch.float16)
    c=torch.zeros((2,),dtype=torch.float16)
    y=getattr(torch,'addmm' if matrix else 'addmv')(c,a,b,alpha=0)
    assert np.isnan(y.numpy()).all()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('matrix',[False,True])
@pytest.mark.parametrize('layout',['C','F','stride'])
def test_large_affine_values_and_derivatives_for_wasm_contractions(dtype,matrix,layout):
    rng=np.random.default_rng(78190)
    values=[np.asarray(rng.integers(-4,5,size=s)/8,dtype=dtype)
            for s in ((32,) if matrix else (64,),(64,64),(64,32) if matrix else (64,))]
    if layout=='F':values[1:]=[np.asfortranarray(x) for x in values[1:]]
    elif layout=='stride':values[1:]=[np.repeat(x,2,axis=-1)[...,::2] for x in values[1:]]
    c,a,b=values;inputs=tuple(torch.from_numpy(x).requires_grad_() for x in values)
    fn=lambda c,a,b:getattr(torch,'addmm' if matrix else 'addmv')(c,a,b,alpha=1.5,beta=-.5)
    expected=1.5*(a@b)-.5*c;y=fn(*inputs)
    np.testing.assert_allclose(y.numpy(),expected,rtol=2e-6,atol=2e-6)
    seed=np.asarray(rng.integers(-4,5,size=y.shape)/8,dtype=dtype)
    gradients=torch.autograd.grad(y,inputs,torch.from_numpy(seed))
    targets=(-.5*reduce_bias(seed,c.shape),1.5*(seed@b.T if matrix else np.outer(seed,b)),1.5*(a.T@seed))
    for actual,target in zip(gradients,targets):np.testing.assert_allclose(actual.numpy(),target,rtol=2e-6,atol=2e-6)
    _,tangent=torch.func.jvp(fn,inputs,tuple(torch.ones_like(x) for x in inputs))
    target=1.5*(np.ones_like(a)@b+a@np.ones_like(b))-.5
    np.testing.assert_allclose(tangent.numpy(),target,rtol=2e-6,atol=2e-6)


@pytest.mark.parametrize('bias_dtype,factor_dtype',[(torch.float64,torch.float32),(torch.float32,torch.float64)])
@pytest.mark.parametrize('stack',[False,True])
@pytest.mark.parametrize('vector_grad',[False,True])
def test_empty_cat_reverse_jvp_ignores_1d_empty_graph(bias_dtype,factor_dtype,stack,vector_grad):
    inputs=(torch.zeros((0,),dtype=bias_dtype),torch.zeros((0,3),dtype=factor_dtype),torch.ones((3,),dtype=factor_dtype))
    vectors=tuple(torch.ones_like(x).requires_grad_(vector_grad) for x in inputs)
    def fn(c,a,b):
        y=torch.addmv(c,a,b)
        return (torch.stack if stack else torch.cat)((y,y))
    primal,tangent=torch.autograd.functional.jvp(fn,inputs,vectors,create_graph=True)
    assert primal.shape==tangent.shape==((2,0) if stack else (0,))
    assert primal.requires_grad and tangent.requires_grad==stack


def test_ignored_empty_cat_accepts_native_integer_axes_and_checks_overflow():
    a=torch.zeros((0,),requires_grad=True);b=torch.zeros((0,),requires_grad=True)
    for axis in (-999,0,1,2**63-1):
        y=torch.cat((a,b),dim=axis)
        assert y.shape==(0,)
        gradients=torch.autograd.grad(y,(a,b),torch.ones_like(y))
        assert all(g.shape==(0,) for g in gradients)
    for axis in (-2**63-1,2**63):
        with pytest.raises(RuntimeError,match='Overflow'):torch.cat((a,b),dim=axis)
