"""Independent2x2 algebra, derivatives, continued training and storage ownership."""
import gc
import weakref
import numpy as np
import pytest
from torchwasm import torch, UnsupportedOperation


def inverse2(a):
 determinant=a[...,0,0]*a[...,1,1]-a[...,0,1]*a[...,1,0]
 out=np.empty_like(a);out[...,0,0]=a[...,1,1];out[...,0,1]=-a[...,0,1];out[...,1,0]=-a[...,1,0];out[...,1,1]=a[...,0,0]
 return out/determinant[...,None,None]

def unbroadcast(x,shape):
 while x.ndim>len(shape):x=x.sum(0)
 for axis,n in enumerate(shape):
  if n==1 and x.shape[axis]!=1:x=x.sum(axis,keepdims=True)
 return x.reshape(shape)

def values(dtype):
 a=np.array([[[[3.,.2],[.4,2.]]],[[[2.,-.1],[.3,4.]]]],dtype=dtype)
 b=np.arange(12,dtype=dtype).reshape(1,3,2,2)/7+.2
 return a,b

@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('left',[True,False])
def test_closed_form_values_vjp_and_dynamic_seeds(dtype,left):
 a,b=values(dtype);inv=inverse2(a);expected=inv@b if left else b@inv
 A=torch.tensor(a,requires_grad=True);B=torch.tensor(b,requires_grad=True);x=torch.linalg.solve(A,B,left=left)
 np.testing.assert_allclose(x.numpy(),expected,rtol=2e-6,atol=2e-7)
 for c in (.4,-1.7):
  seed=np.full(x.shape,c,dtype=dtype);db=np.swapaxes(inv,-1,-2)@seed if left else seed@np.swapaxes(inv,-1,-2)
  da=-(db@np.swapaxes(expected,-1,-2) if left else np.swapaxes(expected,-1,-2)@db)
  A.grad=None;B.grad=None;x.backward(torch.tensor(seed),retain_graph=True)
  np.testing.assert_allclose(A.grad.numpy(),unbroadcast(da,a.shape),rtol=3e-6,atol=2e-7)
  np.testing.assert_allclose(B.grad.numpy(),unbroadcast(db,b.shape),rtol=3e-6,atol=2e-7)

@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('active',[(0,1),(0,),(1,)])
def test_jvp_closed_form_and_native_constant_factor_reverse(left,active):
 a,b=values('float64');da=np.full_like(a,.13);db=np.full_like(b,.27)
 A=torch.tensor(a,requires_grad=True);B=torch.tensor(b,requires_grad=True);dA=torch.tensor(da,requires_grad=True);dB=torch.tensor(db,requires_grad=True)
 inputs=(A,B);directions=(dA,dB)
 def call(*args):
  values=list(inputs)
  for i,value in zip(active,args):values[i]=value
  return torch.linalg.solve(*values,left=left)
 primal,tangent=torch.func.jvp(call,tuple(inputs[i] for i in active),tuple(directions[i] for i in active))
 inv=inverse2(a);x=inv@b if left else b@inv
 product=(da@x if left else x@da) if 0 in active else np.zeros_like(x)
 residual=(db if 1 in active else 0)-product;expected=inv@residual if left else residual@inv
 np.testing.assert_allclose(primal.numpy(),x,rtol=2e-12,atol=2e-13);np.testing.assert_allclose(tangent.numpy(),expected,rtol=2e-12,atol=2e-13)
 tangent.sum().backward()
 # Reverse of the JVP holds its outer factorization constant. Only its
 # residual contributes derivatives through X, including materialized zeros.
 residual_seed=np.swapaxes(inv,-1,-2)@np.ones_like(x) if left else np.ones_like(x)@np.swapaxes(inv,-1,-2)
 dx=-(np.swapaxes(da,-1,-2)@residual_seed if left else residual_seed@np.swapaxes(da,-1,-2)) if 0 in active else np.zeros_like(x)
 db_x=np.swapaxes(inv,-1,-2)@dx if left else dx@np.swapaxes(inv,-1,-2)
 da_x=-(db_x@np.swapaxes(x,-1,-2) if left else np.swapaxes(x,-1,-2)@db_x)
 np.testing.assert_allclose(A.grad.numpy(),unbroadcast(da_x,a.shape),rtol=2e-12,atol=2e-13);np.testing.assert_allclose(B.grad.numpy(),unbroadcast(db_x,b.shape),rtol=2e-12,atol=2e-13)

@pytest.mark.parametrize('left',[True,False])
def test_training_updates_checkpoint_continuation_against_closed_form(left):
 a=np.array([[2.5,.2],[.1,3.]]);b=np.array([[1.,2.],[3.,4.]])
 A=torch.nn.Parameter(torch.tensor(a));B=torch.nn.Parameter(torch.tensor(b));optimizer=torch.optim.SGD([A,B],lr=.03)
 for step in range(5):
  inv=inverse2(a);x=inv@b if left else b@inv;g=2*x/x.size
  db=inv.T@g if left else g@inv.T;da=-(db@x.T if left else x.T@db)
  optimizer.zero_grad();loss=torch.linalg.solve(A,B,left=left).square().mean();loss.backward();optimizer.step()
  a=a-.03*da;b=b-.03*db
  np.testing.assert_allclose(A.numpy(),a,rtol=2e-12,atol=2e-13);np.testing.assert_allclose(B.numpy(),b,rtol=2e-12,atol=2e-13)
  if step==2:
   state=optimizer.state_dict();A=torch.nn.Parameter(A.detach().clone());B=torch.nn.Parameter(B.detach().clone());optimizer=torch.optim.SGD([A,B],lr=.9);optimizer.load_state_dict(state)

@pytest.mark.parametrize('dtype',['complex64','complex128'])
@pytest.mark.parametrize('left',[True,False])
def test_complex_inference_does_not_conjugate_system(dtype,left):
 a,b=values(dtype);a=a+.3j;b=b-.2j;inverse=inverse2(a);expected=inverse@b if left else b@inverse
 x=torch.linalg.solve(torch.tensor(a),torch.tensor(b),left=left);np.testing.assert_allclose(x.numpy(),expected,rtol=2e-6,atol=2e-7)
 assert not x.requires_grad


def test_alias_guards_b_mutation_and_no_grad():
 a=torch.tensor([[3.,.2],[.1,2.]],requires_grad=True);b=torch.ones(2,requires_grad=True);x=torch.linalg.solve(a,b)
 with torch.no_grad():b.fill_(9.)
 x.sum().backward(retain_graph=True);np.testing.assert_allclose(b.grad.numpy(),inverse2(a.numpy()).T@np.ones(2))
 with torch.no_grad():x.detach().fill_(2.)
 with pytest.raises(UnsupportedOperation,match='modified in-place'):x.sum().backward()
 with torch.no_grad():assert not torch.linalg.solve(a,b).requires_grad
 assert torch._browser_capabilities['torch.linalg.solve'].forward


def test_intermediate_storage_is_released_and_no_seed_is_cached():
 root=torch.tensor([[2.,.2],[.1,3.]],requires_grad=True);b=torch.ones(2,requires_grad=True);coefficient=root*2;ref=weakref.ref(coefficient._data);wrapper=weakref.ref(coefficient);x=torch.linalg.solve(coefficient,b);del coefficient;gc.collect()
 assert wrapper() is None and ref() is not None
 for scale in (1.,-2.):
  b.grad=None;root.grad=None;x.backward(torch.full((2,),scale),retain_graph=True)
  np.testing.assert_allclose(b.grad.numpy(),inverse2(root.numpy()*2).T@np.full(2,scale),rtol=2e-6)
 x.backward(torch.ones(2));gc.collect();assert ref() is None

@pytest.mark.parametrize('call',[
 lambda:torch.linalg.solve(torch.ones(2,3),torch.ones(3)),
 lambda:torch.linalg.solve(torch.eye(2),torch.ones(2),left=False),
 lambda:torch.linalg.solve(torch.eye(2),torch.ones(2),out=torch.empty(2)),
 lambda:torch.linalg.solve(torch.eye(2,dtype=torch.float16),torch.ones(2,dtype=torch.float16)),
 lambda:torch.linalg.solve(torch.eye(2),torch.ones(2,dtype=torch.float64)),
])
def test_invalid_and_unsupported_boundaries_are_explicit(call):
 with pytest.raises((RuntimeError,UnsupportedOperation)):call()
