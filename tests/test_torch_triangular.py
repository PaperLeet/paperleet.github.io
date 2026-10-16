"""Dense triangular systems against native PyTorch,including reverse of JVP."""
import gc
import weakref
import numpy as np
import pytest
import torch as native
from stridetorch import torch as lite, UnsupportedOperation

CONFIGS=[((),(),2,3),((2,1),(1,3),3,2),((2,),(),4,3),((),(2,),5,2),((2,),(2,),2,2),((),(),0,2),((),(),2,0),((0,),(1,),3,2),((),(),1,4),((),(),2,1)]


def arrays(dtype,abatch,bbatch,n,m,left):
 rng=np.random.default_rng(31);a=(rng.normal(size=abatch+(n*2,n*2))*.1).astype(dtype)[...,::2,::2]
 a[...,np.arange(n),np.arange(n)]+=2
 shape=(n,m) if left else (m,n);b=rng.normal(size=bbatch+tuple(2*d for d in shape)).astype(dtype)[...,::2,::2]
 return a,b


def tensor(lib,array,grad=False):return lib.from_numpy(array.copy() if not array.size else array).requires_grad_(grad)

def close(a,b,dtype):np.testing.assert_allclose(a,b,rtol=5e-5 if dtype=='float32' else 2e-11,atol=3e-6 if dtype=='float32' else 3e-12,equal_nan=True)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
@pytest.mark.parametrize('active',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('config',CONFIGS)
def test_native_values_layout_and_both_gradients(dtype,upper,left,unit,active,config):
 a,b=arrays(dtype,*config,left);la,lb=tensor(lite,a,active[0]),tensor(lite,b,active[1]);na,nb=tensor(native,a,active[0]),tensor(native,b,active[1]);kw=dict(upper=upper,left=left,unitriangular=unit)
 x=lite.linalg.solve_triangular(la,lb,**kw);nx=native.linalg.solve_triangular(na,nb,**kw);close(x.numpy(),nx.detach().numpy(),dtype);assert x.stride()==nx.stride()
 seed=np.linspace(.3,1.2,x.numel(),dtype=dtype).reshape(x.shape);x.backward(lite.tensor(seed));nx.backward(native.tensor(seed))
 for l,n in ((la,na),(lb,nb)):
  if l.requires_grad:
   close(l.grad.numpy(),n.grad.numpy(),dtype)
   # Native accumulation may adopt an internal gradient's singleton strides.
   # Its documented implementation contract compares only non-singleton axes.
   assert all(ls==ns if size!=1 else ls>0 and ns>0 for size,ls,ns in zip(l.shape,l.grad.stride(),n.grad.stride()))
   assert l.grad.is_contiguous()==n.grad.is_contiguous()
  else:assert l.grad is n.grad is None


@pytest.mark.parametrize('n',[65,96,128])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
def test_blocked_values_and_gradients(dtype,n,upper,left,unit):
 a,b=arrays(dtype,(),(2,),n,3,left);kw=dict(upper=upper,left=left,unitriangular=unit);la,lb=tensor(lite,a,True),tensor(lite,b,True);na,nb=tensor(native,a,True),tensor(native,b,True)
 x=lite.linalg.solve_triangular(la,lb,**kw);y=native.linalg.solve_triangular(na,nb,**kw);close(x.numpy(),y.detach().numpy(),dtype);assert x.stride()==y.stride()
 x.square().mean().backward();y.square().mean().backward();close(la.grad.numpy(),na.grad.numpy(),dtype);close(lb.grad.numpy(),nb.grad.numpy(),dtype)


@pytest.mark.parametrize('dtype',['complex64','complex128'])
@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
@pytest.mark.parametrize('n',[2,96])
def test_complex_inference(dtype,upper,left,unit,n):
 a,b=arrays(dtype,(2,),(),n,3,left);a=a+np.asarray(.03j,dtype=dtype);b=b-np.asarray(.1j,dtype=dtype);kw=dict(upper=upper,left=left,unitriangular=unit)
 actual=lite.linalg.solve_triangular(lite.tensor(a),lite.tensor(b),**kw);expected=native.linalg.solve_triangular(native.tensor(a),native.tensor(b),**kw)
 close(actual.numpy(),expected.numpy(),'float32' if dtype=='complex64' else 'float64');assert actual.stride()==expected.stride() and not actual.requires_grad


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
@pytest.mark.parametrize('active',[(0,1),(0,),(1,)])
@pytest.mark.parametrize('batched',[False,True])
def test_jvp_and_reverse_through_all_four_inputs(dtype,upper,left,unit,active,batched):
 a,b=arrays(dtype,(2,1) if batched else (),(1,3) if batched else (),3 if batched else 2,2,left);da=np.full_like(a,.13);db=np.full_like(b,.27);kw=dict(upper=upper,left=left,unitriangular=unit);reports=[]
 for lib in (lite,native):
  args=tuple(tensor(lib,v,True) for v in (a,b,da,db))
  def call(*values):
   inputs=list(args[:2])
   for i,v in zip(active,values):inputs[i]=v
   return lib.linalg.solve_triangular(*inputs,**kw)
  primal,tangent=lib.func.jvp(call,tuple(args[i] for i in active),tuple(args[2+i] for i in active));loss=.17*primal.square().mean()+.23*tangent.square().mean();grads=lib.autograd.grad(loss,args,allow_unused=True)
  reports.append((primal.detach().numpy(),tangent.detach().numpy(),primal.stride(),tangent.stride(),[None if g is None else g.detach().numpy() for g in grads]))
 l,n=reports;close(l[0],n[0],dtype);close(l[1],n[1],dtype);assert l[2:4]==n[2:4]
 for a,b in zip(l[4],n[4]):
  if b is None:assert a is None
  else:close(a,b,dtype)


@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
@pytest.mark.parametrize('active',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('mutated',['a','b','out'])
def test_native_storage_mutation_guards(left,unit,active,mutated):
 statuses=[]
 for lib in (lite,native):
  a=lib.tensor([[2.,.2],[.1,3.]],requires_grad=active[0]);b=lib.ones(2,2,requires_grad=active[1]);out=lib.linalg.solve_triangular(a,b,upper=True,left=left,unitriangular=unit)
  with lib.no_grad():{'a':a,'b':b,'out':out}[mutated].detach().add_(.1)
  try:out.sum().backward();statuses.append(False)
  except (RuntimeError,UnsupportedOperation):statuses.append(True)
 assert statuses[0]==statuses[1] and statuses[0]==(mutated!='b')


@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('unit',[True,False])
@pytest.mark.parametrize('n',[2,7,96])
def test_ignored_half_and_unit_diagonal_nan_values(upper,unit,n):
 a=np.full((n,n),np.nan);indices=np.triu_indices(n,1 if unit else 0) if upper else np.tril_indices(n,-1 if unit else 0);a[indices]=.02
 if not unit:a[np.diag_indices(n)]=2.
 b=np.arange(n*2,dtype=np.float64).reshape(n,2)/n;kw=dict(upper=upper,unitriangular=unit)
 actual=lite.linalg.solve_triangular(lite.tensor(a),lite.tensor(b),**kw);expected=native.linalg.solve_triangular(native.tensor(a),native.tensor(b),**kw);close(actual.numpy(),expected.numpy(),'float64');assert np.isfinite(actual.numpy()).all()


def test_singular_diagonal_propagates_nonfinite_without_generic_solve_error():
 for lib in (lite,native):
  a=lib.tensor([[0.,.2],[0.,2.]],dtype=lib.float64);b=lib.ones(2,1,dtype=lib.float64);x=lib.linalg.solve_triangular(a,b,upper=True)
  assert np.isinf(x.numpy()[0,0]) and x[1,0].item()==.5


def test_invalid_arguments_and_unsupported_modes():
 a=lite.eye(2);b=lite.ones(2,1)
 for kwargs in ({},{'upper':1},{'upper':np.bool_(True)},{'upper':True,'left':1},{'upper':True,'unitriangular':0}):
  with pytest.raises(TypeError):lite.linalg.solve_triangular(a,b,**kwargs)
 for aa,bb in ((lite.ones(2,3),b),(a,lite.ones(2)),(a,lite.ones(3,1)),(a,b.double())):
  with pytest.raises(RuntimeError):lite.linalg.solve_triangular(aa,bb,upper=True)
 with pytest.raises(UnsupportedOperation):lite.linalg.solve_triangular(a,b,upper=True,out=b)
 with pytest.raises(UnsupportedOperation):lite.linalg.solve_triangular(a.half(),b.half(),upper=True)
 with pytest.raises(UnsupportedOperation):lite.linalg.solve_triangular(a.to(dtype=lite.complex64).requires_grad_(),b.to(dtype=lite.complex64),upper=True)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
def test_returned_gradient_layouts(dtype,upper,left,unit):
 a,b=arrays(dtype,(),(),3,4,left);results=[]
 for lib in (lite,native):
  A=tensor(lib,a,True);B=tensor(lib,b,True);x=lib.linalg.solve_triangular(A,B,upper=upper,left=left,unitriangular=unit);da,db=lib.autograd.grad(x.sum(),(A,B));results.append((da.stride(),db.stride()))
 assert results[0]==results[1]


@pytest.mark.parametrize('config',['zero-matrix','scalar-matrix','zero-rhs','zero-batch'])
@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
def test_jvp_empty_and_singleton_reverse_boundaries(config,upper,left,unit):
 n,m=(0,2) if config=='zero-matrix' else (1,3) if config=='scalar-matrix' else (3,0) if config=='zero-rhs' else (2,2);abatch,bbatch=((0,),(1,)) if config=='zero-batch' else ((),());a,b=arrays('float64',abatch,bbatch,n,m,left);da=np.full_like(a,.13);db=np.full_like(b,.27);results=[]
 for lib in (lite,native):
  args=tuple(tensor(lib,v,True) for v in (a,b,da,db));primal,tangent=lib.func.jvp(lambda aa,bb:lib.linalg.solve_triangular(aa,bb,upper=upper,left=left,unitriangular=unit),args[:2],args[2:]);gradients=lib.autograd.grad(primal.sum()*.17+tangent.sum()*.23,args,allow_unused=True);results.append((primal.detach().numpy(),tangent.detach().numpy(),primal.stride(),tangent.stride(),[None if g is None else g.numpy() for g in gradients]))
 l,native_result=results;close(l[0],native_result[0],'float64');close(l[1],native_result[1],'float64');assert l[2:4]==native_result[2:4]
 for actual,wanted in zip(l[4],native_result[4]):
  if wanted is None:assert actual is None
  else:close(actual,wanted,'float64')
