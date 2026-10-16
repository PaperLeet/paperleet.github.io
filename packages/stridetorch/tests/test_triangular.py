"""Independent algebra,constructed large systems and training/checkpoint flows."""
import gc
import io
import weakref
import numpy as np
import pytest
from stridetorch import torch, UnsupportedOperation


def effective(a,upper,unit):
 result=np.zeros_like(a);n=a.shape[-1]
 for row in range(n):
  for col in range(n):
   if row==col:result[...,row,col]=1 if unit else a[...,row,col]
   elif col>row if upper else col<row:result[...,row,col]=a[...,row,col]
 return result


def project(a,upper,unit):
 result=effective(a,upper,False)
 if unit:result[...,np.arange(a.shape[-1]),np.arange(a.shape[-1])]=0
 return result


def inverse2(a):
 d=a[...,0,0]*a[...,1,1]-a[...,0,1]*a[...,1,0];out=np.empty_like(a)
 out[...,0,0]=a[...,1,1];out[...,0,1]=-a[...,0,1];out[...,1,0]=-a[...,1,0];out[...,1,1]=a[...,0,0]
 return out/d[...,None,None]


def unbroadcast(x,shape):
 while x.ndim>len(shape):x=x.sum(0)
 for axis,n in enumerate(shape):
  if n==1 and x.shape[axis]!=1:x=x.sum(axis,keepdims=True)
 return x.reshape(shape)


def inputs(dtype):
 a=np.array([[[[2.,.3],[.4,3.]]],[[[3.,-.2],[.1,2.5]]]],dtype=dtype);b=np.arange(12,dtype=dtype).reshape(1,3,2,2)/7+.2
 return a,b


def close(a,b,dtype='float64'):np.testing.assert_allclose(a,b,rtol=5e-5 if dtype=='float32' else 3e-12,atol=3e-6 if dtype=='float32' else 4e-13)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
def test_closed_form_values_gradients_and_dynamic_seeds(dtype,upper,left,unit):
 a,b=inputs(dtype);inv=inverse2(effective(a,upper,unit));expected=inv@b if left else b@inv;A=torch.tensor(a,requires_grad=True);B=torch.tensor(b,requires_grad=True);x=torch.linalg.solve_triangular(A,B,upper=upper,left=left,unitriangular=unit)
 close(x.numpy(),expected,dtype)
 for magnitude in (.4,-1.3):
  seed=np.full(x.shape,magnitude,dtype=dtype);db=inv.swapaxes(-1,-2)@seed if left else seed@inv.swapaxes(-1,-2);da=project(-(db@expected.swapaxes(-1,-2) if left else expected.swapaxes(-1,-2)@db),upper,unit)
  A.grad=None;B.grad=None;x.backward(torch.tensor(seed),retain_graph=True);close(A.grad.numpy(),unbroadcast(da,a.shape),dtype);close(B.grad.numpy(),unbroadcast(db,b.shape),dtype)


@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
@pytest.mark.parametrize('active',[(0,1),(0,),(1,)])
def test_jvp_and_full_reverse_formula(upper,left,unit,active):
 a,b=inputs('float64');da=np.full_like(a,.13);db=np.full_like(b,.27);args=tuple(torch.tensor(v,requires_grad=True) for v in (a,b,da,db))
 def function(*values):
  actual=list(args[:2])
  for i,v in zip(active,values):actual[i]=v
  return torch.linalg.solve_triangular(*actual,upper=upper,left=left,unitriangular=unit)
 primal,tangent=torch.func.jvp(function,tuple(args[i] for i in active),tuple(args[2+i] for i in active));inv=inverse2(effective(a,upper,unit));x=inv@b if left else b@inv;direction=project(da,upper,unit) if 0 in active else np.zeros_like(a);r=(db if 1 in active else 0)-(direction@x if left else x@direction);expected=inv@r if left else r@inv
 close(primal.numpy(),x);close(tangent.numpy(),expected)
 seed=np.ones_like(x);rs=inv.swapaxes(-1,-2)@seed if left else seed@inv.swapaxes(-1,-2);xs=-(direction.swapaxes(-1,-2)@rs if left else rs@direction.swapaxes(-1,-2));bg=inv.swapaxes(-1,-2)@xs if left else xs@inv.swapaxes(-1,-2)
 ag=project(-(bg@x.swapaxes(-1,-2)+rs@expected.swapaxes(-1,-2) if left else x.swapaxes(-1,-2)@bg+expected.swapaxes(-1,-2)@rs),upper,unit);dag=project(-(rs@x.swapaxes(-1,-2) if left else x.swapaxes(-1,-2)@rs),upper,unit)
 gradients=torch.autograd.grad(tangent.sum(),args,allow_unused=True)
 for gradient,wanted,shape in zip(gradients,(ag,bg,dag if 0 in active else None,rs if 1 in active else None),(a.shape,b.shape,da.shape,db.shape)):
  if wanted is None:assert gradient is None
  else:close(gradient.numpy(),unbroadcast(wanted,shape))


@pytest.mark.parametrize('n',[65,96,129])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
def test_constructed_large_systems_and_adjoint_seeds(n,dtype,upper,left,unit):
 rng=np.random.default_rng(31);a=(rng.normal(size=(n,n))*.02).astype(dtype);a[np.diag_indices(n)]=2.;e=effective(a,upper,unit);shape=(2,n,3) if left else (2,3,n);target=(rng.normal(size=shape)*.2).astype(dtype);z=(rng.normal(size=shape)*.1).astype(dtype);b=e@target if left else target@e;seed=e.T@z if left else z@e.T
 A=torch.tensor(a,requires_grad=True);B=torch.tensor(b,requires_grad=True);x=torch.linalg.solve_triangular(A,B,upper=upper,left=left,unitriangular=unit);close(x.numpy(),target,dtype);x.backward(torch.tensor(seed));close(B.grad.numpy(),z,dtype);expected=project(-(z@target.swapaxes(-1,-2) if left else target.swapaxes(-1,-2)@z),upper,unit);close(A.grad.numpy(),expected.sum(0),dtype)


@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
def test_real_training_and_zip_checkpoint_resume(upper,left,unit):
 a=np.array([[2.,.3],[.4,3.]]);b=np.array([[.5,.7],[.2,.9]]);A=torch.nn.Parameter(torch.tensor(a));B=torch.nn.Parameter(torch.tensor(b));opt=torch.optim.SGD([A,B],lr=.03);sched=torch.optim.lr_scheduler.StepLR(opt,2,gamma=.8)
 for step in range(6):
  inv=inverse2(effective(a,upper,unit));x=inv@b if left else b@inv;seed=2*x/x.size;db=inv.T@seed if left else seed@inv.T;da=project(-(db@x.T if left else x.T@db),upper,unit);rate=.03*.8**(step//2);a=a-rate*da;b=b-rate*db
  opt.zero_grad();torch.linalg.solve_triangular(A,B,upper=upper,left=left,unitriangular=unit).square().mean().backward();opt.step();sched.step();close(A.numpy(),a);close(B.numpy(),b)
  if step==2:
   stream=io.BytesIO();torch.save({'a':A.detach(),'b':B.detach(),'opt':opt.state_dict(),'sched':sched.state_dict()},stream);stream.seek(0);state=torch.load(stream);A=torch.nn.Parameter(state['a']);B=torch.nn.Parameter(state['b']);opt=torch.optim.SGD([A,B],lr=.9);sched=torch.optim.lr_scheduler.StepLR(opt,2);sched.load_state_dict(state['sched']);opt.load_state_dict(state['opt'])


@pytest.mark.parametrize('left',[True,False])
def test_array_retention_and_release(left):
 root=torch.tensor([[2.,.3],[.4,3.]],requires_grad=True);rhs=torch.ones(2,2,requires_grad=True);A=root*2;B=rhs*3;ar=weakref.ref(A._data);br=weakref.ref(B._data);aw=weakref.ref(A);bw=weakref.ref(B);x=torch.linalg.solve_triangular(A,B,upper=True,left=left);del A,B;gc.collect();assert aw() is bw() is None and ar() is not None and br() is None
 x.sum().backward();gc.collect();assert ar() is None


@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
def test_ignored_nan_entries_guards_and_numpy_error_mode(upper,left,unit):
 a=np.array([[2.,.3],[.4,3.]])
 a[1,0] = np.nan if upper else a[1,0]
 a[0,1] = np.nan if not upper else a[0,1]
 if unit:a[np.diag_indices(2)]=np.nan
 expected=inverse2(effective(a,upper,unit));previous=np.seterr(all='raise')
 try:
  A=torch.tensor(a,requires_grad=True);B=torch.ones(2,2,dtype=torch.float64,requires_grad=True);x=torch.linalg.solve_triangular(A,B,upper=upper,left=left,unitriangular=unit);close(x.numpy(),expected@np.ones((2,2)) if left else np.ones((2,2))@expected)
  with torch.no_grad():B.fill_(9.)
  x.sum().backward(retain_graph=True)
  with torch.no_grad():A.detach().fill_(.5)
  with pytest.raises(UnsupportedOperation,match='modified in-place'):x.sum().backward()
  assert np.geterr()=={'divide':'raise','over':'raise','under':'raise','invalid':'raise'}
 finally:np.seterr(**previous)


@pytest.mark.parametrize('shape',[(0,2),(2,0),(0,0),(1,4),(2,1)])
@pytest.mark.parametrize('left',[True,False])
def test_empty_and_singleton_output_layouts(shape,left):
 n=shape[0] if left else shape[1];x=torch.linalg.solve_triangular(torch.eye(n,dtype=torch.float64),torch.ones(shape,dtype=torch.float64),upper=True,left=left);assert x.shape==shape and x.stride()==(1,max(1,shape[0]));close(x.numpy(),np.ones(shape))


def test_boundary_validation_and_capability():
 a=torch.eye(2);b=torch.ones(2,1)
 for kwargs in ({},{'upper':1},{'upper':True,'left':0},{'upper':True,'unitriangular':1}):
  with pytest.raises(TypeError):torch.linalg.solve_triangular(a,b,**kwargs)
 for aa,bb in ((a,torch.ones(2)),(torch.ones(2,3),b),(a,torch.ones(3,1))):
  with pytest.raises(RuntimeError):torch.linalg.solve_triangular(aa,bb,upper=True)
 with pytest.raises(UnsupportedOperation):torch.linalg.solve_triangular(a,b,upper=True,out=b)
 with pytest.raises(UnsupportedOperation):torch.linalg.solve_triangular(a.half(),b.half(),upper=True)
 with torch.no_grad():assert not torch.linalg.solve_triangular(a.requires_grad_(),b,upper=True).requires_grad
 assert torch._browser_capabilities['torch.linalg.solve_triangular'].forward


@pytest.mark.parametrize('dtype',['complex64','complex128'])
@pytest.mark.parametrize('upper',[True,False])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
@pytest.mark.parametrize('n',[2,65])
def test_complex_constructed_systems(dtype,upper,left,unit,n):
 rng=np.random.default_rng(31);a=(rng.normal(size=(n,n))*.02+.01j).astype(dtype);a[np.diag_indices(n)]=2.+.1j;e=effective(a,upper,unit);shape=(n,3) if left else (3,n);target=(rng.normal(size=shape)*.2+.03j).astype(dtype);b=e@target if left else target@e
 actual=torch.linalg.solve_triangular(torch.tensor(a),torch.tensor(b),upper=upper,left=left,unitriangular=unit);close(actual.numpy(),target,'float32' if dtype=='complex64' else 'float64');assert not actual.requires_grad


@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
def test_scalar_matrix_jvp_full_reverse_closed_form(left,unit):
 shape=(1,3) if left else (3,1);A=torch.tensor([[2.]],dtype=torch.float64,requires_grad=True);B=torch.ones(shape,dtype=torch.float64,requires_grad=True);dA=torch.tensor([[.3]],dtype=torch.float64,requires_grad=True);dB=torch.full(shape,.2,dtype=torch.float64,requires_grad=True);primal,tangent=torch.func.jvp(lambda a,b:torch.linalg.solve_triangular(a,b,upper=True,left=left,unitriangular=unit),(A,B),(dA,dB));close(primal.numpy(),np.full(shape,1. if unit else .5));close(tangent.numpy(),np.full(shape,.2 if unit else .025));ga,gb,gda,gdb=torch.autograd.grad(tangent.sum(),(A,B,dA,dB));close(ga.numpy(),[[0. if unit else -.2*3/4+2*.3*3/8]]);close(gb.numpy(),np.full(shape,0. if unit else -.3/4));close(gda.numpy(),[[0. if unit else -3/4]]);close(gdb.numpy(),np.full(shape,1. if unit else .5))


@pytest.mark.parametrize('config',['zero-matrix','zero-rhs','zero-batch'])
@pytest.mark.parametrize('left',[True,False])
@pytest.mark.parametrize('unit',[True,False])
def test_empty_jvp_reverse_has_explicit_zero_edges(config,left,unit):
 n,m=(0,2) if config=='zero-matrix' else (3,0) if config=='zero-rhs' else (2,2);a_shape=(0,n,n) if config=='zero-batch' else (n,n);b_shape=((1,)+(n,m) if left else (1,)+(m,n)) if config=='zero-batch' else (n,m) if left else (m,n);A=torch.tensor(np.broadcast_to(np.eye(n),a_shape).copy(),requires_grad=True);B=torch.ones(b_shape,dtype=torch.float64,requires_grad=True);dA=torch.ones_like(A,requires_grad=True);dB=torch.ones_like(B,requires_grad=True);primal,tangent=torch.func.jvp(lambda a,b:torch.linalg.solve_triangular(a,b,upper=True,left=left,unitriangular=unit),(A,B),(dA,dB));assert primal.numel()==tangent.numel()==0
 for gradient,source in zip(torch.autograd.grad(tangent.sum(),(A,B,dA,dB)),(A,B,dA,dB)):close(gradient.numpy(),np.zeros(source.shape))
