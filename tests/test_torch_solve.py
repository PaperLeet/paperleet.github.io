"""Native linear-system contracts, independent of PaperLeet exercises."""
import gc
import weakref
import numpy as np
import pytest
import torch as native
from torchwasm import torch as lite

CASES = [
 ((1,1),(1,),True), ((2,1,1),(1,),True), ((1,1),(1,1),False),
 ((3,3),(3,),True), ((3,3),(3,2),True), ((3,3),(2,3),False),
 ((2,3,3),(2,3),True), ((2,3,3),(3,),True),
 ((2,3,3),(3,3),True), ((2,3,3),(3,3),False),
 ((2,1,3,3),(1,4,3,2),True), ((1,4,3,3),(2,1,2,3),False),
 ((3,3),(2,4,3,2),True), ((3,3),(2,4,2,3),False),
 ((0,3,3),(3,2),True), ((2,0,0),(0,),True),
 ((2,3,3),(3,0),True), ((2,3,3),(0,3),False),
]

def arrays(ashape,bshape,dtype,layout='C'):
 rng=np.random.default_rng(29);a=rng.normal(size=ashape)+np.eye(ashape[-1])*5;b=rng.normal(size=bshape)
 if np.dtype(dtype).kind=='c':a=a+.2j*rng.normal(size=ashape);b=b+.3j*rng.normal(size=bshape)
 def storage(x):
  x=np.asarray(x,dtype=dtype)
  if layout=='F':return np.asfortranarray(x)
  if layout=='gap':
   base=np.empty(x.shape[:-1]+(x.shape[-1]*2,),dtype=x.dtype);base[...,::2]=x;return base[...,::2]
  return x
 return storage(a),storage(b)

def tensor(lib,x,grad=False):return lib.tensor(x,dtype=getattr(lib,str(x.dtype)),requires_grad=grad)

def close(a,b,dtype):np.testing.assert_allclose(a,b,rtol=2e-5 if dtype in ('float32','complex64') else 2e-11,atol=2e-6 if dtype in ('float32','complex64') else 2e-12)

@pytest.mark.parametrize('ashape,bshape,left',CASES)
@pytest.mark.parametrize('dtype',['float32','float64','complex64','complex128'])
@pytest.mark.parametrize('layout',['C','F','gap'])
def test_values_shapes_and_native_output_strides(ashape,bshape,left,dtype,layout):
 a,b=arrays(ashape,bshape,dtype,layout);x=lite.linalg.solve(tensor(lite,a),tensor(lite,b),left=left);n=native.linalg.solve(tensor(native,a),tensor(native,b),left=left)
 close(x.numpy(),n.resolve_conj().numpy(),dtype);assert x.shape==tuple(n.shape) and x.dtype==a.dtype
 # Include conventional native strides even for empty/singleton outputs.
 assert x.stride()==n.stride()

@pytest.mark.parametrize('ashape,bshape,left',CASES)
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('needs',[(True,True),(True,False),(False,True)])
def test_broadcast_vjps_repeated_dynamic_seeds(ashape,bshape,left,dtype,needs):
 a,b=arrays(ashape,bshape,dtype,'gap');inputs=[(tensor(lib,a,needs[0]),tensor(lib,b,needs[1])) for lib in (lite,native)]
 outputs=[lib.linalg.solve(*pair,left=left) for lib,pair in zip((lite,native),inputs)]
 for factor in (1.,-2.3):
  seed=(np.arange(outputs[0].numel(),dtype=dtype).reshape(outputs[0].shape)/11+factor).astype(dtype)
  for lib,pair,x in zip((lite,native),inputs,outputs):
   for t in pair:t.grad=None
   x.backward(tensor(lib,seed),retain_graph=True)
  for i,active in enumerate(needs):
   if active:close(inputs[0][i].grad.numpy(),inputs[1][i].grad.numpy(),dtype)
   else:assert inputs[0][i].grad is None

@pytest.mark.parametrize('ashape,bshape,left',CASES[:14])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('active',[(0,1),(0,),(1,)])
def test_jvp_and_reverse_of_tangent(ashape,bshape,left,dtype,active):
 a,b=arrays(ashape,bshape,dtype,'F');results=[]
 for lib in (lite,native):
  inputs=(tensor(lib,a,True),tensor(lib,b,True));tangents=(tensor(lib,np.ones_like(a)*.13,True),tensor(lib,np.ones_like(b)*.27,True))
  primals=tuple(inputs[i] for i in active);directions=tuple(tangents[i] for i in active)
  def call(*args):
   values=list(inputs)
   for i,value in zip(active,args):values[i]=value
   return lib.linalg.solve(*values,left=left)
  p,t=lib.func.jvp(call,primals,directions);t.square().sum().backward();results.append([p.detach().numpy(),t.detach().numpy()]+[None if x.grad is None else x.grad.numpy() for x in inputs+tuple(tangents[i] for i in active)])
 for x,y in zip(*results):
  if y is None:assert x is None
  else:close(x,y,dtype)

@pytest.mark.parametrize('needs',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('mutated',['a','b','x'])
def test_native_saved_value_mutation_policy(needs,mutated):
 outcomes=[]
 for lib in (lite,native):
  a=tensor(lib,np.array([[3.,1.],[.2,2.]]),needs[0]);b=tensor(lib,np.ones(2),needs[1]);x=lib.linalg.solve(a,b)
  with lib.no_grad():{'a':a,'b':b,'x':x}[mutated].detach().add_(1)
  try:x.sum().backward();outcomes.append('allowed')
  except RuntimeError:outcomes.append('guarded')
 assert outcomes==(['allowed']*2 if mutated=='b' else ['guarded']*2)

@pytest.mark.parametrize('ashape,bshape,left',[((3,), (3,), True),((2,3),(3,),True),((3,3),(),True),((2,3,3),(1,3),True),((3,3),(3,),False),((2,3,3),(2,3),False),((2,3,3),(4,3,2),True)])
def test_invalid_geometry_matches_native(ashape,bshape,left):
 for lib in (lite,native):
  with pytest.raises(RuntimeError):lib.linalg.solve(lib.ones(ashape),lib.ones(bshape),left=left)

@pytest.mark.parametrize('left',[True,False])
def test_singular_empty_rhs_still_factors_matrix(left):
 for lib in (lite,native):
  with pytest.raises(RuntimeError):lib.linalg.solve(lib.zeros(2,2),lib.empty((2,0) if left else (0,2)),left=left)

def test_single_shared_adjoint_solve_and_graph_release(monkeypatch):
 from torchwasm import _linalg
 original=_linalg.np.linalg.solve;calls=[]
 def counted(*args,**kw):calls.append(1);return original(*args,**kw)
 monkeypatch.setattr(_linalg.np.linalg,'solve',counted)
 root=lite.tensor([[2.,.1],[.3,3.]],dtype=lite.float64,requires_grad=True);a=root*2;b=lite.ones(2,dtype=lite.float64,requires_grad=True);storage=weakref.ref(a._data);wrapper=weakref.ref(a)
 x=lite.linalg.solve(a,b);del a;gc.collect();assert wrapper() is None and storage() is not None
 assert len(calls)==1
 for retain in (True,False):
  root.grad=None;b.grad=None;x.backward(lite.tensor([1.,2.],dtype=lite.float64),retain_graph=retain)
 assert len(calls)==3;gc.collect();assert storage() is None
 with pytest.raises(RuntimeError):x.sum().backward()

def test_native_keywords_and_type_errors():
 for lib in (lite,native):
  a=lib.eye(2);b=lib.ones(2);assert lib.linalg.solve(A=a,B=b,left=True).shape==(2,)
  for invalid in (1,None,np.bool_(True)):
   with pytest.raises(TypeError):lib.linalg.solve(a,b,left=invalid)
  with pytest.raises(TypeError):lib.linalg.solve([[1.]],b)
  with pytest.raises(TypeError):lib.linalg.solve(a,b,True)
