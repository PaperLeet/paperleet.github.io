from types import SimpleNamespace
import gc
import numpy as np
import pytest
import torch as native
from torchwasm import torch
import torch_bridge as core
from resident_bridge import ResidentMatmulOffload


class Lease:
 def __init__(self,handle,data):self.handle=handle;self.data=np.array(data,copy=True);self.shape=self.data.shape;self.closed=False


class Backend:
 def __init__(self):self.live={};self.next=1;self.uploads=0;self.programs=0;self.transfers=[];self.callback=None;self.fail=None
 def event(self,kind):
  if self.callback:self.callback(kind)
  if self.fail==kind:raise RuntimeError('injected '+kind+' failure')
 def lease(self,data):
  value=Lease(self.next,data);self.next+=1;self.live[value.handle]=value;return value
 def upload(self,value):
  self.event('upload');self.uploads+=1;self.transfers.append(value.nbytes);return self.lease(value)
 def program(self,jobs,resolved,retain_outputs,retain_inputs):
  created=[];incoming={}
  try:
   for key in retain_inputs:
    self.event('upload');self.uploads+=1;self.transfers.append(resolved[key].nbytes);lease=self.lease(resolved[key]);created.append(lease);incoming[key]=lease
   self.event('program');self.programs+=1;arrays=[];matrices=[]
   for (a,b,ta,tb),keep in zip(jobs,retain_outputs):
    aa,bb=resolved[id(a)],resolved[id(b)]
    for value in [aa,bb]:
     if isinstance(value,Lease):assert value.handle in self.live and not value.closed
    aa=aa.data if isinstance(aa,Lease) else aa;bb=bb.data if isinstance(bb,Lease) else bb
    result=(aa.T if ta else aa)@(bb.T if tb else bb);arrays.append(result.copy());matrix=self.lease(result) if keep else None
    if matrix is not None:created.append(matrix)
    matrices.append(matrix)
   return SimpleNamespace(arrays=arrays,matrices=matrices,inputs=incoming)
  except BaseException:
   for matrix in created:self.live.pop(matrix.handle,None);matrix.closed=True
   raise
 def release(self,matrices):
  self.event('release')
  assert all(value.handle in self.live and not value.closed for value in matrices)
  for value in matrices:self.live.pop(value.handle);value.closed=True


@pytest.fixture(autouse=True)
def gate():
 assert not core._busy
 with torch.enable_grad():yield
 assert not core._busy


def data():
 return (np.arange(12,dtype='float32').reshape(3,4)/8.,np.arange(20,dtype='float32').reshape(4,5)/16.+.25,np.arange(10,dtype='float32').reshape(5,2)/8.-.25)


def test_repeated_values_reuse_buffers_but_still_execute_kernels():
 backend=Backend();a,b,_=data();a=torch.from_numpy(a).requires_grad_();b=torch.from_numpy(b).requires_grad_()
 with ResidentMatmulOffload(backend=backend) as call:
  y=call(a,b);g=torch.ones_like(y);torch.autograd.grad(y,(a,b),g);first=call.cache_info()
  y=call(a,b);torch.autograd.grad(y,(a,b),g);second=call.cache_info()
  assert first['programs']==2 and second['programs']==4
  assert second['uploads']==first['uploads'] and second['cache_hits']>first['cache_hits']
  assert second['entries']<=8 and second['resident_bytes']<=8*1024*1024
 assert not backend.live and call.cache_info()['owned_leases']==0


@pytest.mark.parametrize('layout',['C','F','slice'])
@pytest.mark.parametrize('activity',['none','left','right','both'])
def test_native_two_operator_chain_value_and_actual_autograd(layout,activity):
 values=data();arrays=[]
 for value in values:
  if layout=='F':value=np.array(value,order='F')
  if layout=='slice':
   carrier=np.empty((value.shape[0],value.shape[1]*2),dtype='float32');carrier[:,::2]=value;value=carrier[:,::2]
  arrays.append(value)
 results=[];backend=Backend()
 for api in [torch,native]:
  a,b,c=[api.from_numpy(value) for value in arrays];a.requires_grad_(activity in ['left','both']);b.requires_grad_(activity in ['right','both']);c.requires_grad_(activity=='both')
  targets=tuple(value for value in [a,b,c] if value.requires_grad)
  if api is torch:
   with ResidentMatmulOffload(backend=backend) as call:y=call(call(a,b),c);loss=y.sin().square().sum();grads=api.autograd.grad(loss,targets) if targets else ()
  else:y=(a@b)@c;loss=y.sin().square().sum();grads=api.autograd.grad(loss,targets) if targets else ()
  results.append([value.detach().numpy().copy() for value in [y,*grads]])
 for actual,expected in zip(*results):np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
 assert not backend.live


def test_forward_result_and_direct_upstream_gradient_avoid_reupload():
 backend=Backend();a,b,c=[torch.from_numpy(value).requires_grad_() for value in data()]
 with ResidentMatmulOffload(backend=backend) as call:
  y=call(a,b);first=call.cache_info();z=call(y,c);second=call.cache_info()
  assert second['uploads']==first['uploads']+1
  before=call.cache_info();torch.autograd.grad(z,(a,b,c),torch.ones_like(z));after=call.cache_info()
  # One final seed upload; the gradient of y was retained by the second node.
  assert after['uploads']==before['uploads']+1
  assert after['retained_results']>=3 and after['programs']==4
 assert not backend.live


@pytest.mark.parametrize('kind',['tracked','raw'])
def test_cache_updates_follow_current_tensor_values(kind):
 backend=Backend();a,b,_=data();a=torch.from_numpy(a).requires_grad_();b=torch.from_numpy(b)
 with ResidentMatmulOffload(backend=backend) as call:
  old=call(a,b).detach().numpy().copy()
  if kind=='tracked':
   with torch.no_grad():b.add_(.25)
  else:b.numpy()[:]+=.25
  new=call(a,b);np.testing.assert_allclose(new.numpy(),a.numpy()@b.numpy());assert not np.array_equal(old,new.numpy())
  if kind=='raw':
   b.numpy()[:]+=.25;da,=torch.autograd.grad(new,a,torch.ones_like(new));np.testing.assert_allclose(da.numpy(),np.ones(new.shape,dtype='float32')@b.numpy().T)
 assert not backend.live


@pytest.mark.parametrize('entries,bytes_limit',[(0,0),(1,40),(2,128),(8,128)])
def test_tiny_cache_never_evicts_pinned_inputs_and_uses_inline_descriptors(entries,bytes_limit):
 backend=Backend();a,b,_=[torch.from_numpy(value).requires_grad_() for value in data()]
 with ResidentMatmulOffload(max_entries=entries,max_bytes=bytes_limit,backend=backend) as call:
  y=call(a,b);torch.autograd.grad(y,(a,b),torch.ones_like(y));info=call.cache_info()
  assert info['entries']<=entries and info['resident_bytes']<=bytes_limit
  assert info['inline_upload_bytes']>0
 assert not backend.live


def test_close_is_idempotent_and_existing_graphs_finish_on_cpu():
 backend=Backend();a,b,_=[torch.from_numpy(value).requires_grad_() for value in data()];call=ResidentMatmulOffload(backend=backend);y=call(a,b);programs=backend.programs;call.close();call.close()
 assert not backend.live
 da,db=torch.autograd.grad(y,(a,b),torch.ones_like(y));np.testing.assert_allclose(da.numpy(),np.ones(y.shape,dtype='float32')@b.numpy().T);assert backend.programs==programs
 with pytest.raises(RuntimeError,match='closed'):call(a,b)


@pytest.mark.parametrize('phase',['upload','program'])
@pytest.mark.parametrize('kind',['tracked','raw','mode'])
def test_wait_state_changes_rollback_new_leases(phase,kind):
 backend=Backend();a,b,_=[torch.from_numpy(value).requires_grad_() for value in data()];call=ResidentMatmulOffload(backend=backend);seen=[]
 def callback(current):
  if current!=phase or seen:return
  seen.append(True)
  if kind=='tracked':
   with torch.no_grad():a.add_(1.)
  elif kind=='raw':a.detach().numpy()[0,0]+=1.
  else:torch.set_grad_enabled(False)
 backend.callback=callback
 try:
  with pytest.raises(RuntimeError,match='changed'):call(a,b)
 finally:torch.set_grad_enabled(True)
 assert not backend.live and call.cache_info()['owned_leases']==0
 backend.callback=None;call.close()


@pytest.mark.parametrize('phase',['upload','program'])
def test_transport_failure_propagates_then_allows_recovery(phase):
 backend=Backend();backend.fail=phase;call=ResidentMatmulOffload(backend=backend);a,b,_=[torch.from_numpy(value).requires_grad_() for value in data()]
 with pytest.raises(RuntimeError,match='injected'):call(a,b)
 assert not backend.live and not core._busy
 backend.fail=None;y=call(a,b);torch.autograd.grad(y,(a,b),torch.ones_like(y));call.close();assert not backend.live


def test_failed_cleanup_retains_ownership_and_close_can_retry():
 backend=Backend();call=ResidentMatmulOffload(backend=backend);a,b,_=[torch.from_numpy(value).requires_grad_() for value in data()];call(a,b);backend.fail='release'
 with pytest.raises(RuntimeError,match='injected'):call.close()
 assert call.cache_info()['failed'] and call.cache_info()['owned_leases']==len(backend.live)>0
 with pytest.raises(RuntimeError,match='closed'):call(a,b)
 backend.fail=None;call.close();assert not backend.live and call.cache_info()['owned_leases']==0


def test_reentrant_existing_backward_does_not_steal_the_outer_gate():
 backend=Backend();call=ResidentMatmulOffload(backend=backend);a,b,_=[torch.from_numpy(value).requires_grad_() for value in data()];old=call(a,b);seen=[]
 def callback(phase):
  if phase!='program':return
  with pytest.raises(RuntimeError,match='already active'):torch.autograd.grad(old,(a,b),torch.ones_like(old))
  assert core._busy
  with pytest.raises(RuntimeError,match='already active'):core.MatmulOffload(lambda jobs:[])(a,b)
  with pytest.raises(RuntimeError,match='already active'):call.close()
  seen.append(True)
 backend.callback=callback;call(a,b);assert seen==[True];backend.callback=None;call.close();assert not backend.live


def test_clear_releases_cache_without_closing_session():
 backend=Backend();call=ResidentMatmulOffload(backend=backend);a,b,_=[torch.from_numpy(value).requires_grad_() for value in data()];call(a,b);call.clear();assert not backend.live and not call.cache_info()['closed'];call(a,b);call.close();assert not backend.live


@pytest.mark.parametrize('option,value,error',[('max_entries',True,TypeError),('max_entries',9,ValueError),('max_bytes',-1,ValueError),('max_bytes',8388609,ValueError)])
def test_cache_limits_do_not_raise_host_quotas(option,value,error):
 with pytest.raises(error):ResidentMatmulOffload(**{option:value},backend=Backend())


def test_equal_corner_samples_do_not_alias_distinct_interior_values():
 backend=Backend();a=torch.ones(3,3,requires_grad=True);b=torch.full((3,3),.5,requires_grad=True)
 with ResidentMatmulOffload(backend=backend) as call:
  first=call(a,b).detach().numpy().copy();a.detach().numpy()[0,1]=3.
  second=call(a,b);np.testing.assert_array_equal(second.numpy(),a.numpy()@b.numpy());assert not np.array_equal(first,second.numpy())
 assert not backend.live
