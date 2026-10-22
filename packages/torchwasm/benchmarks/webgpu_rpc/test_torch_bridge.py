import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch as native
from torchwasm import torch
from torch_bridge import MatmulOffload
import torch_bridge as bridge_module


def array(rows,cols,offset,layout):
 value=((np.arange(rows*cols,dtype='float32')%17-8)*.0625+offset).reshape(rows,cols)
 if layout=='F':return np.array(value,order='F')
 if layout=='slice':
  storage=np.empty((rows,cols*2),dtype='float32');storage[:,::2]=value;return storage[:,::2]
 return value


class Submit:
 def __init__(self):self.calls=[];self.callback=None
 def __call__(self,jobs):
  self.calls.append([(a.shape,b.shape,ta,tb) for a,b,ta,tb in jobs])
  if self.callback:self.callback()
  return [(a.T if ta else a)@(b.T if tb else b) for a,b,ta,tb in jobs]


@pytest.fixture(autouse=True)
def reset():
 assert not bridge_module._busy
 with torch.enable_grad():yield
 assert not bridge_module._busy


@pytest.mark.parametrize('shape',[(3,5,4),(1,7,3),(37,53,29),(64,64,64)])
@pytest.mark.parametrize('layout',['C','F','slice'])
@pytest.mark.parametrize('activity',['none','left','right','both'])
def test_native_value_nonlinear_grad_and_caller_buffers(shape,layout,activity):
 m,k,n=shape;aa=array(m,k,.2,layout);bb=array(k,n,.3,layout);apis=[torch,native];results=[];sender=Submit();offload=MatmulOffload(sender)
 for api in apis:
  a=api.from_numpy(aa).requires_grad_(activity in ['left','both']);b=api.from_numpy(bb).requires_grad_(activity in ['right','both'])
  assert a.stride()==tuple(v//aa.itemsize for v in aa.strides);assert b.stride()==tuple(v//bb.itemsize for v in bb.strides)
  targets=tuple(v for v in [a,b] if v.requires_grad);buffers=[]
  for v in targets:v.grad=api.full_like(v,7.);buffers.append(v.grad)
  y=offload(a,b) if api is torch else a@b;loss=(y.sin()+y.square()*.1).sum()
  grads=api.autograd.grad(loss,targets) if targets else ()
  assert all(v.grad is old and np.all(v.grad.numpy()==7.) for v,old in zip(targets,buffers))
  results.append((y.detach().numpy().copy(),*[g.detach().numpy().copy() for g in grads]))
 for value,expected in zip(*results):np.testing.assert_allclose(value,expected,rtol=8e-6,atol=4e-6)
 assert len(sender.calls)==(1 if activity=='none' else 2)
 assert len(sender.calls[-1])==(2 if activity=='both' else 1)


@pytest.mark.parametrize('target',['left','right','both'])
def test_target_pruning_and_duplicate_input_edges(target):
 sender=Submit();offload=MatmulOffload(sender);a=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True);b=torch.tensor([[2.,3.],[4.,5.]],requires_grad=True)
 y=offload(a,b);inputs=(a,) if target=='left' else (b,) if target=='right' else (a,b)
 grads=torch.autograd.grad(y,inputs,torch.ones_like(y));assert len(sender.calls[-1])==len(inputs)
 for g,expected in zip(grads,([5.,9.] if target=='left' else [4.,6.],)):
  assert np.isfinite(g.numpy()).all()
 x=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True);z=offload(x,x);dx,=torch.autograd.grad(z,x,torch.ones_like(z));np.testing.assert_array_equal(dx.numpy(),[[7.,11.],[9.,13.]])


@pytest.mark.parametrize('kind',['input','alias','activity','mode'])
def test_forward_rejects_state_changes_during_submission(kind):
 sender=Submit();offload=MatmulOffload(sender);a=torch.ones(2,2,requires_grad=True);b=torch.ones(2,2,requires_grad=True);sentinel=torch.full_like(a,7.);a.grad=sentinel
 def change():
  if kind in ['input','alias']:
   with torch.no_grad():(a if kind=='input' else a.view(4)).add_(1.)
  elif kind=='activity':a.requires_grad_(False)
  else:torch.set_grad_enabled(False)
 sender.callback=change
 with pytest.raises(RuntimeError,match='changed'):offload(a,b)
 assert a.grad is sentinel and sentinel.tolist()==[[7.,7.],[7.,7.]]
 sender.callback=None;torch.set_grad_enabled(True);a.requires_grad_(True);offload(a,b)


@pytest.mark.parametrize('where',['forward','backward'])
def test_transport_failure_propagates_and_busy_state_recovers(where):
 sender=Submit();offload=MatmulOffload(sender);a=torch.ones(2,2,requires_grad=True);b=torch.ones(2,2);y=offload(a,b) if where=='backward' else None
 def fail():raise RuntimeError('injected transport failure')
 sender.callback=fail
 with pytest.raises(RuntimeError,match='injected'):
  if y is None:offload(a,b)
  else:torch.autograd.grad(y,a,torch.ones_like(y))
 assert a.grad is None
 sender.callback=None
 if y is None:y=offload(a,b)
 da,=torch.autograd.grad(y,a,torch.ones_like(y));np.testing.assert_array_equal(da.numpy(),np.full((2,2),2.))


def test_required_operand_mutation_during_backward_and_seed_version_guard():
 sender=Submit();offload=MatmulOffload(sender);a=torch.ones(2,2,requires_grad=True);b=torch.ones(2,2);y=offload(a,b)
 def mutate():
  with torch.no_grad():b.add_(1.)
 sender.callback=mutate
 with pytest.raises(RuntimeError,match='changed'):torch.autograd.grad(y,a,torch.ones_like(y))
 sender=Submit();offload=MatmulOffload(sender);a=torch.ones(2,2,requires_grad=True);b=torch.ones(2,2);y=offload(a,b);g=torch.ones_like(y)
 def seed():
  with torch.no_grad():g.add_(1.)
 sender.callback=seed
 with pytest.raises(RuntimeError,match='seed changed'):torch.autograd.grad(y,a,g)
 assert a.grad is None


def test_only_needed_values_are_guarded_and_result_mutation_is_allowed():
 sender=Submit();offload=MatmulOffload(sender);a=torch.ones(2,2,requires_grad=True);b=torch.ones(2,2);y=offload(a,b)
 with torch.no_grad():a.add_(1.);y.detach().add_(1.)
 da,=torch.autograd.grad(y,a,torch.ones_like(y));np.testing.assert_array_equal(da.numpy(),np.full((2,2),2.))
 y=offload(a,b)
 with torch.no_grad():b.add_(1.)
 with pytest.raises(RuntimeError,match='modified'):torch.autograd.grad(y,a,torch.ones_like(y))


def test_reentry_does_not_issue_nested_requests():
 sender=Submit();offload=MatmulOffload(sender);a=torch.ones(2,2,requires_grad=True);seen=[]
 def nested():
  with pytest.raises(RuntimeError,match='already active'):MatmulOffload(sender)(a,a)
  seen.append(True)
 sender.callback=nested;y=offload(a,a);assert seen==[True] and len(sender.calls)==1


def test_retained_graph_consumption_and_returned_gradient_ownership():
 sender=Submit();offload=MatmulOffload(sender);a=torch.ones(2,2,requires_grad=True);b=torch.ones(2,2,requires_grad=True);y=offload(a,b)
 first=torch.autograd.grad(y,(a,b),torch.ones_like(y),retain_graph=True)
 with torch.no_grad():first[0].fill_(99.)
 second=torch.autograd.grad(y,(a,b),torch.ones_like(y));np.testing.assert_array_equal(second[0].numpy(),np.full((2,2),2.));assert not np.shares_memory(first[0].numpy(),second[0].numpy())
 with pytest.raises(RuntimeError,match='second time'):torch.autograd.grad(y,(a,b),torch.ones_like(y))


@pytest.mark.parametrize('kind',['no_grad','float64','vector','batch','empty','nonfinite','subnormal','large_range','jvp','hvp','invalid'])
def test_cpu_paths_preserve_existing_functionality_without_transport(kind):
 sender=Submit();offload=MatmulOffload(sender);a=torch.ones(2,2,requires_grad=True);b=torch.ones(2,2)
 if kind=='no_grad':
  with torch.no_grad():y=offload(a,b);assert not y.requires_grad
 elif kind=='float64':y=offload(a.double(),b.double())
 elif kind=='vector':y=offload(a[0],b)
 elif kind=='batch':y=offload(a.unsqueeze(0),b.unsqueeze(0))
 elif kind=='empty':y=offload(torch.ones(2,0,requires_grad=True),torch.ones(0,2))
 elif kind in ['nonfinite','subnormal','large_range']:
  value={'nonfinite':float('inf'),'subnormal':np.finfo(np.float32).tiny,'large_range':1e30}[kind];other=torch.full((2,2),value);y=offload(a,other)
 elif kind=='jvp':
  (value,tangent)=torch.func.jvp(lambda x:offload(x,b),(a,),(torch.ones_like(a),));np.testing.assert_array_equal(tangent.numpy(),np.full((2,2),2.))
 elif kind=='hvp':
  value,hessian=torch.autograd.functional.hvp(lambda x:offload(x,b).square().sum(),a,torch.ones_like(a));na=native.ones(2,2,requires_grad=True);nb=native.ones(2,2);_,nh=native.autograd.functional.hvp(lambda x:(x@nb).square().sum(),na,native.ones_like(na));np.testing.assert_array_equal(hessian.numpy(),nh.numpy())
 else:
  with pytest.raises(RuntimeError):offload(a,torch.ones(3,4))
 assert not sender.calls


@pytest.mark.parametrize('kind',['count','dtype','shape','object'])
def test_invalid_transport_results_are_not_published(kind):
 values={'count':[],'dtype':[np.ones((2,2),dtype='float64')],'shape':[np.ones((2,3),dtype='float32')],'object':['invalid']}
 offload=MatmulOffload(lambda jobs:values[kind]);a=torch.ones(2,2,requires_grad=True)
 with pytest.raises(RuntimeError,match='invalid'):offload(a,a)
 assert a.grad is None


@pytest.mark.parametrize('momentum',[0.,.2])
def test_actual_backward_optimizer_and_transposed_nonleaf_weights(momentum):
 sender=Submit();offload=MatmulOffload(sender);results=[]
 for api in [torch,native]:
  weight=api.tensor([[.2,.3,.4],[.1,-.2,.3]],requires_grad=True);bias=api.tensor([.1,-.1],requires_grad=True);x=api.tensor([[1.,2.,3.],[-1.,.5,.25]])
  optimizer=api.optim.SGD([weight,bias],lr=.05,momentum=momentum,weight_decay=.01)
  for _ in range(5):
   optimizer.zero_grad();y=(offload(x,weight.T) if api is torch else x@weight.T)+bias;loss=y.sin().square().mean();loss.backward();optimizer.step()
  results.append((weight.detach().numpy().copy(),bias.detach().numpy().copy()))
 for v,e in zip(*results):np.testing.assert_allclose(v,e,rtol=8e-6,atol=4e-6)
 assert len(sender.calls)==10


def test_nonleaf_and_leaf_hooks_and_retain_grad():
 sender=Submit();offload=MatmulOffload(sender);x=torch.ones(2,2,requires_grad=True);a=x+1.;b=torch.ones(2,2,requires_grad=True);a.retain_grad();seen=[]
 a.register_hook(lambda g:seen.append(g.detach().numpy().copy()))
 y=offload(a,b);y.register_hook(lambda g:g*2.);y.sum().backward()
 np.testing.assert_array_equal(x.grad.numpy(),np.full((2,2),4.));np.testing.assert_array_equal(a.grad.numpy(),np.full((2,2),4.));np.testing.assert_array_equal(b.grad.numpy(),np.full((2,2),8.));assert len(seen)==1


def test_backward_under_no_grad_does_not_suspend():
 sender=Submit();offload=MatmulOffload(sender);a=torch.ones(2,2,requires_grad=True);y=offload(a,a)
 with torch.no_grad():g,=torch.autograd.grad(y,a,torch.ones_like(y))
 np.testing.assert_array_equal(g.numpy(),np.full((2,2),4.));assert len(sender.calls)==1


def test_program_partition_respects_aggregate_work_without_raising_limits():
 sender=Submit();offload=MatmulOffload(sender);a=torch.ones(1024,512,requires_grad=True);b=torch.ones(512,512,requires_grad=True);y=offload(a,b);da,db=torch.autograd.grad(y,(a,b),torch.ones_like(y))
 assert [len(jobs) for jobs in sender.calls]==[1,1,1]
 assert y.shape==(1024,512) and np.all(y.numpy()==512.) and np.all(da.numpy()==512.) and np.all(db.numpy()==1024.)


def test_numpy_shared_storage_changes_between_calls_match_native_matmul():
 values=[]
 for api in [torch,native]:
  a=api.ones(2,2,requires_grad=True);b=api.ones(2,2);sender=Submit();y=MatmulOffload(sender)(a,b) if api is torch else a@b
  version=b._version;b.numpy()[:]=3.;assert b._version==version
  da,=api.autograd.grad(y,a,api.ones_like(y));values.append(da.numpy().copy())
 np.testing.assert_array_equal(*values);np.testing.assert_array_equal(values[0],np.full((2,2),6.))


@pytest.mark.parametrize('when',['forward','backward_operand','backward_seed','signed_zero'])
def test_numpy_byte_writes_during_submission_are_rejected(when):
 sender=Submit();offload=MatmulOffload(sender);a=torch.zeros(2,2,requires_grad=True) if when=='signed_zero' else torch.ones(2,2,requires_grad=True);b=torch.ones(2,2)
 y=offload(a,b) if when.startswith('backward') else None;g=torch.ones_like(y) if y is not None else None
 target=a if when in ['forward','signed_zero'] else b if when=='backward_operand' else g
 def mutate():target.detach().numpy()[0,0]=-0. if when=='signed_zero' else 3.
 sender.callback=mutate
 with pytest.raises(RuntimeError,match='changed'):
  if y is None:offload(a,b)
  else:torch.autograd.grad(y,a,g)
 assert a.grad is None
