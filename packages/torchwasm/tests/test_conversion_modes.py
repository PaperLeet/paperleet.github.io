"""CPU model/dtype/gradient-mode contracts with NumPy only, also run in WASM."""
import copy
import gc

import numpy as np
import pytest

from torchwasm import UnsupportedOperation, torch


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_standard_cpu_device_selection_model_conversion_optimizer_and_checkpoint(dtype):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert str(device)=='cpu' and device.type=='cpu' and device.index is None
    assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
    model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.SiLU(),torch.nn.Linear(3,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    params=list(model.parameters())
    model.to(device=device,dtype=dtype)
    assert all(a is b for a,b in zip(params,model.parameters()))
    assert all(p.dtype==dtype and p.is_leaf and p.requires_grad for p in params)
    x=torch.tensor([[1.,.3],[-.2,2.]],dtype=dtype)
    with torch.no_grad():
        for p in params:p.fill_(.1)
    for _ in range(3):
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
    checkpoint=copy.deepcopy({'model':model.state_dict(),'optimizer':optimizer.state_dict()})
    model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
    model.cpu().eval()
    with torch.no_grad():assert not model(x).requires_grad
    model.train();optimizer.zero_grad();model(x).sum().backward();optimizer.step()


def test_shared_parameters_existing_gradient_objects_and_buffer_conversion_rules():
    model=torch.nn.Module();model.p=torch.nn.Parameter(torch.ones(2))
    model.alias=model.p;model.p.grad=torch.ones(2)
    model.register_buffer('running',torch.ones(2));model.register_buffer('alias_running',model.running)
    model.register_buffer('counter',torch.tensor([2],dtype=torch.int64))
    model.register_buffer('scratch',torch.zeros(1),persistent=False)
    p,grad,counter=model.p,model.p.grad,model.counter
    old=model.state_dict();assert model.double() is model
    assert model.p is p is model.alias and model.p.grad is grad
    assert grad.dtype==torch.float64 and old['p'].dtype==torch.float32
    assert model.counter is counter and counter.dtype==torch.int64
    assert model.running is not model.alias_running and model.scratch.dtype==torch.float64
    model.type('torch.DoubleTensor');assert model.counter.dtype==torch.float64
    model.float();assert all(p.dtype==torch.float32 for p in model.parameters())


def test_copy_layout_device_metadata_and_same_dtype_identity():
    x=torch.arange(12,dtype=torch.float64).reshape(3,4).requires_grad_()
    assert x.to() is x and x.to(dtype=x.dtype) is x and x.cpu() is x and x.contiguous() is x
    assert x.to(tensor=torch.ones(1,dtype=x.dtype)) is x
    assert not (x.device=='cpu') and x.device==torch.device('cpu')
    assert repr(torch.device('cpu',1))=="device(type='cpu', index=1)"
    assert x.to('cpu:0').data_ptr()!=x.data_ptr()
    assert x.to(torch.device('cpu',1)).data_ptr()!=x.data_ptr()
    view=x.T;assert view.to(memory_format=torch.contiguous_format) is view
    assert not view.clone().is_contiguous() and view.contiguous().is_contiguous()
    packed=view.to(copy=True,memory_format=torch.contiguous_format)
    assert packed.is_contiguous() and packed is not view
    packed.square().sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),np.arange(12).reshape(3,4)*2)


def test_contiguous_copy_preserves_forward_tangents_and_reverse_gradients():
    data=np.arange(12,dtype=np.float64).reshape(3,4).T
    tangent=np.linspace(.1,1.2,12).reshape(3,4).T
    x=torch.tensor(data,requires_grad=True)
    primal,derivative=torch.func.jvp(lambda value:value.contiguous().square(),
                                    (x,),(torch.tensor(tangent),))
    assert primal.is_contiguous()
    np.testing.assert_array_equal(primal.numpy(),data**2)
    np.testing.assert_allclose(derivative.numpy(),2*data*tangent)
    primal.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),2*data)


def test_gradient_modes_recursive_decorators_factories_and_generator_resumption():
    before=torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(True)
        x=torch.tensor([2.,3.],requires_grad=True)
        with torch.no_grad():
            assert not x.square().requires_grad
            assert torch.nn.Parameter(torch.ones(1)).requires_grad
            with torch.enable_grad():loss=x.square().sum()
            assert not torch.is_grad_enabled()
        loss.backward();np.testing.assert_array_equal(x.grad.numpy(),[4.,6.])
        @torch.no_grad()
        def recursive(depth):
            assert not torch.is_grad_enabled()
            return recursive(depth-1) if depth else x.square()
        assert not recursive(3).requires_grad and torch.is_grad_enabled()
        flags=[]
        @torch.set_grad_enabled(False)
        def stream():
            try:
                flags.append(torch.is_grad_enabled());received=yield x.square()
                assert received==9
                try:yield x*3
                except ValueError:flags.append(torch.is_grad_enabled());yield x*4
                return 17
            finally:flags.append(torch.is_grad_enabled())
        iterator=stream();assert not next(iterator).requires_grad and torch.is_grad_enabled()
        assert not iterator.send(9).requires_grad and torch.is_grad_enabled()
        assert not iterator.throw(ValueError()).requires_grad and torch.is_grad_enabled()
        with pytest.raises(StopIteration) as done:next(iterator)
        assert done.value.value==17 and flags==[False,False,False] and torch.is_grad_enabled()
        torch.set_grad_enabled(False);assert not torch.is_grad_enabled()
        torch.set_grad_enabled(True)
    finally:torch.set_grad_enabled(before)


def test_live_conversion_edges_gradient_accumulation_and_storage_mutation_guards():
    model=torch.nn.Module();model.p=torch.nn.Parameter(torch.tensor([.2,.7]))
    old=model.p.square().sum();model.double();old.backward()
    assert model.p.grad.dtype==torch.float32
    np.testing.assert_allclose(model.p.grad.numpy(),[.4,1.4])
    grad=model.p.grad
    model.p.square().sum().backward()
    assert model.p.grad is grad and model.p.grad.dtype==torch.float32
    np.testing.assert_allclose(model.p.grad.numpy(),[.8,2.8])
    model.zero_grad();old=model.p.square().sum();alias=model.p.detach()
    model.float();new=model.p.square().sum()
    with torch.no_grad():model.p.fill_(.9)
    assert alias._version==model.p._version
    for loss in (old,new):
        with pytest.raises(UnsupportedOperation,match='modified in-place'):loss.backward()


def test_saved_self_and_saved_output_have_different_cast_underflow_behavior():
    outputs={}
    for kind in ('abs','relu','norm','amax'):
        model=torch.nn.Module();model.p=torch.nn.Parameter(torch.tensor([1e-50,3e-50],dtype=torch.float64))
        loss=getattr(model.p,kind)().sum();model.float()
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):loss.backward()
        outputs[kind]=model.p.grad.numpy()
    np.testing.assert_array_equal(outputs['abs'],[0.,0.])
    np.testing.assert_array_equal(outputs['relu'],[1.,1.])
    assert np.isnan(outputs['norm']).all()
    np.testing.assert_array_equal(outputs['amax'],[.5,.5])


def test_rebinding_counters_do_not_retain_discarded_models_or_arrays():
    from torchwasm import _state
    gc.collect();before=len(_state._storages)
    for _ in range(30):
        model=torch.nn.Linear(3,2);model.double().float().to('cpu:0')
        model(torch.ones(1,3)).square().sum().backward()
    del model;gc.collect()
    assert len(_state._storages)<=before


def test_unsupported_execution_and_invalid_conversion_arguments_fail_before_gpu_work():
    model=torch.nn.Linear(2,1)
    for device in ('cuda','mps','meta'):
        assert torch.device(device).type==device
        with pytest.raises(UnsupportedOperation):model.to(device)
    with pytest.raises(TypeError):model.to(dtype=torch.int64)
    with pytest.raises(RuntimeError):model.to(copy=False)
    with pytest.raises(UnsupportedOperation):model.to(memory_format=torch.channels_last)
    with pytest.raises(TypeError):torch.set_grad_enabled(1)
    with pytest.raises(TypeError):torch.ones(1).to('cpu',device='cpu')
