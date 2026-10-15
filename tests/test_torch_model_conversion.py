"""Native-reference model conversion, retained storage and gradient-mode contracts."""
import copy

import numpy as np
import pytest
import torch as native

from paperleet._vendor.torch import UnsupportedOperation, torch as lite


def tensor(api, values, dtype='float32', requires_grad=False):
    return api.tensor(np.asarray(values), dtype=getattr(api, dtype), requires_grad=requires_grad)


def build(api):
    model = api.nn.Module()
    model.weight = api.nn.Parameter(tensor(api, [[.123456789, .2], [.3, .987654321]]))
    model.alias = model.weight
    model.separate = api.nn.Parameter(model.weight.detach())
    model.child = api.nn.Linear(2, 1)
    model.same_child = model.child
    model.register_parameter('none', None)
    model.register_buffer('running', tensor(api, [1., 2.]))
    model.register_buffer('same_running', model.running)
    model.register_buffer('counter', api.tensor([3, 4], dtype=api.int64))
    model.register_buffer('boolean', api.tensor([True, False], dtype=api.bool))
    model.register_buffer('temporary', tensor(api, [5.]), persistent=False)
    model.register_buffer('absent', None)
    model.plain = tensor(api, [7.])
    for p in model.parameters():p.grad = api.ones_like(p)
    model.eval()
    return model


def summarize(model):
    names = list(model.state_dict())
    parameters = [(n, str(p.dtype).split('.')[-1], p.requires_grad, p.is_leaf,
                   str(p.grad.dtype).split('.')[-1] if p.grad is not None else None)
                  for n, p in model.named_parameters(remove_duplicate=False)]
    buffers = [(n, str(p.dtype).split('.')[-1], p.requires_grad, p.is_leaf)
               for n, p in model.named_buffers(remove_duplicate=False)]
    return names, parameters, buffers


@pytest.mark.parametrize('method', ['double', 'float_roundtrip', 'to_dtype', 'to_device_dtype', 'to_tensor', 'type_dtype', 'type_string', 'cpu'])
def test_recursive_conversion_parameter_gradient_identity_aliases_and_buffers_match_native(method):
    results = []
    for api in (lite, native):
        model = build(api)
        parameters, gradients = list(model.parameters()), [p.grad for p in model.parameters()]
        state = model.state_dict()
        old_pointers = [p.data_ptr() for p in parameters]
        integer, plain = model.counter, model.plain
        if method == 'double':result = model.double()
        elif method == 'float_roundtrip':result = model.double().float()
        elif method == 'to_dtype':result = model.to(dtype=api.float64)
        elif method == 'to_device_dtype':result = model.to(api.device('cpu', 1), api.float64, True)
        elif method == 'to_tensor':result = model.to(api.zeros(1, dtype=api.float64))
        elif method == 'type_dtype':result = model.type(api.float64)
        elif method == 'type_string':result = model.type('torch.DoubleTensor')
        else:result = model.cpu()
        assert result is model and model.child is model.same_child and model.weight is model.alias
        assert all(p is old for p, old in zip(model.parameters(), parameters))
        assert all(p.grad is old for p, old in zip(model.parameters(), gradients))
        assert all(not child.training for child in model.modules())
        assert model.plain is plain and model.plain.dtype == api.float32
        assert (model.counter is integer) == (method not in ('type_dtype', 'type_string', 'to_device_dtype'))
        assert state['weight'].dtype == api.float32
        if method == 'cpu':assert [p.data_ptr() for p in parameters] == old_pointers
        else:assert all(p.data_ptr() != old for p, old in zip(parameters, old_pointers))
        results.append((summarize(model), model.running is model.same_running,
                        model.weight.data_ptr() == model.separate.data_ptr()))
    assert results[0] == results[1]


def test_noop_to_and_copy_contiguous_layouts_preserve_native_identity_and_gradients():
    for api in (lite, native):
        leaf = api.arange(12, dtype=api.float64).reshape(3, 4).requires_grad_()
        view = leaf.T
        for args, kwargs in [([], {}), ([api.float64], {}), (['cpu'], {}),
                                 ([], {'dtype': None}),
                             ([], {'non_blocking': True})]:
            assert leaf.to(*args, **kwargs) is leaf
        assert leaf.contiguous() is leaf and leaf.cpu() is leaf
        cloned = view.clone()
        assert not cloned.is_contiguous()
        assert view.to(memory_format=api.contiguous_format) is view
        contiguous = view.to(copy=True, memory_format=api.contiguous_format)
        assert contiguous is not view and contiguous.is_contiguous()
        assert view.contiguous().is_contiguous()
        forced = leaf.to(copy=True)
        assert forced is not leaf and forced.data_ptr() != leaf.data_ptr()
        assert leaf.to(api.device('cpu', 1)) is not leaf
        (contiguous.square().sum() + forced.square().sum()).backward()
        np.testing.assert_array_equal(leaf.grad.numpy(), np.arange(12).reshape(3, 4) * 4)


def test_apply_child_overrides_recursion_modes_and_buffer_graphs_match_native():
    outputs = []
    for api in (lite, native):
        visits = []
        class Child(api.nn.Module):
            def __init__(self):
                super().__init__();self.p = api.nn.Parameter(tensor(api, [.3]));self.register_buffer('b', tensor(api, [.4]))
            def _apply(self, fn, recurse=True):
                visits.append('child');return super()._apply(fn, recurse=recurse)
        model = api.nn.Module();model.child = Child();model.root = api.nn.Parameter(tensor(api, [.2]))
        original = tensor(api, [.7], requires_grad=True);model.register_buffer('trainable', original)
        def convert(value):
            visits.append(('value', api.is_grad_enabled(), value.requires_grad))
            return value.double()
        model._apply(convert, recurse=False)
        assert model.child.p.dtype == api.float32 and model.root.dtype == api.float64
        assert model.trainable.requires_grad and not model.trainable.is_leaf
        model.trainable.square().sum().backward()
        np.testing.assert_allclose(original.grad.numpy(), [1.4])
        model.double();assert model.child.p.dtype == api.float64
        outputs.append(visits)
    assert outputs[0] == outputs[1]


OPERATIONS = ['pow', 'log', 'logit', 'relu', 'prod', 'cumprod', 'gather', 'pdist', 'einsum', 'solve', 'det', 'svd', 'eigh', 'embedding']


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('existing_grad', [False, True])
def test_graph_built_before_conversion_retains_original_values_dtypes_and_gradient_buffers(operation, existing_grad):
    results = []
    for api in (lite, native):
        model = api.nn.Module()
        model.p = api.nn.Parameter(tensor(api, [[.234567891, .123456789], [.321987654, .876543219]], 'float64'))
        p = model.p
        if operation == 'pow':value = p.pow(2.7)
        elif operation == 'log':value = p.log()
        elif operation == 'logit':value = p.logit()
        elif operation == 'relu':value = p.relu()
        elif operation == 'prod':value = p.prod(1)
        elif operation == 'cumprod':value = p.cumprod(1)
        elif operation == 'gather':value = p.gather(1, api.tensor([[1, 0], [0, 1]], dtype=api.int64))
        elif operation == 'pdist':value = api.pdist(p)
        elif operation == 'einsum':value = api.einsum('ij,jk->ik', p, p)
        elif operation == 'solve':value = api.linalg.solve(p, api.ones(2, dtype=api.float64))
        elif operation == 'det':value = api.linalg.det(p)
        elif operation == 'svd':value = api.linalg.svd(p, full_matrices=False).S
        elif operation == 'eigh':value = api.linalg.eigh(p @ p.T).eigenvalues
        else:value = api.nn.functional.embedding(api.tensor([1, 0, 1], dtype=api.int64), p)
        loss = value.sum()
        if existing_grad:p.grad = api.ones_like(p)
        before = p._version
        model.float()
        assert p is model.p and p._version == before
        grad = p.grad
        if operation in ('pdist', 'eigh'):
            with pytest.raises(RuntimeError):loss.backward()
            results.append(('dtype_error', None))
            continue
        loss.backward()
        assert (grad is p.grad) == existing_grad
        results.append((p.grad.dtype, p.grad.detach().numpy()))
    if operation in ('pdist', 'eigh'):
        assert results == [('dtype_error', None), ('dtype_error', None)]
        return
    assert results[0][0].name == str(results[1][0]).split('.')[-1]
    # Native saved-self formulas can execute float32 subexpressions after the
    # cast even though the accumulation edge still produces a float64 gradient.
    tolerance = 3e-6 if existing_grad or operation in ('pow', 'log', 'logit', 'prod', 'cumprod') else 1e-11
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('source,target', [('float32', 'float64'), ('float64', 'float32')])
@pytest.mark.parametrize('momentum', [0., .8])
def test_optimizer_parameter_references_and_existing_sgd_momentum_survive_conversion(source, target, momentum):
    outputs = []
    for api in (lite, native):
        model = api.nn.Linear(2, 1, dtype=getattr(api, source))
        with api.no_grad():model.weight.fill_(.2);model.bias.fill_(.1)
        optimizer = api.optim.SGD(model.parameters(), lr=.03, momentum=momentum)
        x = api.ones(2, 2, dtype=getattr(api, source))
        model(x).square().sum().backward();optimizer.step();optimizer.zero_grad()
        old = model(x).square().sum()
        parameters = list(model.parameters());model.to(dtype=getattr(api, target))
        old.backward();optimizer.step()
        for p, old_p in zip(model.parameters(), parameters):assert p is old_p
        optimizer.zero_grad()
        model(x.to(dtype=getattr(api, target))).square().sum().backward();optimizer.step()
        outputs.append([p.detach().numpy() for p in model.parameters()])
    tolerance = 4e-6 if 'float32' in (source, target) else 1e-12
    for a, b in zip(*outputs):np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)


def test_mutating_rebound_storage_invalidates_old_and_new_graphs_and_alias_counters():
    for api in (lite, native):
        model = api.nn.Linear(2, 1)
        old_alias = model.weight.detach()
        old = model.weight.square().sum()
        model.double()
        new = model.weight.square().sum()
        with api.no_grad():model.weight.fill_(.3)
        for loss in (old, new):
            with pytest.raises((RuntimeError, UnsupportedOperation)):loss.backward()
        assert old_alias._version == model.weight._version


@pytest.mark.parametrize('spec', [('cpu',), ('cpu', 0), ('cpu', 1), ('cpu:1',), ('cuda:0',), ('mps',), ('meta',)])
def test_device_metadata_matches_native_without_claiming_execution(spec):
    a, b = lite.device(*spec), native.device(*spec)
    assert (str(a), repr(a), a.type, a.index, a == 'cpu') == (str(b), repr(b), b.type, b.index, b == 'cpu')
    assert lite.device(a) == a and hash(lite.device(a)) == hash(a)
    if a.type != 'cpu':
        with pytest.raises(UnsupportedOperation):lite.zeros(1, device=a)


def test_invalid_conversions_execution_boundaries_and_cpu_capabilities_are_explicit():
    model = lite.nn.Linear(2, 1)
    for kwargs in ({'device': 'cuda'}, {'device': 'mps'}, {'memory_format': lite.channels_last}):
        with pytest.raises(UnsupportedOperation):model.to(**kwargs)
    with pytest.raises(TypeError):model.to(dtype=lite.int64)
    with pytest.raises(RuntimeError):model.to(copy=False)
    with pytest.raises(TypeError):model.to('cpu', device='cpu')
    with pytest.raises(TypeError):model.to(non_blocking=1)
    with pytest.raises(UnsupportedOperation):model._apply(lambda p: p.reshape(-1))
    assert not lite.cuda.is_available() and not lite.cuda.is_initialized() and lite.cuda.device_count() == 0
    with pytest.raises(UnsupportedOperation):lite.cuda.synchronize()
    with pytest.raises(RuntimeError):lite.device('cpu', -1)


@pytest.mark.parametrize('operation', ['abs','log','pow','prod','cumprod','norm','std','amax','logsigmoid','silu'])
def test_saved_self_cast_underflow_matches_native_values_nans_and_errors(operation):
    results=[]
    for api in (lite,native):
        model=api.nn.Module();model.p=api.nn.Parameter(tensor(api,[1e-50,3e-50],'float64'))
        p=model.p
        if operation=='pow':value=p.pow(.5)
        elif operation=='cumprod':value=p.cumprod(0)
        elif operation=='norm':value=p.norm()
        elif operation=='logsigmoid':value=api.nn.functional.logsigmoid(p)
        elif operation=='silu':value=api.nn.functional.silu(p)
        else:value=getattr(p,operation)()
        loss=value.sum();model.float()
        if operation=='logsigmoid':
            with pytest.raises(RuntimeError):loss.backward()
            continue
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            loss.backward()
        results.append(p.grad.detach().numpy())
    if results:np.testing.assert_allclose(results[0],results[1],rtol=3e-6,atol=0,equal_nan=True)


@pytest.mark.parametrize('shapes', [((2,),(2,)),((2,2),(2,)),((2,),(2,2)),((2,2),(2,2)),
                                   ((1,2,2),(1,2,2)),((1,2,2),(2,2)),((2,2),(1,2,2)),
                                   ((2,),(1,2,2)),((1,2,2),(2,))])
@pytest.mark.parametrize('requires', [(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('tiny',[False,True])
def test_matmul_saved_views_and_dtype_errors_across_conversion_match_native(shapes,requires,tiny):
    results=[]
    for api in (lite,native):
        model=api.nn.Module()
        scale=1e-50 if tiny else .3
        model.a=api.nn.Parameter(tensor(api,np.full(shapes[0],scale),'float64'),requires_grad=requires[0])
        model.b=api.nn.Parameter(tensor(api,np.full(shapes[1],scale*2),'float64'),requires_grad=requires[1])
        loss=(model.a@model.b).sum();model.float()
        try:
            loss.backward()
            results.append([p.grad.detach().numpy() if p.grad is not None else None for p in model.parameters()])
        except RuntimeError:results.append('dtype_error')
    if isinstance(results[0],str) or isinstance(results[1],str):assert results==['dtype_error','dtype_error']
    else:
        for a,b in zip(*results):
            if a is None or b is None:assert a is b
            else:np.testing.assert_allclose(a,b,rtol=3e-6,atol=0,equal_nan=True)
