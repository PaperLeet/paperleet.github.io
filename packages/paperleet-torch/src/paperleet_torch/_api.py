"""NumPy-backed api implementation; extracted from PaperLeet."""
import types
import sys
import numpy as np


def create_torch(name='torch', *, register=False):
    torch = types.ModuleType(name)
    torch.__path__ = []
    torch._browser_unsupported = _errors.BrowserUnsupported
    torch.Tensor, torch.Generator, torch.no_grad = _tensor.Tensor, _random.Generator, _autograd.no_grad
    torch.enable_grad, torch.set_grad_enabled, torch.is_grad_enabled = _autograd.enable_grad, _autograd.set_grad_enabled, _autograd.is_grad_enabled
    for fmt in _conversion.MemoryFormat:
        setattr(torch, fmt.name, fmt)
    torch.float64, torch.float32 = np.dtype('float64'), np.dtype('float32')
    torch.complex64, torch.complex128 = np.dtype('complex64'), np.dtype('complex128')
    torch.float, torch.double = torch.float32, torch.float64
    torch.long, torch.int64, torch.bool = np.dtype('int64'), np.dtype('int64'), np.dtype('bool')
    torch.int, torch.int32, torch.uint8 = np.dtype('int32'), np.dtype('int32'), np.dtype('uint8')
    for dtype in ('float16', 'int8', 'int16', 'uint16', 'uint32', 'uint64'):
        setattr(torch, dtype, np.dtype(dtype))
    torch.half, torch.short, torch.char = torch.float16, torch.int16, torch.int8
    torch.cfloat, torch.cdouble = torch.complex64, torch.complex128
    torch.inf, torch.pi = np.inf, np.pi
    torch.device, torch.finfo = _device.Device, np.finfo
    root_name = name
    modules = {'torch': torch}
    for name in ('nn', 'nn.functional', 'nn.parameter', 'nn.modules', 'nn.modules.module',
                 'nn.modules.container', 'nn.modules.linear', 'nn.modules.activation',
                 'nn.modules.normalization', 'nn.modules.dropout', 'nn.modules.sparse', 'nn.modules.loss',
                 'nn.modules.flatten', 'nn.modules.distance',
                 'utils', 'utils.hooks', 'optim', 'optim.optimizer', 'optim.sgd', 'optim.adam',
                 'optim.adamw', 'optim.lr_scheduler', 'autograd', 'linalg', 'testing', 'random', 'func', 'cuda'):
        module = types.ModuleType(root_name + '.' + name)
        def missing(key, _name=name):
            if key.startswith('_'): raise AttributeError(key)
            return _errors.unsupported('torch.' + _name + '.' + key)
        module.__getattr__ = missing
        modules['torch.' + name] = module
        if register: sys.modules[module.__name__] = module
    for name in ('nn', 'optim', 'autograd', 'linalg', 'testing', 'random', 'func', 'cuda'):
        setattr(torch, name, modules['torch.' + name])
    torch.nn.functional = modules['torch.nn.functional']
    for name, cls in {
        'Module': _nn.Module, 'Parameter': _nn.Parameter, 'Linear': _nn.Linear, 'Sequential': _nn.Sequential,
        'ModuleList': _nn.ModuleList, 'ModuleDict': _nn.ModuleDict,
        'ParameterList': _nn.ParameterList, 'ParameterDict': _nn.ParameterDict,
        'SiLU': _nn.SiLU, 'ReLU': _nn.ReLU, 'Softplus': _nn.Softplus, 'LogSigmoid': _nn.LogSigmoid,
        'BCEWithLogitsLoss': _nn.BCEWithLogitsLoss, 'KLDivLoss': _nn.KLDivLoss, 'CosineSimilarity': _nn.CosineSimilarity,
    }.items(): setattr(torch.nn, name, cls)
    for module, names in (
        (_normalization,('LayerNorm','RMSNorm')),
        (_activation,('GELU','Dropout')),
        (_nn_modules,('Embedding','Identity','Flatten','Tanh','Sigmoid','MSELoss','SmoothL1Loss','CrossEntropyLoss')),
    ):
        for name in names:setattr(torch.nn,name,getattr(module,name))
    torch.utils = modules['torch.utils']
    torch.utils.hooks = modules['torch.utils.hooks']
    torch.utils.hooks.RemovableHandle = _module_hooks.RemovableHandle
    torch.nn.parameter = modules['torch.nn.parameter']
    torch.nn.parameter.Parameter = torch.nn.Parameter
    torch.nn.modules = modules['torch.nn.modules']
    for category,names in {
        'module':('Module',),
        'container':('Sequential','ModuleList','ModuleDict','ParameterList','ParameterDict'),
        'linear':('Linear','Identity'),
        'activation':('ReLU','SiLU','Softplus','LogSigmoid','GELU','Tanh','Sigmoid'),
        'normalization':('LayerNorm','RMSNorm'),
        'dropout':('Dropout',),
        'sparse':('Embedding',),
        'loss':('MSELoss','SmoothL1Loss','CrossEntropyLoss','BCEWithLogitsLoss','KLDivLoss'),
        'flatten':('Flatten',),
        'distance':('CosineSimilarity',),
    }.items():
        child = modules['torch.nn.modules.'+category]
        setattr(torch.nn.modules,category,child)
        for name in names:
            setattr(child,name,getattr(torch.nn,name))
            setattr(torch.nn.modules,name,getattr(torch.nn,name))
    torch.optim.Optimizer = _optim.Optimizer
    torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD = _optim.Adam, _optim.AdamW, _optim.SGD
    for name, cls in (('optimizer', _optim.Optimizer), ('sgd', _optim.SGD),
                      ('adam', _optim.Adam), ('adamw', _optim.AdamW)):
        module = modules['torch.optim.' + name]
        setattr(module, cls.__name__, cls)
        setattr(torch.optim, name, module)

    torch.optim.lr_scheduler = modules['torch.optim.lr_scheduler']
    for name in ('LRScheduler', '_LRScheduler', 'StepLR', 'MultiStepLR', 'ExponentialLR',
                 'CosineAnnealingLR', 'LinearLR', 'ConstantLR', 'PolynomialLR', 'LambdaLR',
                 'MultiplicativeLR', 'ChainedScheduler', 'SequentialLR', 'ReduceLROnPlateau'):
        setattr(torch.optim.lr_scheduler, name, getattr(_schedulers, name))

    registry = _registry.OperatorRegistry(modules)
    registry.add('torch.promote_types', _construction.promote_types, reverse='control', forward=True)
    registry.group('torch.cuda', {'is_available': lambda: False, 'device_count': lambda: 0,
                                 'is_initialized': lambda: False})
    registry.group('torch', {'tensor': _ops.tensor, 'as_tensor': _ops.as_tensor}, reverse='factory', forward=True)
    registry.group('torch', {
        'from_numpy': _ops.from_numpy, 'full': _ops.full, 'full_like': _ops.full_like,
        'randint': _random.randint, 'randperm': _random.randperm,
        'eye': lambda n,m=None,*,dtype=np.float32,device=None,requires_grad=False: (_device.check_device(device), _tensor.Tensor(np.eye(n,m,dtype=dtype),requires_grad=requires_grad))[1],
        'linspace': lambda start,end,steps,dtype=np.float32,device=None,requires_grad=False: (_device.check_device(device), _tensor.Tensor(np.linspace(start,end,steps,dtype=dtype),requires_grad=requires_grad))[1],
        'arange': lambda *args,dtype=None,device=None: (_device.check_device(device), _tensor.Tensor(np.arange(*args,dtype=dtype or (np.float32 if any(isinstance(x,float) for x in args) else np.int64))))[1],
    }, reverse='factory')
    for name in ('randn', 'rand', 'zeros', 'ones', 'empty'):
        registry.add('torch.' + name, lambda *shape,_kind=name,**kw: _random.create(_kind,*shape,**kw), reverse='factory')
    for name in ('randn', 'zeros', 'ones', 'empty'):
        registry.add('torch.' + name + '_like',
                     lambda input,_kind=name,**kw: _random.create(_kind,*input.shape,**_ops.inherited_options(input.dtype,kw)),
                     reverse='factory', forward=True)
    registry.group('torch', {
        'where': _ops.where, 'cat': _ops.concatenate, 'stack': lambda tensors,dim=0: _ops.concatenate(tensors,dim,stack=True),
        'matmul': lambda input,other: _ops.wrap(input) @ other, 'pow': _ops.power,
        'cosine_similarity': _nn.cosine_similarity,
    }, reverse='analytic', forward=True)
    registry.group('torch', {
        'einsum': _ops.einsum, 'det': _linalg.det,
        'remainder': lambda input,other,*,out=None: input.remainder(other,out=out),
        'trace': lambda input: input.trace(),
        'diagonal': lambda input,offset=0,dim1=0,dim2=1: input.diagonal(offset,dim1,dim2),
        'logaddexp': lambda input,other: input.logaddexp(other),
        'minimum': _ops.elementwise_extreme,
        'maximum': lambda input,other: _ops.elementwise_extreme(input,other,maximum=True),
        'meshgrid': _ops.meshgrid, 'cartesian_prod': _ops.cartesian_prod,
        'sort': lambda input,dim=-1,descending=False,*,stable=False,out=None: input.sort(dim,descending,stable=stable,out=out),
        'trapezoid': _ops.trapezoid,
        'topk': lambda input,k,dim=None,largest=True,sorted=True,*,out=None: input.topk(k,dim,largest,sorted,out=out),
        'scatter': lambda input,dim,index,src,*,reduce=None: input.scatter(dim,index,src,reduce=reduce),
        'broadcast_to': lambda input,shape: input.expand(shape),
        'cov': _ops.cov, 'pdist': _ops.pdist, 'cdist': _ops.cdist,
    }, reverse='analytic')
    registry.add('torch.nan_to_num', _ops.nan_to_num, reverse='analytic', unsupported_keywords=('out',))
    def outer(input, other):
        if input.ndim != 1 or other.ndim != 1: raise RuntimeError('outer expects vectors')
        return input.unsqueeze(-1) * other.unsqueeze(0)
    def dot(input, other):
        if input.ndim != 1 or other.ndim != 1 or input.dtype != other.dtype:
            raise RuntimeError('dot expects vectors with matching dtypes')
        return input @ other
    registry.group('torch', {'outer': outer, 'dot': dot}, reverse='analytic')
    registry.group('torch', {
        'floor_divide': lambda input,other: _ops.wrap(input).floor_divide(other),
        'unique': _ops.unique, 'bincount': _ops.bincount,
    })  # Reverse AD is unavailable; plain values remain supported.
    registry.group('torch', {
        'equal': lambda input,other: bool(np.array_equal(_ops.wrap(input)._data,_ops.wrap(other)._data)),
        'argsort': lambda input,dim=-1,descending=False,*,stable=False: input.argsort(dim,descending,stable=stable),
        'count_nonzero': _ops.count_nonzero,
        'multinomial': _random.multinomial,
        'unravel_index': lambda index,shape: tuple(_tensor.Tensor(i) for i in np.unravel_index(int(index),shape)),
        'is_floating_point': lambda input: input.is_floating_point(),
        'isclose': lambda input,other,**kw: _tensor.Tensor(np.isclose(_ops.wrap(input)._data,_ops.wrap(other)._data,**kw)),
        'allclose': lambda input,other,**kw: bool(torch.isclose(input,other,**kw).all()),
    }, reverse='nondifferentiable')
    registry.group('torch', {
        'isfinite': lambda input: _tensor.Tensor(np.isfinite(_ops.wrap(input)._data)),
        'isnan': lambda input: _tensor.Tensor(np.isnan(_ops.wrap(input)._data)),
    }, reverse='nondifferentiable', forward=True)
    registry.group('torch', {'manual_seed': _random._default_generator.manual_seed, 'broadcast_shapes': np.broadcast_shapes}, reverse='control')

    # These declarations generate the top-level Tensor aliases and their guards.
    # Each tuple is (reverse-mode policy, validated JVP, unsupported native kwargs).
    methods = {
        **dict.fromkeys(('sum','exp','log','sqrt','abs','sin','cos','tanh','rsqrt','sigmoid'), ('analytic',True,('out',))),
        **dict.fromkeys(('mean',), ('analytic',True,('out','dtype'))),
        **dict.fromkeys(('relu','softmax','log_softmax','reshape','flatten'), ('analytic',True,())),
        **dict.fromkeys(('clamp','clamp_min','clamp_max','acos','var','std','cumsum','cumprod','norm','logsumexp','diag','triu','tril','amax','prod','round'), ('analytic',False,('out',))),
        **dict.fromkeys(('transpose','repeat_interleave','flip','roll','chunk','erfinv'), ('analytic',False,())),
        **dict.fromkeys(('max','min'), ('analytic',False,('out',))),
        **dict.fromkeys(('all','any','argmax','argmin'), ('nondifferentiable',False,('out',))),
    }
    for name, (reverse, forward, keywords) in methods.items():
        registry.method(name, reverse=reverse, forward=forward, unsupported_keywords=keywords)
    registry.add('torch.logit', lambda input,eps=None: input.logit(eps), reverse='analytic')

    registry.group('torch.nn.functional', {
        'linear': _nn.linear, 'softmax': lambda input,dim,dtype=None: input.softmax(dim,dtype),
        'log_softmax': lambda input,dim,dtype=None: input.log_softmax(dim,dtype),
        'logsigmoid': lambda input: input.logsigmoid(), 'softplus': _nn.softplus,
        'cosine_similarity': _nn.cosine_similarity, 'binary_cross_entropy_with_logits': _nn.binary_cross_entropy_with_logits,
        'kl_div': _nn.kl_div, 'silu': _nn.silu, 'gelu': _activation.gelu, 'dropout': _activation.dropout,
        'layer_norm': _normalization.layer_norm, 'rms_norm': _normalization.rms_norm,
        'embedding': _nn.embedding, 'mse_loss': _nn.mse_loss,
    }, reverse='analytic', forward=True)
    def relu(input, inplace=False):
        if inplace: raise _errors.BrowserUnsupported('In-place ReLU requires the server')
        return input.relu()
    registry.add('torch.nn.functional.relu', relu, reverse='analytic', forward=True)
    registry.group('torch.nn.functional', {
        'interpolate': _nn.interpolate, 'adaptive_avg_pool2d': _nn.adaptive_avg_pool2d,
        'normalize': _nn.normalize, 'cross_entropy': _nn.cross_entropy, 'smooth_l1_loss': _nn.smooth_l1_loss,
        'scaled_dot_product_attention': _nn.scaled_dot_product_attention_oracle,
    }, reverse='analytic')
    registry.group('torch.linalg', {
        'solve': _linalg.solve, 'inv': _linalg.inv, 'norm': _linalg.linalg_norm, 'svd': _linalg.svd, 'eigh': _linalg.eigh,
        'qr': _linalg.qr, 'eigvalsh': _linalg.eigvalsh, 'det': _linalg.det,
        'svdvals': lambda input,*,driver=None,out=None: _linalg.svd(input,False,driver=driver,out=out).S,
    }, reverse='analytic')
    registry.add('torch.linalg.matrix_rank', lambda input: _tensor.Tensor(np.linalg.matrix_rank(input._data),dtype=np.int64), reverse='nondifferentiable')
    registry.add('torch.testing.assert_close', _ops.assert_close, reverse='control')
    registry.add('torch.random.fork_rng', _random.fork_rng, reverse='control')
    registry.group('torch.autograd', {'grad': _autograd.autograd_grad, 'gradcheck': _autograd.gradcheck,
                                    'backward': _autograd.autograd_backward}, reverse='control')
    registry.add('torch.func.jvp', _autograd.jvp, reverse='control')
    torch._browser_capabilities = types.MappingProxyType(registry.capabilities)
    torch.__getattr__ = _errors.unsupported
    torch._compat_modules = types.MappingProxyType(modules)
    return torch


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _activation, _autograd, _construction, _conversion, _device, _errors, _linalg, _module_hooks, _nn, _nn_modules, _normalization, _ops, _optim, _random, _registry, _schedulers, _tensor
