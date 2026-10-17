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
                 'nn.modules.normalization', 'nn.modules.conv', 'nn.modules.batchnorm', 'nn.modules.dropout', 'nn.modules.sparse', 'nn.modules.loss',
                 'nn.modules.flatten', 'nn.modules.distance', 'nn.modules.pooling',
                 'utils', 'utils.hooks', 'optim', 'optim.optimizer', 'optim.sgd', 'optim.adam',
                 'optim.adamw', 'optim.lr_scheduler', 'autograd', 'linalg', 'testing', 'random', 'func', 'cuda', 'serialization'):
        module = types.ModuleType(root_name + '.' + name)
        def missing(key, _name=name, _module=module):
            return _errors.missing_attribute(_module, key, 'torch.' + _name + '.' + key)
        module.__getattr__ = missing
        modules['torch.' + name] = module
        if register: sys.modules[module.__name__] = module
    for name in ('nn', 'optim', 'autograd', 'linalg', 'testing', 'random', 'func', 'cuda', 'serialization'):
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
        (_groupnorm,('GroupNorm',)),
        (_classification,('NLLLoss',)),
        (_convolution,('Conv1d','Conv2d','Conv3d')),
        (_pooling,('MaxPool1d','MaxPool2d')),
        (_pooling3d,('MaxPool3d',)),
        (_average_pooling,('AvgPool1d','AdaptiveAvgPool1d','AvgPool2d','AdaptiveAvgPool2d')),
        (_batchnorm,('BatchNorm1d','BatchNorm2d','BatchNorm3d')),
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
        'normalization':('LayerNorm','RMSNorm','GroupNorm'),
        'conv':('Conv1d','Conv2d','Conv3d'),
        'pooling':('MaxPool1d','MaxPool2d','MaxPool3d','AvgPool1d','AdaptiveAvgPool1d','AvgPool2d','AdaptiveAvgPool2d'),
        'batchnorm':('BatchNorm1d','BatchNorm2d','BatchNorm3d'),
        'dropout':('Dropout',),
        'sparse':('Embedding',),
        'loss':('NLLLoss','MSELoss','SmoothL1Loss','CrossEntropyLoss','BCEWithLogitsLoss','KLDivLoss'),
        'flatten':('Flatten',),
        'distance':('CosineSimilarity',),
    }.items():
        child = modules['torch.nn.modules.'+category]
        setattr(torch.nn.modules,category,child)
        for name in names:
            setattr(child,name,getattr(torch.nn,name))
            setattr(torch.nn.modules,name,getattr(torch.nn,name))
    modules['torch.nn.modules.batchnorm']._BatchNorm = _batchnorm._BatchNorm
    modules['torch.nn.modules.conv']._ConvNd = _convolution._ConvNd
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
    registry.add('torch.result_type', _promotion.result_type, reverse='control', forward=True)
    registry.group('torch.cuda', {'is_available': lambda: False, 'device_count': lambda: 0,
                                 'is_initialized': lambda: False})
    registry.group('torch', {'tensor': _ops.tensor, 'as_tensor': _ops.as_tensor}, reverse='factory', forward=True)
    registry.group('torch', {
        'from_numpy': _ops.from_numpy, 'full': _ops.full, 'full_like': _ops.full_like,
        'randint': _random.randint, 'randperm': _random.randperm,
        'eye': _ops.eye,
        'linspace': _ops.linspace,
        'arange': _ops.arange,
    }, reverse='factory')
    for name in ('randn', 'rand', 'zeros', 'ones', 'empty'):
        registry.add('torch.' + name, lambda *shape,_kind=name,**kw: _random.create(_kind,*shape,**kw), reverse='factory')
    for name in ('randn', 'zeros', 'ones', 'empty'):
        registry.add('torch.' + name + '_like',
                     lambda input,_kind=name,**kw: _random.create(_kind,*input.shape,**_ops.inherited_options(input.dtype,kw)),
                     reverse='factory', forward=True)
    registry.group('torch', {
        'where': _ops.where, 'cat': _ops.cat, 'concat': _ops.cat, 'concatenate': _ops.cat, 'stack': _ops.stack,
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
        'trapezoid': _ops.trapezoid,
        'scatter': lambda input,dim,index,src,*,reduce=None: input.scatter(dim,index,src,reduce=reduce),
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
        'reciprocal': _arithmetic.reciprocal,
        'add': _arithmetic.add, 'sub': _arithmetic.sub, 'subtract': _arithmetic.sub,
        'mul': _arithmetic.mul, 'multiply': _arithmetic.mul,
        'div': _arithmetic.div, 'divide': _arithmetic.div, 'true_divide': _arithmetic.true_divide,
    }, reverse='analytic', forward=True)
    registry.group('torch', {
        'floor_divide': lambda input,other: _ops.wrap(input).floor_divide(other),
        'unique': _ops.unique, 'bincount': _ops.bincount,
    })  # Reverse AD is unavailable; plain values remain supported.
    registry.group('torch', {
        'equal': lambda input,other: bool(np.array_equal(_ops.wrap(input)._data,_ops.wrap(other)._data)),
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
        **dict.fromkeys(('var','std'), ('analytic',True,('out',))),
        **dict.fromkeys(('sum','exp','log','sqrt','abs','sin','cos','tanh','rsqrt','sigmoid'), ('analytic',True,('out',))),
        **dict.fromkeys(('mean','nansum','nanmean','gather','sort','topk'), ('analytic',True,('out',))),
        **dict.fromkeys(('relu','softmax','log_softmax','reshape'), ('analytic',True,())),
        **dict.fromkeys(('clamp','clamp_min','clamp_max','acos','cumsum','cumprod','norm','logsumexp','diag','triu','tril','amax','prod','round'), ('analytic',False,('out',))),
        **dict.fromkeys(('repeat_interleave','flip','roll','chunk','erfinv'), ('analytic',False,())),
        **dict.fromkeys(('max','min'), ('analytic',False,('out',))),
        **dict.fromkeys(('all','any','argmax','argmin'), ('nondifferentiable',False,('out',))),
    }
    for name, (reverse, forward, keywords) in methods.items():
        registry.method(name, reverse=reverse, forward=forward, unsupported_keywords=keywords)
    registry.method('argsort',reverse='nondifferentiable',forward=True)
    registry.group('torch', {'flatten':_ops.flatten,'transpose':_ops.transpose,'unsqueeze':_ops.unsqueeze,'squeeze':_ops.squeeze}, reverse='analytic', forward=True)
    registry.group('torch', {'t':_ops.t,'swapaxes':_ops.swapaxes,'swapdims':_ops.swapdims,
                            'movedim':_ops.movedim,'moveaxis':_ops.moveaxis}, reverse='analytic', forward=True)
    registry.add('torch.permute', _ops.permute, reverse='analytic', forward=True)
    registry.add('torch.broadcast_to', _ops.broadcast_to, reverse='analytic', forward=True)
    from . import _statistics
    registry.add('torch.var_mean', _statistics.var_mean, reverse='analytic', forward=True, unsupported_keywords=('out',))
    registry.add('torch.std_mean', _statistics.std_mean, reverse='analytic', forward=True, unsupported_keywords=('out',))
    registry.add('torch.logit', lambda input,eps=None: input.logit(eps), reverse='analytic')

    registry.group('torch.nn.functional', {
        'linear': _nn.linear, 'softmax': lambda input,dim,dtype=None: input.softmax(dim,dtype),
        'log_softmax': lambda input,dim,dtype=None: input.log_softmax(dim,dtype),
        'logsigmoid': lambda input: input.logsigmoid(), 'softplus': _nn.softplus,
        'cosine_similarity': _nn.cosine_similarity, 'binary_cross_entropy_with_logits': _nn.binary_cross_entropy_with_logits,
        'kl_div': _nn.kl_div, 'silu': _nn.silu, 'gelu': _activation.gelu, 'dropout': _activation.dropout,
        'layer_norm': _normalization.layer_norm, 'rms_norm': _normalization.rms_norm,
        'batch_norm': _batchnorm.batch_norm,
        'group_norm': _groupnorm.group_norm,
        'conv1d': _convolution.conv1d, 'conv2d': _convolution.conv2d, 'conv3d': _convolution.conv3d,
        'cross_entropy': _classification.cross_entropy, 'nll_loss': _classification.nll_loss,
        'embedding': _nn.embedding, 'mse_loss': _nn.mse_loss,
    }, reverse='analytic', forward=True)
    def relu(input, inplace=False):
        if inplace: raise _errors.BrowserUnsupported('In-place ReLU requires the server')
        return input.relu()
    registry.add('torch.nn.functional.relu', relu, reverse='analytic', forward=True)
    registry.add('torch.avg_pool1d', _average_pooling.avg_pool1d, reverse='analytic', forward=True)
    registry.add('torch.adaptive_avg_pool1d', _average_pooling.adaptive_avg_pool1d, reverse='analytic', forward=True)
    registry.add('torch.max_pool1d', _pooling.torch_max_pool1d, reverse='analytic', forward=True)
    registry.add('torch.max_pool1d_with_indices', _pooling.torch_max_pool1d_with_indices, reverse='analytic', forward=True)
    registry.add('torch.max_pool3d', _pooling3d.torch_max_pool3d, reverse='analytic', forward=True)
    registry.add('torch.max_pool2d', _pooling.torch_max_pool2d, reverse='analytic', forward=True)
    registry.group('torch.nn.functional', {
        'max_pool1d': _pooling.max_pool1d, 'max_pool1d_with_indices': _pooling.max_pool1d_with_indices,
        'max_pool3d': _pooling3d.max_pool3d, 'max_pool3d_with_indices': _pooling3d.max_pool3d_with_indices,
        'max_pool2d': _pooling.max_pool2d, 'max_pool2d_with_indices': _pooling.max_pool2d_with_indices,
        'avg_pool1d': _average_pooling.avg_pool1d,
        'adaptive_avg_pool1d': _average_pooling.adaptive_avg_pool1d,
        'avg_pool2d': _average_pooling.avg_pool2d,
        'adaptive_avg_pool2d': _nn.adaptive_avg_pool2d, 'interpolate': _nn.interpolate,
    }, reverse='analytic', forward=True)
    registry.group('torch.nn.functional', {
        'normalize': _nn.normalize, 'smooth_l1_loss': _nn.smooth_l1_loss,
        'scaled_dot_product_attention': _nn.scaled_dot_product_attention_oracle,
    }, reverse='analytic')
    registry.group('torch.linalg', {
        'inv': _linalg.inv, 'norm': _linalg.linalg_norm, 'svd': _linalg.svd, 'eigh': _linalg.eigh,
        'qr': _linalg.qr, 'eigvalsh': _linalg.eigvalsh, 'det': _linalg.det,
        'svdvals': lambda input,*,driver=None,out=None: _linalg.svd(input,False,driver=driver,out=out).S,
    }, reverse='analytic')
    registry.add('torch.linalg.solve', _linalg.solve, reverse='analytic', forward=True)
    registry.add('torch.linalg.solve_triangular', _linalg.solve_triangular, reverse='analytic', forward=True)
    registry.add('torch.linalg.cholesky', _cholesky.cholesky, reverse='analytic', forward=True)
    registry.add('torch.cholesky_solve', _cholesky.cholesky_solve, reverse='analytic', forward=True)
    registry.add('torch.cholesky_inverse', _cholesky.cholesky_inverse, reverse='analytic', forward=True)
    registry.group('torch', {'save':_serialization.save,'load':_serialization.load}, reverse='control')
    registry.group('torch.serialization', {'save':_serialization.save,'load':_serialization.load}, reverse='control')
    torch.serialization.save,torch.serialization.load = torch.save,torch.load
    registry.add('torch.linalg.matrix_rank', lambda input: _tensor.Tensor(np.linalg.matrix_rank(input._data),dtype=np.int64), reverse='nondifferentiable')
    registry.add('torch.testing.assert_close', _ops.assert_close, reverse='control')
    registry.add('torch.random.fork_rng', _random.fork_rng, reverse='control')
    registry.group('torch.autograd', {'grad': _autograd.autograd_grad, 'gradcheck': _autograd.gradcheck,
                                    'backward': _autograd.autograd_backward}, reverse='control')
    registry.add('torch.func.jvp', _autograd.jvp, reverse='control')
    torch._browser_capabilities = types.MappingProxyType(registry.capabilities)
    def missing_root(key):
        return _errors.missing_attribute(torch, key)
    torch.__getattr__ = missing_root
    torch._compat_modules = types.MappingProxyType(modules)
    return torch


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _arithmetic, _activation, _autograd, _average_pooling, _batchnorm, _cholesky, _classification, _construction, _convolution, _conversion, _device, _errors, _groupnorm, _linalg, _module_hooks, _nn, _nn_modules, _normalization, _ops, _optim, _pooling, _pooling3d, _promotion, _random, _registry, _schedulers, _serialization, _tensor
