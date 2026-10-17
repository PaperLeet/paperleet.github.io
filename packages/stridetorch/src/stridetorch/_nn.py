"""NumPy-backed nn implementation; extracted from PaperLeet."""
import math
import sys
import numpy as np


from . import _defaults, _tensor

from ._module import Module, Parameter
from ._containers import Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict

# NumPy on the pinned WASM runtime has much faster contraction loops for the
# two linear pullbacks. Native NumPy retains its BLAS-backed matrix products.
_WASM_LINEAR_GRADS = sys.platform == 'emscripten'


def linear(input, weight, bias=None):
    if any(isinstance(v, _autograd.ForwardTensor) for v in (input, weight, bias)):
        x, dx = _autograd.forward_parts(input); w, dw = _autograd.forward_parts(weight)
        b, db = _autograd.forward_parts(bias) if bias is not None else (None, 0)
        return _autograd.ForwardTensor(linear(x, w, b), linear(dx, w) + linear(x, dw) + db)
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[1] or input.dtype != weight.dtype:
        raise RuntimeError('linear expects matching feature dimensions and dtypes')
    if input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('linear requires float32/float64')
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != input.dtype):
        raise _errors.BrowserUnsupported('Other linear bias shapes/dtypes require the server')
    rows = math.prod(input.shape[:-1])
    x, w = input._data, weight._data
    # Direct contractions avoid the slower WASM matmul loop for measured
    # large, contiguous feature vectors. Keep CPU BLAS and small/strided inputs.
    if (_WASM_LINEAR_GRADS and w.shape[1] >= 64 and rows * w.size >= 32768
            and x.strides[-1] == x.itemsize and w.strides[1] == w.itemsize):
        value = np.einsum('...i,oi->...o', x, w, optimize=False)
    else:
        value = x @ w.T
    if bias is not None: value = value + bias._data
    if _WASM_LINEAR_GRADS:
        edges = [(input, lambda g: np.einsum('...o,oi->...i', g, w, optimize=False)),
                 (weight, lambda g: np.einsum('bo,bi->oi', g.reshape(rows, w.shape[0]),
                                             x.reshape(rows, w.shape[1]), optimize=False))]
    else:
        edges = [(input, lambda g: g @ w),
                 (weight, lambda g: g.reshape(rows, w.shape[0]).T @ x.reshape(rows, w.shape[1]))]
    if bias is not None: edges.append((bias, lambda g: g.reshape(rows, w.shape[0]).sum(axis=0)))
    return _autograd.result(value, *edges)


def silu(input, inplace=False):
    input._real_only()
    if inplace: raise _errors.BrowserUnsupported('In-place SiLU requires the server')
    if isinstance(input, _autograd.ForwardTensor): return input * input.sigmoid()
    x = input._data
    exp = np.exp(-np.abs(x))
    probability = np.where(x >= 0, 1 / (1 + exp), exp / (1 + exp))
    def backward(g):
        source, activation = x, probability
        if input._data is not x:
            source = input._data.astype(np.result_type(g.dtype,input.dtype),copy=False)
            exponential = np.exp(-np.abs(source))
            activation = np.where(source>=0,1/(1+exponential),exponential/(1+exponential))
        return g * activation * (1 + source * (1 - activation))
    return _autograd.result(x * probability, (input, backward))


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        _device.check_device(device)
        dtype = _defaults._float_dtype if dtype is None else np.dtype(dtype)
        if dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise _errors.BrowserUnsupported('Linear requires float32/float64')
        if type(in_features) is not int or type(out_features) is not int or min(in_features, out_features) < 0:
            raise ValueError('Linear dimensions must be nonnegative integers')
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(_tensor.Tensor(np.empty((out_features, in_features), dtype=dtype)))
        if bias:
            self.bias = Parameter(_tensor.Tensor(np.empty(out_features, dtype=dtype)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = self.in_features ** -.5 if self.in_features else 0
        with _autograd.no_grad():
            for parameter in (self.weight, self.bias):
                if parameter is not None:
                    parameter[...] = (_random.uniform_samples(np.random, parameter.shape, parameter.dtype) * (2 * bound) - bound).astype(parameter.dtype)

    def forward(self, input):
        return linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'




class SiLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        if inplace: raise _errors.BrowserUnsupported('In-place SiLU requires the server')
    def forward(self, input): return silu(input)


class Softplus(Module):
    def __init__(self, beta=1., threshold=20.):
        super().__init__()
        self.beta,self.threshold = beta,threshold
    def forward(self, input): return softplus(input,self.beta,self.threshold)


class LogSigmoid(Module):
    def forward(self, input): return input.logsigmoid()


class CosineSimilarity(Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim,self.eps = dim,eps
    def forward(self, x1, x2): return cosine_similarity(x1,x2,self.dim,self.eps)


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        if inplace: raise _errors.BrowserUnsupported('In-place ReLU requires the server')
    def forward(self, input): return input.relu()


def layer_norm_oracle(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return _normalization.layer_norm(x,normalized_shape,weight,bias,eps)


def rms_norm_oracle(x, normalized_shape, weight=None, eps=None):
    return _normalization.rms_norm(x,normalized_shape,weight,eps)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Native cosine normalization, including its distinct reverse/forward epsilon rules."""
    if not isinstance(x1, _tensor.Tensor) or not isinstance(x2, _tensor.Tensor):
        raise TypeError('cosine_similarity expects two tensors')
    if not isinstance(dim, (int, np.integer)) or isinstance(dim, (bool, np.bool_)):
        raise TypeError('cosine_similarity dim must be an integer')
    if not isinstance(eps, (int, float, np.integer, np.floating)):
        raise TypeError('cosine_similarity eps must be a number')
    if not eps >= 0: raise RuntimeError('cosine_similarity eps must be nonnegative')
    if any(x.dtype.kind == 'c' for x in (x1,x2)):
        raise RuntimeError('cosine_similarity requires a real floating result dtype')
    if any(x.dtype.kind == 'f' and x.dtype not in (np.dtype('float32'),np.dtype('float64')) for x in (x1,x2)):
        raise _errors.BrowserUnsupported('cosine_similarity supports real float32/float64')
    floating = [x.dtype for x in (x1,x2) if x.dtype.kind == 'f']
    if not floating: raise RuntimeError('cosine_similarity requires a floating result dtype')
    dtype = np.result_type(*floating)
    try: shape = np.broadcast_shapes(x1.shape,x2.shape)
    except ValueError as error: raise RuntimeError('cosine_similarity inputs are not broadcastable') from error
    _ops.reduction_axes(dim,len(shape))
    axis = dim if shape else None
    forward = isinstance(x1,_autograd.ForwardTensor) or isinstance(x2,_autograd.ForwardTensor)
    def normalized(input):
        primal,tangent = _autograd.forward_parts(input) if forward else (input,None)
        if primal.dtype.kind != 'f':
            primal = primal.to(dtype=dtype)
            if forward: tangent = tangent.to(dtype=dtype)
        primal = primal.expand(shape)
        length = primal.norm(2,dim,keepdim=True)
        # Native clamps cloned norm values under no_grad: the reverse edge is
        # still identity below eps, unlike F.normalize's differentiable clamp.
        denominator = _autograd.result(np.maximum(length._data,eps),(length,lambda g:g))
        unit = primal / denominator
        if not forward: return unit
        tangent = tangent.expand(shape)
        with np.errstate(divide='ignore',invalid='ignore'):
            norm_tangent = _ops.where(length == 0,0.,(primal*tangent).sum(axis,True)/length)
        # Forward AD through that same clamp is zero below eps and half at a
        # tie. Its tangent value is also changed under no_grad, retaining the
        # identity reverse edge for reverse-over-forward differentiation.
        clipped = np.where(length._data<eps,0.,np.where(length._data>eps,norm_tangent._data,.5*norm_tangent._data))
        denominator_tangent = _autograd.result(clipped,(norm_tangent,lambda g:g))
        return unit,(tangent-unit*denominator_tangent)/denominator
    if forward:
        a,da = normalized(x1);b,db = normalized(x2)
        return _autograd.ForwardTensor((a*b).sum(axis),(da*b+a*db).sum(axis))
    return (normalized(x1)*normalized(x2)).sum(axis)


def normalize(input, p=2., dim=1, eps=1e-12, out=None):
    if p != 2 or out is not None: raise _errors.BrowserUnsupported('Only Euclidean normalization without out is supported')
    return input / input.norm(2, dim, keepdim=True).clamp_min(eps)


def softplus(input, beta=1., threshold=20.):
    if not isinstance(input,_tensor.Tensor): raise TypeError('Softplus expects a tensor')
    if input.dtype.kind != 'f': raise RuntimeError('Softplus requires floating inputs')
    if input.dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise _errors.BrowserUnsupported('Softplus requires float32/float64 inputs')
    if any(isinstance(value,_tensor.Tensor) for value in (beta,threshold)):
        raise _errors.BrowserUnsupported('Tensor-valued Softplus parameters require the server')
    if any(not isinstance(value,(int,float,np.integer,np.floating)) for value in (beta,threshold)):
        raise TypeError('Softplus beta and threshold must be numbers')
    forward = isinstance(input,_autograd.ForwardTensor)
    primal = input._primal if forward else input
    # CPU Softplus uses log1p(exp(beta*x)) unless beta*x is strictly above
    # threshold. Preserve its overflow/zero-beta behavior, including NaNs in
    # first derivatives when the caller disables that stability branch.
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        coefficient = np.asarray(beta,dtype=primal.dtype)
        cutoff = np.asarray(threshold,dtype=primal.dtype)
        scaled = coefficient * primal._data
        exponential = np.exp(scaled)
        value = np.where(scaled>cutoff,primal._data,np.log1p(exponential)/coefficient)
        denominator = exponential+1
        probability = 1 / (1 + np.exp(-scaled))
        # The native double-backward rule is zero at the threshold itself,
        # although the forward/first derivative use the nonlinear branch there.
        curvature = probability * (1-probability) * (scaled<cutoff) * coefficient
    def backward(gradient):
        # Native CPU multiplies the incoming gradient before dividing by z+1;
        # precomputing a bounded slope would hide valid overflow behavior.
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            return np.where(scaled>cutoff,gradient,(gradient*exponential)/denominator)
    output = _autograd.result(value,(primal,backward))
    if not forward: return output
    tangent = input._tangent
    tangent_data = tangent._data
    derivative = _autograd.result(backward(tangent_data),
                        (primal,lambda g:g*tangent_data*curvature),
                        (tangent,backward))
    return _autograd.ForwardTensor(output,derivative)


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.):
    """Independent raw NumPy loss and analytical derivatives for both inputs."""
    if size_average is not None or reduce is not None or input.shape != target.shape:
        raise _errors.BrowserUnsupported('Legacy or broadcast smooth L1 requires the server')
    if input.dtype != target.dtype or input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('Smooth L1 requires matching float32/float64 tensors')
    if not np.isfinite(beta) or beta < 0: raise ValueError('beta must be finite and nonnegative')
    if reduction not in ('none', 'sum', 'mean'): raise ValueError('Invalid smooth L1 reduction')
    difference = input._data - target._data
    absolute = np.abs(difference)
    if beta == 0:
        losses, derivative = absolute, np.sign(difference)
    else:
        quadratic = absolute < beta
        losses = np.where(quadratic, .5 * difference ** 2 / beta, absolute - .5 * beta)
        derivative = np.where(quadratic, difference / beta, np.sign(difference))
    count = difference.size
    value = losses if reduction == 'none' else losses.sum() if reduction == 'sum' else losses.mean() if count else np.asarray(np.nan, dtype=input.dtype)
    def vjp(g): return derivative * g / (max(count, 1) if reduction == 'mean' else 1)
    return _autograd.result(np.asarray(value, dtype=input.dtype), (input, vjp), (target, lambda g: -vjp(g)))


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None or input.shape != target.shape:
        raise _errors.BrowserUnsupported('Legacy or broadcast MSE requires the server')
    if input.dtype != target.dtype or input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('MSE requires matching float32/float64 tensors')
    if reduction not in ('none', 'sum', 'mean'): raise ValueError('Invalid MSE reduction')
    losses = (input - target).square()
    return losses if reduction == 'none' else losses.sum() if reduction == 'sum' else losses.mean()


def kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False):
    """Raw NumPy KL values and VJPs, including the target distribution graph."""
    if size_average is not None or reduce is not None:
        raise _errors.BrowserUnsupported('Legacy KL divergence options require the server')
    if not isinstance(input, _tensor.Tensor) or not isinstance(target, _tensor.Tensor):
        raise TypeError('KL divergence expects Tensor inputs and targets')
    if input.dtype != target.dtype or input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('KL divergence requires matching float32/float64 tensors')
    if type(log_target) is not bool: raise TypeError('log_target must be a boolean')
    if reduction not in ('none', 'sum', 'mean', 'batchmean'): raise ValueError('Invalid KL divergence reduction')
    try: shape = np.broadcast_shapes(input.shape, target.shape)
    except ValueError: raise RuntimeError('KL divergence input and target cannot broadcast') from None
    divisor = int(np.prod(shape)) if reduction == 'mean' else input.shape[0] if reduction == 'batchmean' and input.ndim else 1
    if isinstance(input, _autograd.ForwardTensor) or isinstance(target, _autograd.ForwardTensor):
        x, dx = _autograd.forward_parts(input); y, dy = _autograd.forward_parts(target)
        with np.errstate(over='ignore'):
            valid_target = np.isfinite(np.exp(y._data)).all() if log_target else (y._data > 0).all()
        if not valid_target or not np.isfinite(x._data).all() or not np.isfinite(y._data).all():
            raise _errors.BrowserUnsupported('Nonfinite or nonpositive-probability KL JVP requires the server')
        primal = kl_div(x, y, reduction=reduction, log_target=log_target)
        tangent = y.exp() * ((y - x + 1) * dy - dx) if log_target else (y.log() + 1 - x) * dy - y * dx
        tangent = tangent if reduction == 'none' else tangent.sum() / divisor
        return _autograd.ForwardTensor(primal, tangent)
    x, y = input._data, target._data
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        if log_target:
            probabilities = np.exp(y)
            losses = probabilities * (y - x)
            input_gradient = -probabilities
            target_gradient = probabilities * (y - x) + probabilities
        else:
            logarithm = np.log(y)
            # Native xlogy(0, 0) is zero, while target gradients at zero remain
            # undefined. Keep 0*Inf/NaN from the separate target*input term.
            losses = np.where(y == 0, np.zeros((), dtype=input.dtype), y * logarithm) - y * x
            input_gradient = -y
            target_gradient = logarithm + y / y - x
        value = losses if reduction == 'none' else losses.sum() / divisor
    def vjp(gradient, g, operand):
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            return _ops.unbroadcast(np.broadcast_to(gradient, shape) * g / divisor, operand.shape)
    return _autograd.result(np.asarray(value, dtype=input.dtype),
                  (input, lambda g: vjp(input_gradient, g, input)),
                  (target, lambda g: vjp(target_gradient, g, target)))


class KLDivLoss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean', log_target=False):
        super().__init__()
        if size_average is not None or reduce is not None:
            raise _errors.BrowserUnsupported('Legacy KL divergence options require the server')
        self.reduction, self.log_target = reduction, log_target

    def forward(self, input, target):
        return kl_div(input, target, reduction=self.reduction, log_target=self.log_target)


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean', label_smoothing=0.):
    return _classification.cross_entropy(input, target, weight, size_average, ignore_index,
                                         reduce, reduction, label_smoothing)


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='mean', pos_weight=None):
    """Stable raw NumPy values and analytical derivatives for logits and targets."""
    if size_average is not None or reduce is not None:
        raise _errors.BrowserUnsupported('Legacy binary cross entropy options require the server')
    if not isinstance(input, _tensor.Tensor) or not isinstance(target, _tensor.Tensor):
        raise TypeError('Binary cross entropy expects Tensor inputs and targets')
    if input.shape != target.shape: raise ValueError('Target and input shapes must match')
    if input.dtype != target.dtype or input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('Binary cross entropy requires matching float32/float64 tensors')
    if reduction not in ('none', 'sum', 'mean'): raise ValueError('Invalid binary cross entropy reduction')
    for name, tensor_value in (('weight', weight), ('pos_weight', pos_weight)):
        if tensor_value is None: continue
        if not isinstance(tensor_value, _tensor.Tensor): raise TypeError(f'{name} must be a Tensor')
        if isinstance(tensor_value, _autograd.ForwardTensor) or tensor_value.dtype != input.dtype:
            raise _errors.BrowserUnsupported('Differentiated or mixed-dtype binary loss weights require the server')
        if _state._grad_enabled and tensor_value.requires_grad:
            raise _errors.BrowserUnsupported('Binary loss weight gradients require the server')
        try: np.broadcast_to(tensor_value._data, input.shape)
        except ValueError: raise RuntimeError(f'{name} cannot broadcast to the input shape') from None
    if isinstance(input, _autograd.ForwardTensor) or isinstance(target, _autograd.ForwardTensor):
        x, dx = _autograd.forward_parts(input); y, dy = _autograd.forward_parts(target)
        if any(not np.isfinite(v._data).all() for v in (x, y, weight, pos_weight) if v is not None):
            raise _errors.BrowserUnsupported('Nonfinite binary-loss JVP requires the server')
        primal = binary_cross_entropy_with_logits(x, y, weight, reduction=reduction, pos_weight=pos_weight)
        positive = 1 if pos_weight is None else pos_weight
        target_gradient = -x if pos_weight is None else (-x).logsigmoid() - positive * x.logsigmoid()
        tangent = dx * ((positive * y + 1 - y) * x.sigmoid() - positive * y) + dy * target_gradient
        if weight is not None: tangent = tangent * weight
        tangent = tangent if reduction == 'none' else tangent.sum() if reduction == 'sum' else tangent.mean()
        return _autograd.ForwardTensor(primal, tangent)
    x, y = input._data, target._data
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        log_positive = -np.logaddexp(np.zeros((), dtype=input.dtype), -x)
        probability = np.exp(-np.logaddexp(np.zeros((), dtype=input.dtype), -x))
        if pos_weight is None:
            losses = (1 - y) * x - log_positive
            input_gradient, target_gradient = probability - y, -x
        else:
            positive = pos_weight._data
            losses = (1 - y) * x - (1 + (positive - 1) * y) * log_positive
            weighted_target = positive * y
            input_gradient = (weighted_target + 1 - y) * probability - weighted_target
            target_gradient = -np.logaddexp(np.zeros((), dtype=input.dtype), x) - positive * log_positive
        if weight is not None:
            losses = losses * weight._data
            input_gradient, target_gradient = input_gradient * weight._data, target_gradient * weight._data
        count = x.size
        value = losses if reduction == 'none' else losses.sum() if reduction == 'sum' else losses.mean() if count else np.asarray(np.nan, dtype=input.dtype)
    def vjp(gradient, g):
        with np.errstate(invalid='ignore', over='ignore'):
            return gradient * g / (max(count, 1) if reduction == 'mean' else 1)
    return _autograd.result(np.asarray(value, dtype=input.dtype),
                  (input, lambda g: vjp(input_gradient, g)), (target, lambda g: vjp(target_gradient, g)),
                  saved=tuple(v for v in (weight, pos_weight) if v is not None))


class BCEWithLogitsLoss(Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super().__init__()
        if size_average is not None or reduce is not None:
            raise _errors.BrowserUnsupported('Legacy binary cross entropy options require the server')
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return binary_cross_entropy_with_logits(input, target, self.weight, reduction=self.reduction, pos_weight=self.pos_weight)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False):
    if sparse: raise _errors.BrowserUnsupported('Sparse embedding gradients require the server')
    if input.dtype not in (np.dtype('int32'),np.dtype('int64')) or weight.ndim!=2:
        raise RuntimeError('Embedding expects integer indices and a matrix codebook')
    if weight.dtype not in (np.dtype('float32'),np.dtype('float64')): raise _errors.BrowserUnsupported('Embedding requires real float32/float64 weights')
    if not isinstance(scale_grad_by_freq,bool): raise TypeError('scale_grad_by_freq must be boolean')
    n=weight.shape[0]
    if padding_idx is not None:
        if type(padding_idx) is not int or not -n<=padding_idx<n: raise RuntimeError('Invalid padding index')
        padding_idx%=n
    indices=input._data
    weight_shape, weight_dtype = weight.shape, weight.dtype
    if np.any(indices<0) or np.any(indices>=n): raise IndexError('Embedding index out of bounds')
    if max_norm is not None:
        if _state._jvp_active:raise _errors.BrowserUnsupported('Embedding renormalization during JVP is unsupported')
        if isinstance(max_norm,_tensor.Tensor) or isinstance(norm_type,_tensor.Tensor):
            raise _errors.BrowserUnsupported('Tensor embedding norm parameters are unsupported')
        if not np.isscalar(max_norm) or not np.isscalar(norm_type):raise TypeError('Embedding norm parameters must be numbers')
        rows=np.unique(indices).astype(np.intp,copy=False)
        selected=weight._data[rows];absolute=np.abs(selected)
        if rows.size and not weight.shape[1] and abs(norm_type)==math.inf:
            raise RuntimeError('Infinite embedding norms require nonempty feature dimensions')
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            if not rows.size:norms=np.empty(0,dtype=weight_dtype)
            elif norm_type==0:norms=np.count_nonzero(selected,axis=1).astype(weight_dtype)
            elif norm_type==math.inf:norms=absolute.max(axis=1)
            elif norm_type==-math.inf:norms=absolute.min(axis=1)
            else:norms=np.sum(absolute**norm_type,axis=1)**(1/norm_type)
            active=norms>max_norm
            if active.any():
                factor=np.asarray(max_norm,dtype=weight_dtype)/(norms[active]+np.asarray(1e-7,dtype=weight_dtype))
                with _autograd.no_grad():weight[rows[active]]=selected[active]*factor[:,None]
            else:weight._storage.bump()
    if isinstance(weight,_autograd.ForwardTensor):
        primal,tangent=_autograd.forward_parts(weight)
        return _autograd.ForwardTensor(embedding(input,primal,padding_idx,None,norm_type,scale_grad_by_freq),
                                      embedding(input,tangent,padding_idx,None,norm_type,scale_grad_by_freq))
    def vjp(g):
        gradient=np.zeros(weight_shape, dtype=weight_dtype)
        flat=indices.reshape(-1).astype(np.intp,copy=False); incoming=g.reshape(input.numel(),weight.shape[1])
        if scale_grad_by_freq:
            counts=np.bincount(flat,minlength=n)
            incoming=incoming/counts[flat,None].astype(weight_dtype)
        if padding_idx is not None:
            valid=flat!=padding_idx;flat=flat[valid];incoming=incoming[valid]
        np.add.at(gradient,flat,incoming)
        return gradient
    return _autograd.result(weight._data[indices],(weight,vjp),saved=(input,))


def _spatial_pair(value, name, current=None):
    pair=(value,value) if isinstance(value,(int,np.integer)) else tuple(value) if isinstance(value,(tuple,list)) else ()
    if len(pair)!=2: raise _errors.BrowserUnsupported(f'{name} requires two spatial dimensions')
    pair=tuple(current[i] if x is None and current is not None else x for i,x in enumerate(pair))
    if any(not isinstance(x,(int,np.integer)) or isinstance(x,(bool,np.bool_)) or x<=0 for x in pair):
        raise RuntimeError(f'{name} requires positive integer sizes')
    return tuple(int(x) for x in pair)


def _resize_matrix(length, output, dtype, *, cubic=False, align=False, factor=None):
    weights=np.zeros((output,length),dtype=dtype)
    if not cubic:
        for i in range(output):
            start=i*length//output;end=((i+1)*length+output-1)//output
            weights[i,start:end]=1/(end-start)
        return weights
    scalar=dtype.type
    scale=scalar((length-1)/(output-1) if align and output>1 else 0 if align else 1/factor if factor is not None else length/output)
    position=np.arange(output,dtype=dtype)*scale if align else (np.arange(output,dtype=dtype)+scalar(.5))*scale-scalar(.5)
    base=np.floor(position).astype(np.int64);t=position-base.astype(dtype)
    a=scalar(-.75)
    def inner(x):return ((a+2)*x-(a+3))*x*x+1
    def outer(x):return ((a*x-5*a)*x+8*a)*x-4*a
    coefficients=(outer(t+1),inner(t),inner(1-t),outer(2-t))
    for offset,coefficient in zip((-1,0,1,2),coefficients):
        np.add.at(weights,(np.arange(output),np.clip(base+offset,0,length-1)),coefficient)
    return weights


def _resize_2d(input, size, *, cubic=False, align=False, factors=None):
    if input.ndim not in (3,4) or min(input.shape[-3:])<=0: raise RuntimeError('2D resize expects nonempty channel and spatial dimensions')
    if input.dtype not in (np.dtype('float32'),np.dtype('float64')): raise _errors.BrowserUnsupported('2D resize requires real float32/float64')
    h,w=input.shape[-2:];oh,ow=size
    planes=math.prod(input.shape[:-2])
    if h*oh+w*ow>4_000_000 or planes*max(oh*ow,min(oh*w,h*ow))>8_000_000:
        raise _errors.BrowserUnsupported('This resize allocation requires the server')
    wy=_resize_matrix(h,oh,input.dtype,cubic=cubic,align=align,factor=factors[0] if factors else None)
    wx=_resize_matrix(w,ow,input.dtype,cubic=cubic,align=align,factor=factors[1] if factors else None)
    # Choose the smaller intermediate for each direction, including thin maps.
    value=(wy @ input._data) @ wx.T if oh*w<=h*ow else wy @ (input._data @ wx.T)
    def vjp(g):return (wy.T @ g) @ wx if h*ow<=oh*w else wy.T @ (g @ wx)
    return _autograd.result(value,(input,vjp),saves_tensors=not cubic and size!=(1,1))


def adaptive_avg_pool2d(input, output_size):
    return _average_pooling.adaptive_avg_pool2d(input,output_size)


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if input.ndim!=4 or mode not in ('area','bicubic') or antialias:
        raise _errors.BrowserUnsupported('Interpolation supports non-antialiased 2D area/bicubic modes')
    if mode=='area' and align_corners is not None: raise ValueError('align_corners requires an interpolating mode')
    if align_corners is not None and not isinstance(align_corners,bool): raise TypeError('align_corners must be boolean')
    if recompute_scale_factor is not None and not isinstance(recompute_scale_factor,bool): raise TypeError('recompute_scale_factor must be boolean')
    if (size is None)==(scale_factor is None): raise ValueError('Specify size or scale_factor')
    factors=None
    if size is not None:
        if recompute_scale_factor: raise ValueError('recompute_scale_factor requires scale_factor')
        shape=_spatial_pair(size,'interpolate')
    else:
        factors=(scale_factor,scale_factor) if np.isscalar(scale_factor) else tuple(scale_factor)
        if len(factors)!=2 or any(not np.isscalar(x) or not np.isfinite(x) or x<=0 for x in factors): raise ValueError('Invalid scale factors')
        shape=tuple(math.floor(n*f) for n,f in zip(input.shape[-2:],factors))
        shape=_spatial_pair(shape,'interpolate')
        if recompute_scale_factor or mode=='area':factors=None
    if mode=='area':return adaptive_avg_pool2d(input,shape)
    if isinstance(input,_autograd.ForwardTensor):
        raise _errors.BrowserUnsupported('Bicubic interpolation forward AD requires the server')
    return _resize_2d(input,shape,cubic=True,align=bool(align_corners),factors=factors)


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _autograd, _average_pooling, _classification, _device, _errors, _normalization, _ops, _random, _state

from ._attention import scaled_dot_product_attention as scaled_dot_product_attention_oracle
