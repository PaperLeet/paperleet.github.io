"""GELU and dropout without native Torch or optional numerical dependencies."""
import math
import numbers
import numpy as np

from . import _autograd, _errors, _random, _tensor
from ._module import Module


def _gelu_terms(data, approximate, *, curvature=False):
    dtype = data.dtype.type
    half,one,two,three,six = (dtype(v) for v in (.5,1,2,3,6))
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        if approximate=='none':
            erf = np.asarray(np.frompyfunc(math.erf,1,1)(data*dtype(1/math.sqrt(2))),dtype=data.dtype)
            density = np.exp(-half*data*data)*dtype(1/math.sqrt(2*math.pi))
            cumulative = half*(one+erf)
            return data*cumulative, cumulative+data*density, density*(two-data*data) if curvature else None
        coefficient, cubic = dtype(math.sqrt(2/math.pi)), dtype(.044715)
        tangent = np.tanh(coefficient*(data+cubic*(data*data*data)))
        slope = coefficient*(one+three*cubic*data*data)
        delta = one-tangent*tangent
        value = half*data*(one+tangent)
        derivative = half*(one+tangent)+half*data*delta*slope
        second = delta*(slope+half*data*(six*coefficient*cubic*data-two*tangent*slope*slope)) if curvature else None
        return value, derivative, second


def gelu(input, approximate='none'):
    if not isinstance(input,_tensor.Tensor):raise TypeError('GELU expects a Tensor')
    if input.dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise _errors.BrowserUnsupported('GELU requires real float32/float64')
    if approximate not in ('none','tanh'):raise RuntimeError('GELU approximate must be none or tanh')
    forward = isinstance(input,_autograd.ForwardTensor)
    source = input._primal if forward else input
    data = source._data
    value, slope, curvature = _gelu_terms(data,approximate,curvature=forward)
    def backward(g):
        if source._data is not data:
            raise _errors.BrowserUnsupported('GELU backward after storage rebinding is unsupported')
        return g*slope
    output = _autograd.result(value,(source,backward))
    if not forward:return output
    tangent = input._tangent
    derivative = _autograd.result(tangent._data*slope,
                  (source,lambda g:g*tangent._data*curvature),(tangent,backward))
    return _autograd.ForwardTensor(output,derivative)


def dropout(input, p=.5, training=True, inplace=False):
    if not isinstance(input,_tensor.Tensor):raise TypeError('Dropout expects a Tensor')
    if not isinstance(p,numbers.Real):raise TypeError('Dropout p must be a number')
    if not 0<=p<=1:raise ValueError('Dropout p must be in [0,1]')
    if type(training) is not bool:raise TypeError('Dropout training must be bool')
    if type(inplace) is not bool:raise TypeError('Dropout inplace must be bool')
    if p==0 or not training or not input.numel():return input
    if inplace and (input.requires_grad or isinstance(input,_autograd.ForwardTensor)):
        raise _errors.BrowserUnsupported('In-place differentiable dropout is unsupported')
    if p==1:
        if input.dtype.kind=='b':return input.zero_() if inplace else _tensor.Tensor(np.zeros_like(input._data))
        with np.errstate(invalid='ignore'):
            if inplace:return input.mul_(0)
            return input*0
    if input.dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise _errors.BrowserUnsupported('Training dropout requires float32/float64')
    # Distribution-faithful Bernoulli masks; this is not native Torch's seeded
    # dropout bit stream. Eval/zero/one/empty paths do not consume RNG draws.
    mask = (_random.uniform_samples(np.random,input.shape,input.dtype)>=p).astype(input.dtype)
    mask /= np.asarray(1-p,dtype=input.dtype)
    if inplace:
        input[...] = input._data*mask
        return input
    if isinstance(input,_autograd.ForwardTensor):
        multiplier = _tensor.Tensor(mask)
        return _autograd.ForwardTensor(input._primal*multiplier,input._tangent*multiplier)
    return _autograd.result(input._data*mask,(input,lambda g:g*mask))


class GELU(Module):
    def __init__(self, approximate='none'):
        super().__init__();self.approximate = approximate
    def forward(self,input):return gelu(input,self.approximate)
    def extra_repr(self):return f'approximate={self.approximate!r}'


class Dropout(Module):
    def __init__(self,p=.5,inplace=False):
        super().__init__()
        if not isinstance(p,numbers.Real) or not 0<=p<=1:raise ValueError('Dropout p must be in [0,1]')
        self.p,self.inplace = p,inplace
    def forward(self,input):return dropout(input,self.p,self.training,self.inplace)
    def extra_repr(self):return f'p={self.p}, inplace={self.inplace}'
