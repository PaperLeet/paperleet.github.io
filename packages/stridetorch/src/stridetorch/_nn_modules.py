"""Registered embeddings, losses and shape/activation modules."""
import numbers
import numpy as np

from . import _autograd, _errors, _normalization, _random, _tensor
from ._module import Module,Parameter


class Embedding(Module):
    def __init__(self,num_embeddings,embedding_dim,padding_idx=None,max_norm=None,norm_type=2.,
                 scale_grad_by_freq=False,sparse=False,_weight=None,_freeze=False,device=None,dtype=None):
        super().__init__()
        if any(not isinstance(n,numbers.Integral) or isinstance(n,(bool,np.bool_)) for n in (num_embeddings,embedding_dim)):
            raise TypeError('Embedding dimensions must be integers')
        if min(num_embeddings,embedding_dim)<0:raise RuntimeError('Negative embedding dimensions')
        self.num_embeddings,self.embedding_dim = int(num_embeddings),int(embedding_dim)
        if padding_idx is not None:
            if not isinstance(padding_idx,numbers.Integral) or not -num_embeddings<=padding_idx<num_embeddings:
                raise AssertionError('padding_idx must be within num_embeddings')
            padding_idx = int(padding_idx)%num_embeddings
        if sparse:raise _errors.BrowserUnsupported('Sparse embedding gradients are unsupported')
        self.padding_idx,self.max_norm,self.norm_type = padding_idx,max_norm,norm_type
        self.scale_grad_by_freq,self.sparse = scale_grad_by_freq,sparse
        if _weight is None:
            dtype = _normalization.parameter_dtype(dtype,device)
            self.weight = Parameter(_tensor.Tensor(np.empty((num_embeddings,embedding_dim),dtype=dtype)),requires_grad=not _freeze)
            self.reset_parameters()
        else:
            if not isinstance(_weight,_tensor.Tensor) or _weight.shape!=(num_embeddings,embedding_dim):
                raise AssertionError('Embedding _weight shape must match its dimensions')
            if _weight.dtype not in (np.dtype('float32'),np.dtype('float64')):
                raise _errors.BrowserUnsupported('Embedding weights require float32/float64')
            _normalization.parameter_dtype(_weight.dtype,device)
            self.weight = Parameter(_weight,requires_grad=not _freeze)

    def reset_parameters(self):
        with _autograd.no_grad():
            self.weight[...] = _random.normal_samples(_random._default_generator,self.weight.shape,self.weight.dtype)
            if self.padding_idx is not None:self.weight[self.padding_idx] = 0

    @classmethod
    def from_pretrained(cls,embeddings,freeze=True,padding_idx=None,max_norm=None,norm_type=2.,scale_grad_by_freq=False,sparse=False):
        if not isinstance(embeddings,_tensor.Tensor) or embeddings.ndim!=2:
            raise AssertionError('Pretrained embeddings must be a matrix')
        return cls(*embeddings.shape,padding_idx,max_norm,norm_type,scale_grad_by_freq,sparse,embeddings,freeze)

    def forward(self,input):
        return _nn.embedding(input,self.weight,self.padding_idx,self.max_norm,self.norm_type,self.scale_grad_by_freq,self.sparse)

    def extra_repr(self):
        result=f'{self.num_embeddings}, {self.embedding_dim}'
        for name,default in (('padding_idx',None),('max_norm',None),('norm_type',2.),('scale_grad_by_freq',False),('sparse',False)):
            if getattr(self,name)!=default:result+=f', {name}={getattr(self,name)}'
        return result


class Identity(Module):
    def __init__(self,*args,**kwargs):super().__init__()
    def forward(self,input):return input


class Flatten(Module):
    def __init__(self,start_dim=1,end_dim=-1):
        super().__init__();self.start_dim,self.end_dim = start_dim,end_dim
    def forward(self,input):return input.flatten(self.start_dim,self.end_dim)
    def extra_repr(self):return f'start_dim={self.start_dim}, end_dim={self.end_dim}'


class Tanh(Module):
    def forward(self,input):return input.tanh()


class Sigmoid(Module):
    def forward(self,input):return input.sigmoid()


class MSELoss(Module):
    def __init__(self,size_average=None,reduce=None,reduction='mean'):
        super().__init__()
        if size_average is not None or reduce is not None:raise _errors.BrowserUnsupported('Legacy MSE loss options are unsupported')
        self.reduction = reduction
    def forward(self,input,target):return _nn.mse_loss(input,target,reduction=self.reduction)


class SmoothL1Loss(Module):
    def __init__(self,size_average=None,reduce=None,reduction='mean',beta=1.):
        super().__init__()
        if size_average is not None or reduce is not None:raise _errors.BrowserUnsupported('Legacy smooth L1 options are unsupported')
        self.reduction,self.beta = reduction,beta
    def forward(self,input,target):return _nn.smooth_l1_loss(input,target,reduction=self.reduction,beta=self.beta)


class CrossEntropyLoss(Module):
    def __init__(self,weight=None,size_average=None,ignore_index=-100,reduce=None,reduction='mean',label_smoothing=0.):
        super().__init__()
        if size_average is not None or reduce is not None:raise _errors.BrowserUnsupported('Legacy cross entropy options are unsupported')
        self.register_buffer('weight',weight)
        self.ignore_index,self.reduction,self.label_smoothing = ignore_index,reduction,label_smoothing
    def forward(self,input,target):
        return _nn.cross_entropy(input,target,self.weight,ignore_index=self.ignore_index,
                                 reduction=self.reduction,label_smoothing=self.label_smoothing)


from . import _nn
