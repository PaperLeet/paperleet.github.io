"""Compatibility facade for the original PaperLeet backend API."""
from ._errors import BrowserUnsupported, unsupported
from ._tensor import StorageVersion, storage_version, Tensor
from ._autograd import result, backprop, ForwardTensor, forward_parts, jvp, autograd_grad, gradcheck, no_grad
from ._device import check_device, Device
from ._random import fork_rng, Generator, _default_generator, normal_samples, multinomial, randperm, randint, uniform_samples, create
from ._linalg import solve, inv, real_matrix, svd, eigh, eigvalsh, det, qr, linalg_norm
from ._nn import Parameter, Module, linear, silu, Linear, Sequential, SiLU, Softplus, LogSigmoid, CosineSimilarity, ReLU, scaled_dot_product_attention_oracle, layer_norm_oracle, rms_norm_oracle, cosine_similarity, normalize, softplus, smooth_l1_loss, mse_loss, kl_div, KLDivLoss, cross_entropy, binary_cross_entropy_with_logits, BCEWithLogitsLoss, embedding, _spatial_pair, _resize_matrix, _resize_2d, adaptive_avg_pool2d, interpolate
from ._optim import SGD, Adam, AdamW
from ._registry import OperatorCapability, contains_tensor, OperatorRegistry
from ._ops import unbroadcast, wrap, inherited_options, reduction_axes, power, count_nonzero, where, tensor, as_tensor, from_numpy, nan_to_num, elementwise_extreme, trapezoid, bincount, assert_close, einsum, concatenate, cov, full, full_like, cdist, pdist, meshgrid, cartesian_prod, unique
from ._integration import install
from . import _state

__all__ = ['Adam', 'AdamW', 'BCEWithLogitsLoss', 'BrowserUnsupported', 'CosineSimilarity', 'Device', 'ForwardTensor', 'Generator', 'KLDivLoss', 'Linear', 'LogSigmoid', 'Module', 'OperatorCapability', 'OperatorRegistry', 'Parameter', 'ReLU', 'SGD', 'Sequential', 'SiLU', 'Softplus', 'StorageVersion', 'Tensor', '_default_generator', '_resize_2d', '_resize_matrix', '_spatial_pair', 'adaptive_avg_pool2d', 'as_tensor', 'assert_close', 'autograd_grad', 'backprop', 'binary_cross_entropy_with_logits', 'bincount', 'cartesian_prod', 'cdist', 'check_device', 'concatenate', 'contains_tensor', 'cosine_similarity', 'count_nonzero', 'cov', 'create', 'cross_entropy', 'det', 'eigh', 'eigvalsh', 'einsum', 'elementwise_extreme', 'embedding', 'fork_rng', 'forward_parts', 'from_numpy', 'full', 'full_like', 'gradcheck', 'inherited_options', 'interpolate', 'inv', 'jvp', 'kl_div', 'layer_norm_oracle', 'linalg_norm', 'linear', 'meshgrid', 'mse_loss', 'multinomial', 'nan_to_num', 'no_grad', 'normal_samples', 'normalize', 'pdist', 'power', 'qr', 'randint', 'randperm', 'real_matrix', 'reduction_axes', 'result', 'rms_norm_oracle', 'scaled_dot_product_attention_oracle', 'silu', 'smooth_l1_loss', 'softplus', 'solve', 'storage_version', 'svd', 'tensor', 'trapezoid', 'unbroadcast', 'uniform_samples', 'unique', 'unsupported', 'where', 'wrap', 'install']

def __getattr__(name):
    if name in ('_grad_enabled', '_jvp_active', '_jvp_level', '_storages'):
        return getattr(_state, name)
    raise AttributeError(name)
