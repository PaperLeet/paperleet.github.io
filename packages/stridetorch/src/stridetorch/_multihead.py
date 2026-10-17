"""Multihead attention module and functional math path on validated tensor ops."""
import math
import numbers
import warnings
import numpy as np

from . import _activation, _attention, _autograd, _errors, _nn, _ops, _random, _tensor
from ._module import Module, Parameter
from ._normalization import parameter_dtype


def _positive(value, name):
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer')
    if value <= 0:
        raise ValueError(f'{name} must be greater than zero')
    return int(value)


def _tensor_argument(value, name, shape=None, dtype=None, optional=False):
    if value is None and optional:
        return
    if not isinstance(value, _tensor.Tensor):
        raise TypeError(f'{name} must be a Tensor')
    if shape is not None and value.shape != shape:
        raise AssertionError(f'{name} must have shape {shape}, got {value.shape}')
    if dtype is not None and value.dtype != dtype:
        raise RuntimeError(f'{name} must match the query dtype')


def _mask_kind(mask, name):
    if mask is None:
        return None
    _tensor_argument(mask, name)
    if mask.dtype.kind not in ('b', 'f'):
        raise AssertionError(f'Only boolean and floating {name} are supported')
    return mask.dtype


def _canonical_mask(mask, dtype):
    if mask is not None and mask.dtype.kind == 'b':
        return _ops.where(mask, _tensor.Tensor(-np.inf, dtype=dtype), _tensor.Tensor(0., dtype=dtype))
    return mask


def _extend_mask(mask):
    if mask is None:
        return None
    return _ops.concatenate([mask, _tensor.Tensor(np.zeros((*mask.shape[:-1], 1), dtype=mask.dtype))], -1)


def _projections(query, key, value, embed, weight, bias, separate, qw, kw, vw):
    if separate:
        for name, parameter, shape in [('q_proj_weight', qw, (embed, query.shape[-1])), ('k_proj_weight', kw, (embed, key.shape[-1])), ('v_proj_weight', vw, (embed, value.shape[-1]))]:
            _tensor_argument(parameter, name, shape, query.dtype)
        pieces = [None] * 3 if bias is None else [bias[i * embed:(i + 1) * embed] for i in range(3)]
        return tuple(_nn.linear(x, w, b) for x, w, b in zip((query, key, value), (qw, kw, vw), pieces))
    _tensor_argument(weight, 'in_proj_weight', (3 * embed, embed), query.dtype)
    if query is key and key is value:
        packed = _nn.linear(query, weight, bias)
        return tuple(packed[..., i * embed:(i + 1) * embed] for i in range(3))
    qb = None if bias is None else bias[:embed]
    q = _nn.linear(query, weight[:embed], qb)
    if key is value:
        packed = _nn.linear(key, weight[embed:], None if bias is None else bias[embed:])
        return q, packed[..., :embed], packed[..., embed:]
    return (q, _nn.linear(key, weight[embed:2 * embed], None if bias is None else bias[embed:2 * embed]),
            _nn.linear(value, weight[2 * embed:], None if bias is None else bias[2 * embed:]))


def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads,
                                 in_proj_weight, in_proj_bias, bias_k, bias_v,
                                 add_zero_attn, dropout_p, out_proj_weight, out_proj_bias,
                                 training=True, key_padding_mask=None, need_weights=True,
                                 attn_mask=None, use_separate_proj_weight=False,
                                 q_proj_weight=None, k_proj_weight=None, v_proj_weight=None,
                                 static_k=None, static_v=None, average_attn_weights=True,
                                 is_causal=False):
    """Sequence-first math MHA; native fused inference kernels are not emulated."""
    for name, tensor in [('query', query), ('key', key), ('value', value)]:
        _tensor_argument(tensor, name)
    if query.ndim not in (2, 3) or key.ndim != query.ndim or value.ndim != query.ndim:
        raise AssertionError('MHA query/key/value must all be rank 2 or all rank 3')
    for name, flag in [('training', training), ('need_weights', need_weights), ('average_attn_weights', average_attn_weights), ('is_causal', is_causal), ('add_zero_attn', add_zero_attn), ('use_separate_proj_weight', use_separate_proj_weight)]:
        if type(flag) is not bool:
            raise TypeError(f'MHA {name} must be bool')
    if not isinstance(dropout_p, numbers.Real) or not 0 <= dropout_p <= 1:
        raise ValueError('MHA dropout_p must be in [0,1]')
    if query.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('MHA supports real float32/float64')
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise RuntimeError('MHA query/key/value must have matching dtypes')
    embed, heads = _positive(embed_dim_to_check, 'embed_dim'), _positive(num_heads, 'num_heads')
    if query.shape[-1] != embed or embed % heads:
        raise AssertionError('MHA embedding dimension must match and divide evenly into heads')
    unbatched = query.ndim == 2
    length, batch, source = query.shape[0], 1 if unbatched else query.shape[1], key.shape[0]
    if key.shape[:-1] != value.shape[:-1] or (not unbatched and (key.shape[1] != batch or value.shape[1] != batch)):
        raise AssertionError('MHA key/value sequence and batch dimensions must match')
    if not use_separate_proj_weight and (key.shape[-1] != embed or value.shape[-1] != embed):
        raise AssertionError('Packed MHA requires matching query/key/value embedding dimensions')
    mask_type = _mask_kind(attn_mask, 'attn_mask'); padding_type = _mask_kind(key_padding_mask, 'key_padding_mask')
    if mask_type is not None and padding_type is not None and mask_type != padding_type:
        warnings.warn('Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.', UserWarning, stacklevel=2)
    if key_padding_mask is not None and key_padding_mask.ndim != (1 if unbatched else 2):
        raise AssertionError('MHA key_padding_mask rank does not match the input')
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            if attn_mask.shape != (length, source):
                raise RuntimeError('MHA 2D attn_mask must have shape (target_length, source_length)')
        elif attn_mask.ndim == 3:
            if attn_mask.shape != (batch * heads, length, source):
                raise RuntimeError('MHA 3D attn_mask must have shape (batch * heads, target_length, source_length)')
        else:
            raise AssertionError('MHA attn_mask must be rank 2 or 3')
    if is_causal and attn_mask is None:
        raise RuntimeError('Need attn_mask when specifying the MHA is_causal hint')
    # Canonical boolean masks use True=blocked, unlike functional SDPA masks.
    mask = _canonical_mask(attn_mask, query.dtype)
    padding = _canonical_mask(key_padding_mask, query.dtype)
    _tensor_argument(in_proj_bias, 'in_proj_bias', (3 * embed,), query.dtype, optional=True)
    _tensor_argument(out_proj_weight, 'out_proj_weight', (embed, embed), query.dtype)
    _tensor_argument(out_proj_bias, 'out_proj_bias', (embed,), query.dtype, optional=True)
    if (bias_k is None) != (bias_v is None):
        raise AssertionError('MHA bias_k and bias_v must be specified together')
    if unbatched:
        if query is key and key is value:
            query = key = value = query.unsqueeze(1)
        elif key is value:
            query, key = query.unsqueeze(1), key.unsqueeze(1); value = key
        else:
            query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
        if padding is not None: padding = padding.unsqueeze(0)
    q, k, v = _projections(query, key, value, embed, in_proj_weight, in_proj_bias,
                          use_separate_proj_weight, q_proj_weight, k_proj_weight, v_proj_weight)
    if bias_k is not None:
        if static_k is not None or static_v is not None:
            raise AssertionError('MHA bias tokens cannot be combined with static K/V')
        _tensor_argument(bias_k, 'bias_k', (1, 1, embed), query.dtype)
        _tensor_argument(bias_v, 'bias_v', (1, 1, embed), query.dtype)
        k = _ops.concatenate([k, bias_k.expand(1, batch, embed)], 0)
        v = _ops.concatenate([v, bias_v.expand(1, batch, embed)], 0)
        mask, padding = _extend_mask(mask), _extend_mask(padding)
    width = embed // heads
    q = q.reshape(length, batch * heads, width).transpose(0, 1)
    k = k.reshape(k.shape[0], batch * heads, width).transpose(0, 1)
    v = v.reshape(v.shape[0], batch * heads, width).transpose(0, 1)
    for name, tensor in [('static_k', static_k), ('static_v', static_v)]:
        if tensor is not None:
            _tensor_argument(tensor, name, dtype=query.dtype)
            if tensor.ndim != 3 or tensor.shape[0] != batch * heads or tensor.shape[2] != width:
                raise AssertionError(f'MHA {name} must have shape (batch * heads, source_length, head_dim)')
    if static_k is not None: k = static_k
    if static_v is not None: v = static_v
    if k.shape[1] != v.shape[1]:
        raise RuntimeError('MHA projected/static K/V source lengths must match')
    if add_zero_attn:
        zeros = _tensor.Tensor(np.zeros((batch * heads, 1, width), dtype=query.dtype))
        k, v = _ops.concatenate([k, zeros], 1), _ops.concatenate([v, zeros], 1)
        mask, padding = _extend_mask(mask), _extend_mask(padding)
    source = k.shape[1]
    if padding is not None:
        if padding.shape != (batch, source):
            raise AssertionError(f'MHA key_padding_mask must have shape {(batch, source)}')
        padding = padding.reshape(batch, 1, 1, source).expand(batch, heads, 1, source).reshape(batch * heads, 1, source)
    causal_hint = is_causal and not need_weights and key_padding_mask is None
    if causal_hint:
        mask = None
    elif mask is not None and mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if padding is not None:
        mask = padding if mask is None else mask + padding
    probability = dropout_p if training else 0.
    if need_weights:
        if mask is not None and mask.dtype != query.dtype:
            raise RuntimeError('MHA attention weights require the mask to match the query dtype')
        logits = (q * math.sqrt(1 / width)) @ k.transpose(-1, -2)
        if mask is not None: logits = logits + mask
        with np.errstate(all='ignore'):
            weights = logits if not logits.shape[-1] else logits.softmax(-1)
        weights = _activation.dropout(weights, probability, training=True)
        result = weights @ v
        weights = weights.reshape(batch, heads, length, source)
        if average_attn_weights: weights = weights.mean(1)
        if unbatched: weights = weights[0]
    else:
        # Explicit dimensions keep zero-length inputs valid (no ambiguous -1).
        qm, km, vm = q.reshape(batch, heads, length, width), k.reshape(batch, heads, source, width), v.reshape(batch, heads, source, width)
        if mask is not None:
            mask = mask.reshape(1 if mask.shape[0] == 1 else batch, 1 if mask.shape[0] == 1 else heads, mask.shape[-2], source)
        result = _attention.scaled_dot_product_attention(qm, km, vm, mask, probability, causal_hint).reshape(batch * heads, length, width)
        weights = None
    output = _nn.linear(result.transpose(0, 1).reshape(length * batch, embed), out_proj_weight, out_proj_bias).reshape(length, batch, embed)
    return (output[:, 0] if unbatched else output), weights


class NonDynamicallyQuantizableLinear(_nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        Module.__init__(self)
        dtype = parameter_dtype(dtype, device)
        if type(in_features) is not int or type(out_features) is not int or min(in_features, out_features) < 0:
            raise ValueError('Linear dimensions must be nonnegative integers')
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(_tensor.Tensor(np.empty((out_features, in_features), dtype=dtype)))
        self.register_parameter('bias', Parameter(_tensor.Tensor(np.empty(out_features, dtype=dtype))) if bias else None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = self.in_features ** -.5 if self.in_features else 0
        with _autograd.no_grad():
            for parameter in [self.weight, self.bias]:
                if parameter is not None:
                    parameter[...] = (_random.uniform_samples(np.random, parameter.shape, parameter.dtype) * (2 * bound) - bound).astype(parameter.dtype)


class MultiheadAttention(Module):
    __constants__ = ['batch_first']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=False,
                 device=None, dtype=None):
        super().__init__()
        self.embed_dim, self.num_heads = _positive(embed_dim, 'embed_dim'), _positive(num_heads, 'num_heads')
        if self.embed_dim % self.num_heads:
            raise AssertionError('embed_dim must be divisible by num_heads')
        if not isinstance(dropout, numbers.Real) or not 0 <= dropout <= 1:
            raise ValueError('MHA dropout must be in [0,1]')
        for name, flag in [('bias', bias), ('add_bias_kv', add_bias_kv), ('add_zero_attn', add_zero_attn), ('batch_first', batch_first)]:
            if type(flag) is not bool: raise TypeError(f'MHA {name} must be bool')
        dtype = parameter_dtype(dtype, device)
        self.kdim = self.embed_dim if kdim is None else _positive(kdim, 'kdim')
        self.vdim = self.embed_dim if vdim is None else _positive(vdim, 'vdim')
        self._qkv_same_embed_dim = self.kdim == self.vdim == self.embed_dim
        self.dropout, self.batch_first, self.head_dim = dropout, batch_first, self.embed_dim // self.num_heads
        if self._qkv_same_embed_dim:
            self.in_proj_weight = Parameter(_tensor.Tensor(np.empty((3 * self.embed_dim, self.embed_dim), dtype=dtype)))
            for name in ['q_proj_weight', 'k_proj_weight', 'v_proj_weight']: self.register_parameter(name, None)
        else:
            for name, width in [('q_proj_weight', self.embed_dim), ('k_proj_weight', self.kdim), ('v_proj_weight', self.vdim)]:
                setattr(self, name, Parameter(_tensor.Tensor(np.empty((self.embed_dim, width), dtype=dtype))))
            self.register_parameter('in_proj_weight', None)
        self.register_parameter('in_proj_bias', Parameter(_tensor.Tensor(np.empty(3 * self.embed_dim, dtype=dtype))) if bias else None)
        self.out_proj = NonDynamicallyQuantizableLinear(self.embed_dim, self.embed_dim, bias=bias, dtype=dtype)
        for name in ['bias_k', 'bias_v']:
            if add_bias_kv: setattr(self, name, Parameter(_tensor.Tensor(np.empty((1, 1, self.embed_dim), dtype=dtype))))
            else: setattr(self, name, None)
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        with _autograd.no_grad():
            for parameter in ([self.in_proj_weight] if self._qkv_same_embed_dim else [self.q_proj_weight, self.k_proj_weight, self.v_proj_weight]):
                bound = math.sqrt(6 / sum(parameter.shape))
                parameter[...] = (_random.uniform_samples(np.random, parameter.shape, parameter.dtype) * (2 * bound) - bound).astype(parameter.dtype)
            if self.in_proj_bias is not None:
                self.in_proj_bias.zero_(); self.out_proj.bias.zero_()
            for parameter in [self.bias_k, self.bias_v]:
                if parameter is not None:
                    parameter[...] = _random.normal_samples(_random._default_generator, parameter.shape, parameter.dtype) * math.sqrt(1 / self.embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if not isinstance(query, _tensor.Tensor): raise TypeError('MHA query must be a Tensor')
        batched = query.ndim == 3
        if self.batch_first and batched:
            if query is key and key is value:
                query = key = value = query.transpose(0, 1)
            elif key is value:
                query, key = query.transpose(0, 1), key.transpose(0, 1); value = key
            else:
                query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        result, weights = multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias, self.training, key_padding_mask,
            need_weights, attn_mask, not self._qkv_same_embed_dim, self.q_proj_weight,
            self.k_proj_weight, self.v_proj_weight, average_attn_weights=average_attn_weights, is_causal=is_causal)
        return (result.transpose(0, 1) if self.batch_first and batched else result), weights

    def merge_masks(self, attn_mask, key_padding_mask, query):
        if attn_mask is None: return key_padding_mask, 1 if key_padding_mask is not None else None
        batch, length = query.shape[:2]
        merged = attn_mask.reshape(batch, self.num_heads, length, length) if attn_mask.ndim == 3 else attn_mask.reshape(1, 1, length, length).expand(batch, self.num_heads, length, length)
        if key_padding_mask is not None:
            merged = merged + key_padding_mask.reshape(batch, 1, 1, length).expand(batch, self.num_heads, 1, length)
        return merged, 2

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault('_qkv_same_embed_dim', True)
