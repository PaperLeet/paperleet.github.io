"""Math SDPA with fused VJPs, forward AD and unrepeated grouped K/V heads."""
import math
import numbers
from functools import lru_cache
import numpy as np

from . import _autograd, _errors, _ops, _random, _state, _tensor, _matmul

_WASM_ATTENTION = _tensor._WASM_MATMUL

_REAL = (np.dtype('float32').str, np.dtype('float64').str)
_BOOL = np.dtype('bool').str


def _head_product(left, right, grouped):
    """Broadcast a K/V head across its query group without repeating storage."""
    if not grouped or left.shape[-3] == right.shape[-3] or right.shape[-3] == 1:
        return _matmul.product(left, right) if _WASM_ATTENTION and left.size >= 4096 else left @ right
    heads, queries = right.shape[-3], left.shape[-3]
    a = left.reshape((*left.shape[:-3], heads, queries // heads, *left.shape[-2:]))
    expanded = np.expand_dims(right, -3)
    value = _matmul.product(a, expanded) if _WASM_ATTENTION and left.size >= 4096 else a @ expanded
    return value.reshape((*value.shape[:-4], queries, *value.shape[-2:]))


def _head_gradient(left, right, shape, grouped):
    if not grouped or left.shape[-3] == shape[-3] or shape[-3] == 1:
        product = _matmul.product(left, right) if _WASM_ATTENTION and left.size >= 4096 else left @ right
        return _ops.unbroadcast(product, shape)
    heads, queries = shape[-3], left.shape[-3]
    a = left.reshape((*left.shape[:-3], heads, queries // heads, *left.shape[-2:]))
    b = right.reshape((*right.shape[:-3], heads, queries // heads, *right.shape[-2:]))
    product = _matmul.product(a, b) if _WASM_ATTENTION and left.size >= 4096 else a @ b
    return _ops.unbroadcast(product.sum(axis=-3), shape)


def _tensor_head_product(left, right, grouped):
    if not grouped or left.shape[-3] == right.shape[-3] or right.shape[-3] == 1:
        return left @ right
    heads, queries = right.shape[-3], left.shape[-3]
    a = left.reshape((*left.shape[:-3], heads, queries // heads, *left.shape[-2:]))
    value = a @ right.unsqueeze(-3)
    return value.reshape((*value.shape[:-4], queries, *value.shape[-2:]))


def _probabilities(scores):
    if not scores.shape[-1]:
        return np.zeros_like(scores)
    with np.errstate(all='ignore'):
        maximum = scores.max(axis=-1, keepdims=True)
        exponentials = np.exp(scores - np.where(maximum == -np.inf, 0, maximum))
        denominator = exponentials.sum(axis=-1, keepdims=True)
        return np.divide(exponentials, denominator, out=np.zeros_like(exponentials), where=denominator != 0)


def _dropout_multiplier(shape, dtype, probability):
    if probability == 0:
        return None
    if probability == 1:
        return np.zeros(shape, dtype=dtype)
    multiplier = (_random.uniform_samples(np.random, shape, dtype) >= probability).astype(dtype)
    multiplier /= np.asarray(1 - probability, dtype=dtype)
    return multiplier


@lru_cache(maxsize=64)
def _geometry(query, key, value, mask, grouped, dtypes=None):
    """Cache canonical shapes/dtype strings, never tensors or dtype metadata."""
    if len(query) < 2 or len(key) < 2 or len(value) < 2 or query[-1] != key[-1] or key[-2] != value[-2]:
        raise RuntimeError('Invalid SDPA shapes')
    if dtypes is not None:
        qtype, ktype, vtype, mtype = dtypes
        if qtype != ktype or qtype != vtype or qtype[1] != 'f':
            raise RuntimeError('SDPA expects floating tensors with matching dtypes')
        if qtype not in _REAL:
            raise _errors.BrowserUnsupported('SDPA supports real float32/float64')
    if grouped:
        if any(len(x) < 3 for x in (query, key, value)):
            raise IndexError('SDPA grouped query attention requires a head dimension')
        if any(x[-3] and query[-3] % x[-3] for x in (key, value)):
            raise RuntimeError('SDPA K/V heads must divide the query heads')
    try:
        batch = np.broadcast_shapes(*(x[:(-3 if grouped else -2)] for x in (query, key, value)))
        if grouped:
            batch += (query[-3],)
        score_shape = (*batch, query[-2], key[-2])
        if mask is not None and np.broadcast_shapes(score_shape, mask) != score_shape:
            raise ValueError()
    except ValueError as error:
        raise RuntimeError('SDPA batch, head or mask dimensions are not broadcastable') from error
    if dtypes is not None and mtype not in (None, _BOOL, qtype, _REAL[0]):
        raise RuntimeError('SDPA mask must be boolean, float32 or match the query dtype')
    return (1 / math.sqrt(query[-1]) if query[-1] else 1.), (*batch, query[-2], value[-1])


def _validate(query, key, value, mask, dropout_p, is_causal, scale, grouped):
    if not isinstance(query, _tensor.Tensor) or not isinstance(key, _tensor.Tensor) or not isinstance(value, _tensor.Tensor):
        raise TypeError('scaled_dot_product_attention expects Tensor query, key and value')
    if mask is not None and not isinstance(mask, _tensor.Tensor):
        raise TypeError('SDPA attn_mask must be a Tensor')
    if type(dropout_p) not in (float, int) and not isinstance(dropout_p, numbers.Real):
        raise TypeError('SDPA dropout_p must be a number')
    if not 0 <= dropout_p <= 1:
        raise RuntimeError('SDPA dropout_p must be in [0,1]')
    if type(is_causal) is not bool or type(grouped) is not bool:
        raise TypeError('SDPA is_causal and enable_gqa must be bool')
    if is_causal and mask is not None:
        raise RuntimeError('SDPA explicit attn_mask cannot be combined with is_causal')
    if scale is not None and type(scale) not in (float, int) and not isinstance(scale, numbers.Real):
        raise TypeError('SDPA scale must be a number')
    q, k, v = query._data, key._data, value._data
    m = mask._data if mask is not None else None
    factor, shape = _geometry(q.shape, k.shape, v.shape, m.shape if m is not None else None, grouped,
                              (q.dtype.str, k.dtype.str, v.dtype.str, m.dtype.str if m is not None else None))
    return factor if scale is None else float(scale), shape


def _scales(factor):
    root = math.sqrt(abs(factor))
    return math.copysign(root, factor), root


def _primal(query, key, value, mask, dropout_p, causal, factor, grouped, output_shape):
    q, k, v = query._data, key._data, value._data
    operands = [query, key, value]
    saved = []
    mask_shape = None
    if mask is not None:
        if mask.dtype.kind == 'f':
            operands.append(mask)
            mask_shape = mask.shape
        else:
            saved.append(mask)
    if grouped and (not key.shape[-3] or not value.shape[-3]):
        return _autograd.joint_result(np.zeros(output_shape, dtype=q.dtype), operands,
            lambda g, needs: [np.zeros_like(x._data) if need else None for x, need in zip(operands, needs)], saved=saved)
    qshape, kshape, vshape = q.shape, k.shape, v.shape
    qscale, kscale = _scales(factor)
    q, k = q * qscale, k * kscale
    scores = _head_product(q, np.swapaxes(k, -1, -2), grouped)
    if causal:
        scores = np.where(np.arange(k.shape[-2])[None, :] <= np.arange(q.shape[-2])[:, None], scores, -np.inf)
    if mask is not None:
        scores = np.where(mask._data, scores, -np.inf) if mask.dtype.kind == 'b' else scores + mask._data
    probabilities = _probabilities(scores)
    multiplier = _dropout_multiplier(probabilities.shape, probabilities.dtype, dropout_p)
    used = probabilities if multiplier is None else probabilities * multiplier
    output = _head_product(used, v, grouped)
    enabled = _state._grad_enabled
    need_q, need_k, need_v = enabled and query.requires_grad, enabled and key.requires_grad, enabled and value.requires_grad
    need_mask = enabled and mask_shape is not None and mask.requires_grad
    if not (need_q or need_k or need_v or need_mask):
        return _tensor.Tensor(output, _metadata_owned=True)
    need_logits = need_q or need_k or need_mask
    saved_p = probabilities if need_logits else None
    saved_multiplier = multiplier if need_logits else None
    saved_used = used if need_v and not need_logits else None
    saved_q, saved_k, saved_v = q if need_k else None, k if need_q else None, v if need_logits else None

    def backward(g, needs):
        logits_gradient = None
        if needs[0] or needs[1] or (mask_shape is not None and needs[3]):
            incoming = _head_product(g, np.swapaxes(saved_v, -1, -2), grouped)
            if saved_multiplier is not None:
                incoming = incoming * saved_multiplier
            logits_gradient = saved_p * (incoming - (incoming * saved_p).sum(axis=-1, keepdims=True))
        weights = None
        if needs[2]:
            weights = saved_used if saved_used is not None else saved_p if saved_multiplier is None else saved_p * saved_multiplier
        gradients = [
            _ops.unbroadcast(_head_product(logits_gradient, saved_k, grouped) * qscale, qshape) if needs[0] else None,
            _head_gradient(np.swapaxes(logits_gradient, -1, -2), saved_q, kshape, grouped) * kscale if needs[1] else None,
            _head_gradient(np.swapaxes(weights, -1, -2), g, vshape, grouped) if needs[2] else None,
        ]
        if mask_shape is not None:
            gradients.append(_ops.unbroadcast(logits_gradient, mask_shape) if needs[3] else None)
        return gradients

    return _autograd.joint_result(output, operands, backward, saved=saved)


def _forward(query, key, value, mask, dropout_p, causal, factor, grouped, output_shape):
    q, dq = _autograd.forward_parts(query)
    k, dk = _autograd.forward_parts(key)
    v, dv = _autograd.forward_parts(value)
    m, dm = _autograd.forward_parts(mask) if mask is not None and mask.dtype.kind == 'f' else (mask, None)
    if grouped and (not k.shape[-3] or not v.shape[-3]):
        primal = _primal(q, k, v, m, dropout_p, causal, factor, grouped, output_shape)
        inputs = [q, k, v] + ([m] if dm is not None else []) + [dq, dk, dv] + ([dm] if dm is not None else [])
        tangent = _autograd.joint_result(np.zeros(output_shape, dtype=q.dtype), inputs,
            lambda g, needs: [np.zeros_like(x._data) if need else None for x, need in zip(inputs, needs)])
        return _autograd.ForwardTensor(primal, tangent)
    qscale, kscale = _scales(factor)
    q, k, dq, dk = q * qscale, k * kscale, dq * qscale, dk * kscale
    scores = _tensor_head_product(q, k.transpose(-1, -2), grouped)
    direction = _tensor_head_product(dq, k.transpose(-1, -2), grouped) + _tensor_head_product(q, dk.transpose(-1, -2), grouped)
    if causal:
        keep = _tensor.Tensor(np.arange(k.shape[-2])[None, :] <= np.arange(q.shape[-2])[:, None])
        scores = _ops.where(keep, scores, -np.inf)
        direction = _ops.where(keep, direction, 0.)
    if m is not None:
        if m.dtype.kind == 'b':
            scores = _ops.where(m, scores, -np.inf)
            direction = _ops.where(m, direction, 0.)
        else:
            scores = scores + m
            direction = direction + dm
    # Fully masked rows are constant zero, including their mixed derivatives.
    # Native math SDPA currently produces NaNs in that mixed-AD corner; the
    # stable extension here is covered by independent finite differences.
    raw = _probabilities(scores._data)
    probabilities = _autograd.result(raw, (scores, lambda g: raw * (g - (g * raw).sum(axis=-1, keepdims=True))), saved=(m,) if m is not None else ())
    derivative = probabilities * (direction - (direction * probabilities).sum(-1, True))
    multiplier = _dropout_multiplier(probabilities.shape, probabilities.dtype, dropout_p)
    if multiplier is not None:
        multiplier = _tensor.Tensor(multiplier)
        used = probabilities * multiplier
        derivative = derivative * multiplier
    else:
        used = probabilities
    primal = _tensor_head_product(used, v, grouped)
    tangent = _tensor_head_product(derivative, v, grouped) + _tensor_head_product(used, dv, grouped)
    return _autograd.ForwardTensor(primal, tangent)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.,
                                 is_causal=False, *, scale=None, enable_gqa=False):
    """NumPy math attention; seeded dropout follows this library's RNG stream."""
    factor, shape = _validate(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    forward = (isinstance(query, _autograd.ForwardTensor) or isinstance(key, _autograd.ForwardTensor)
               or isinstance(value, _autograd.ForwardTensor) or isinstance(attn_mask, _autograd.ForwardTensor))
    return (_forward if forward else _primal)(query, key, value, attn_mask, dropout_p, is_causal, factor, enable_gqa, shape)
