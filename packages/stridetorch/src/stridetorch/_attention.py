"""Math SDPA with fused VJPs, forward AD and unrepeated grouped K/V heads."""
import math
import numbers
from functools import lru_cache
import numpy as np

from . import _autograd, _errors, _ops, _random, _tensor


def _head_product(left, right, grouped):
    """Broadcast a K/V head across its query group without repeating storage."""
    if not grouped or left.shape[-3] == right.shape[-3]:
        return left @ right
    heads, queries = right.shape[-3], left.shape[-3]
    a = left.reshape((*left.shape[:-3], heads, queries // heads, *left.shape[-2:]))
    value = a @ np.expand_dims(right, -3)
    return value.reshape((*value.shape[:-4], queries, *value.shape[-2:]))


def _head_gradient(left, right, shape, grouped):
    if not grouped or left.shape[-3] == shape[-3]:
        return _ops.unbroadcast(left @ right, shape)
    heads, queries = shape[-3], left.shape[-3]
    a = left.reshape((*left.shape[:-3], heads, queries // heads, *left.shape[-2:]))
    b = right.reshape((*right.shape[:-3], heads, queries // heads, *right.shape[-2:]))
    return _ops.unbroadcast((a @ b).sum(axis=-3), shape)


def _tensor_head_product(left, right, grouped):
    if not grouped or left.shape[-3] == right.shape[-3]:
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
def _geometry(query, key, value, mask, grouped):
    """Cache immutable shape checks only; never retain tensor or RNG state."""
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
    return (*batch, query[-2], value[-1])


def _validate(query, key, value, mask, dropout_p, is_causal, scale, grouped):
    if any(not isinstance(x, _tensor.Tensor) for x in (query, key, value)):
        raise TypeError('scaled_dot_product_attention expects Tensor query, key and value')
    if mask is not None and not isinstance(mask, _tensor.Tensor):
        raise TypeError('SDPA attn_mask must be a Tensor')
    if not isinstance(dropout_p, numbers.Real):
        raise TypeError('SDPA dropout_p must be a number')
    if not 0 <= dropout_p <= 1:
        raise RuntimeError('SDPA dropout_p must be in [0,1]')
    if type(is_causal) is not bool or type(grouped) is not bool:
        raise TypeError('SDPA is_causal and enable_gqa must be bool')
    if is_causal and mask is not None:
        raise RuntimeError('SDPA explicit attn_mask cannot be combined with is_causal')
    if scale is not None and not isinstance(scale, numbers.Real):
        raise TypeError('SDPA scale must be a number')
    if any(x.ndim < 2 for x in (query, key, value)) or query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise RuntimeError('Invalid SDPA shapes')
    if query.dtype != key.dtype or query.dtype != value.dtype or query.dtype.kind != 'f':
        raise RuntimeError('SDPA expects floating tensors with matching dtypes')
    if query.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('SDPA supports real float32/float64')
    shape = _geometry(query.shape, key.shape, value.shape, mask.shape if mask is not None else None, grouped)
    if mask is not None and mask.dtype != np.dtype('bool') and mask.dtype != query.dtype and mask.dtype != np.dtype('float32'):
        raise RuntimeError('SDPA mask must be boolean, float32 or match the query dtype')
    factor = (1 / math.sqrt(query.shape[-1]) if query.shape[-1] else 1.) if scale is None else float(scale)
    return factor, shape


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

    def backward(g, needs):
        logits_gradient = None
        if needs[0] or needs[1] or (mask_shape is not None and needs[3]):
            incoming = _head_product(g, np.swapaxes(v, -1, -2), grouped)
            if multiplier is not None:
                incoming = incoming * multiplier
            logits_gradient = probabilities * (incoming - (incoming * probabilities).sum(axis=-1, keepdims=True))
        gradients = [
            _ops.unbroadcast(_head_product(logits_gradient, k, grouped) * qscale, q.shape) if needs[0] else None,
            _head_gradient(np.swapaxes(logits_gradient, -1, -2), q, k.shape, grouped) * kscale if needs[1] else None,
            _head_gradient(np.swapaxes(used, -1, -2), g, v.shape, grouped) if needs[2] else None,
        ]
        if mask_shape is not None:
            gradients.append(_ops.unbroadcast(logits_gradient, mask_shape) if needs[3] else None)
        return gradients

    return _autograd.joint_result(_head_product(used, v, grouped), operands, backward, saved=saved)


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
    forward = any(isinstance(x, _autograd.ForwardTensor) for x in (query, key, value, attn_mask))
    return (_forward if forward else _primal)(query, key, value, attn_mask, dropout_p, is_causal, factor, enable_gqa, shape)
