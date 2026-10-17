"""Dense Transformer composition on the existing attention and tensor engine."""
import copy
import math
import numbers
import warnings
import numpy as np

from . import _activation, _autograd, _device, _multihead, _nn, _normalization, _random, _registry, _tensor
from ._containers import ModuleList
from ._module import Module


def _count(value, name):
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer')
    if value < 0:
        raise ValueError(f'{name} must be nonnegative')
    return int(value)


def _flag(value, name, optional=False):
    if optional and value is None:
        return None
    if type(value) is not bool:
        raise TypeError(f'{name} must be bool')
    return value


def _activation_function(value):
    if isinstance(value, str):
        if value not in ('relu', 'gelu'):
            raise RuntimeError('Transformer activation must be relu or gelu')
        value = getattr(_activation, value)
    wrapped = getattr(value, '__wrapped__', value)
    # Store the stable module functions for the library's registry wrappers.
    # This also makes default/string/functional activation modules pickleable.
    if wrapped is _activation.relu or wrapped is _activation.gelu:
        value = wrapped
    if not callable(value):
        raise TypeError('Transformer activation must be a string or callable')
    return value


def _activation_kind(value):
    if value is _activation.relu or isinstance(value, _nn.ReLU):
        return 1
    if value is _activation.gelu or isinstance(value, _activation.GELU):
        return 2
    return 0


def _layer_parts(layer, d_model, nhead, dim_feedforward, dropout, activation,
                 layer_norm_eps, batch_first, norm_first, bias, device, dtype, decoder):
    _flag(norm_first, 'norm_first')
    width = _count(dim_feedforward, 'dim_feedforward')
    options = dict(device=device, dtype=dtype, bias=bias)
    layer.self_attn = _multihead.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **options)
    d_model = layer.self_attn.embed_dim
    if decoder:
        layer.multihead_attn = _multihead.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **options)
    layer.linear1 = _nn.Linear(d_model, width, **options)
    layer.dropout = _activation.Dropout(dropout)
    layer.linear2 = _nn.Linear(width, d_model, **options)
    layer.norm_first = norm_first
    for i in range(1, 4 if decoder else 3):
        setattr(layer, 'norm'+str(i), _normalization.LayerNorm(d_model, eps=layer_norm_eps, **options))
    for i in range(1, 4 if decoder else 3):
        setattr(layer, 'dropout'+str(i), _activation.Dropout(dropout))
    activation = _activation_function(activation)
    if not decoder:
        layer.activation_relu_or_gelu = _activation_kind(activation)
    layer.activation = activation


def _encoder_masks(src, mask, padding):
    if not isinstance(src, _tensor.Tensor):
        raise TypeError('Transformer input must be a Tensor')
    mt = _multihead._mask_kind(mask, 'mask')
    pt = _multihead._mask_kind(padding, 'key_padding_mask')
    if mt is not None and pt is not None and mt != pt:
        warnings.warn('Support for mismatched key_padding_mask and mask is deprecated. Use same type for both instead.', UserWarning, stacklevel=3)
    return _multihead._canonical_mask(mask, src.dtype), _multihead._canonical_mask(padding, src.dtype)


def _length(src, batch_first):
    if not isinstance(src, _tensor.Tensor):
        raise TypeError('Transformer input must be a Tensor')
    if src.ndim not in (2, 3):
        raise RuntimeError('Transformer input must be unbatched rank 2 or batched rank 3')
    return src.shape[1 if src.ndim == 3 and batch_first else 0]


def _causal(mask, hint, size):
    _flag(hint, 'is_causal', optional=True)
    if hint is not None:
        return hint
    if mask is None or mask.shape != (size, size):
        return False
    return bool(np.array_equal(mask._data, Transformer.generate_square_subsequent_mask(size, dtype=mask.dtype)._data))


class _DenseLayer(Module):
    __constants__ = ['norm_first']
    _feedforward_dropout = 'dropout2'

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        y = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                           is_causal=is_causal, need_weights=False)[0]
        return self.dropout1(y)

    def _ff_block(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return getattr(self, self._feedforward_dropout)(y)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'activation' in state:
            state['activation'] = _registry.pickle_operator(state['activation'])
        return state

    def __setstate__(self, state):
        # copy.copy passes state directly, without pickling its contents.
        if isinstance(state.get('activation'), _registry._OperatorReference):
            state = {**state, 'activation': state['activation'].function}
        self.__dict__.update(state)
        if not hasattr(self, 'activation'):
            self.activation = _activation.relu
        if isinstance(self, TransformerEncoderLayer) and not hasattr(self, 'activation_relu_or_gelu'):
            self.activation_relu_or_gelu = _activation_kind(self.activation)


class TransformerEncoderLayer(_DenseLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=.1,
                 activation=_activation.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        super().__init__()
        _layer_parts(self, d_model, nhead, dim_feedforward, dropout, activation,
                     layer_norm_eps, batch_first, norm_first, bias, device, dtype, False)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        mask, padding = _encoder_masks(src, src_mask, src_key_padding_mask)
        if self.norm_first:
            x = src + self._sa_block(self.norm1(src), mask, padding, is_causal)
            return x + self._ff_block(self.norm2(x))
        x = self.norm1(src + self._sa_block(src, mask, padding, is_causal))
        return self.norm2(x + self._ff_block(x))


class TransformerDecoderLayer(_DenseLayer):
    _feedforward_dropout = 'dropout3'

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=.1,
                 activation=_activation.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        super().__init__()
        _layer_parts(self, d_model, nhead, dim_feedforward, dropout, activation,
                     layer_norm_eps, batch_first, norm_first, bias, device, dtype, True)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal=False):
        y = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                                is_causal=is_causal, need_weights=False)[0]
        return self.dropout2(y)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        def cross(x):
            return self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
        if self.norm_first:
            x = tgt + self._sa_block(self.norm1(tgt), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + cross(self.norm2(x))
            return x + self._ff_block(self.norm3(x))
        x = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
        x = self.norm2(x + cross(x))
        return self.norm3(x + self._ff_block(x))


class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        self.num_layers = _count(num_layers, 'num_layers')
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for _ in range(self.num_layers)])
        self.norm = norm
        self.enable_nested_tensor = _flag(enable_nested_tensor, 'enable_nested_tensor')
        self.use_nested_tensor = False  # This backend always executes dense math.
        self.mask_check = _flag(mask_check, 'mask_check')

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        mask, padding = _encoder_masks(src, mask, src_key_padding_mask)
        length = _length(src, self.layers[0].self_attn.batch_first)
        causal = _causal(mask, is_causal, length)
        x = src
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=padding, is_causal=causal)
        return x if self.norm is None else self.norm(x)


class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_layers = _count(num_layers, 'num_layers')
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.num_layers)])
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=False):
        length = _length(tgt, self.layers[0].self_attn.batch_first)
        causal = _causal(tgt_mask, tgt_is_causal, length)
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                      tgt_is_causal=causal, memory_is_causal=memory_is_causal)
        return x if self.norm is None else self.norm(x)


class Transformer(Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=.1, activation=_activation.relu,
                 custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, bias=True, device=None, dtype=None):
        super().__init__()
        self.d_model = _multihead._positive(d_model, 'd_model')
        self.nhead = _multihead._positive(nhead, 'nhead')
        self.batch_first = _flag(batch_first, 'batch_first')
        options = dict(dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                       layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first,
                       bias=bias, device=device, dtype=dtype)
        def norm():
            return _normalization.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
        self.encoder = custom_encoder if custom_encoder is not None else TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, **options), num_encoder_layers, norm())
        self.decoder = custom_decoder if custom_decoder is not None else TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, **options), num_decoder_layers, norm())
        self._reset_parameters()

    def _reset_parameters(self):
        with _autograd.no_grad():
            for parameter in self.parameters():
                if parameter.ndim > 1:
                    fan_in = parameter.shape[1] * math.prod(parameter.shape[2:])
                    fan_out = parameter.shape[0] * math.prod(parameter.shape[2:])
                    bound = math.sqrt(6 / (fan_in + fan_out)) if fan_in + fan_out else 0.
                    parameter[...] = (_random.uniform_samples(np.random, parameter.shape, parameter.dtype) * (2 * bound) - bound).astype(parameter.dtype)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                src_is_causal=None, tgt_is_causal=None, memory_is_causal=False):
        _length(src, self.batch_first); _length(tgt, self.batch_first)
        if src.ndim != tgt.ndim:
            raise RuntimeError('Transformer src and tgt must have matching ranks')
        if src.ndim == 3 and src.shape[0 if self.batch_first else 1] != tgt.shape[0 if self.batch_first else 1]:
            raise RuntimeError('the batch number of src and tgt must be equal')
        if src.shape[-1] != self.d_model or tgt.shape[-1] != self.d_model:
            raise RuntimeError('the feature number of src and tgt must be equal to d_model')
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=src_is_causal)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                            tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)

    @staticmethod
    def generate_square_subsequent_mask(sz, device=None, dtype=None):
        size = _count(sz, 'sz')
        _device.check_device(device)
        dtype = np.dtype('float32' if dtype is None else dtype)
        if dtype.kind not in 'bfc':
            raise RuntimeError('Causal masks require boolean, floating or complex dtype')
        return _tensor.Tensor(np.triu(np.full((size, size), -np.inf, dtype=dtype), 1))
