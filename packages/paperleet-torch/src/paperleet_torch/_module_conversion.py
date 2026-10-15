"""Recursive CPU model conversions preserving Parameters and gradient identity."""
import numpy as np

from . import _autograd, _conversion, _errors, _state, _tensor


class ModuleConversion:
    def _apply(self, fn, recurse=True):
        if _state._jvp_active:
            raise _errors.BrowserUnsupported('Module storage conversion during JVP is unsupported')
        if self.__dict__.get('_converting_storage'):
            raise _errors.BrowserUnsupported('Cyclic module conversion is unsupported')
        self.__dict__['_converting_storage'] = True
        try:
            if recurse:
                for child in self.children():
                    child._apply(fn)
            for parameter in self._parameters.values():
                if parameter is None:
                    continue
                with _autograd.no_grad():
                    converted = fn(parameter)
                grad = parameter.grad
                parameter._replace_data(converted)
                if grad is not None:
                    with _autograd.no_grad():
                        converted_grad = fn(grad)
                    grad._replace_data(converted_grad)
            for name, buffer in self._buffers.items():
                if buffer is not None:
                    converted = fn(buffer)
                    if not isinstance(converted, _tensor.Tensor):
                        raise TypeError('A buffer conversion must return a Tensor')
                    self._buffers[name] = converted
        finally:
            self.__dict__.pop('_converting_storage', None)
        return self

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _, fmt = _conversion.parse_to(args, kwargs, module=True)
        if dtype is not None:
            if dtype.kind not in 'fc':
                raise TypeError('Module.to only accepts floating or complex dtypes')
            if dtype not in (np.dtype('float32'), np.dtype('float64'), np.dtype('complex64'), np.dtype('complex128')):
                raise _errors.BrowserUnsupported('Module conversion requires float32/float64 or frozen complex tensors')
            if dtype.kind == 'c' and any(p.requires_grad for p in self.parameters()):
                raise _errors.BrowserUnsupported('Complex module autograd is unsupported')
        def convert(value):
            target = dtype if value.dtype.kind in 'fc' else None
            return value.to(device=device, dtype=target, non_blocking=non_blocking, memory_format=fmt)
        return self._apply(convert)

    def cpu(self):
        return self._apply(lambda value: value.cpu())

    def float(self):
        return self._apply(lambda value: value.float() if value.is_floating_point() else value)

    def double(self):
        return self._apply(lambda value: value.double() if value.is_floating_point() else value)

    def type(self, dtype):
        return self._apply(lambda value: value.type(dtype))
