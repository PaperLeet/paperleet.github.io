"""Scoped tensor pullbacks for native-order vector-norm Hessian products.

Ordinary graphs retain their existing numerical engine and metadata. Rules and
temporary Tensor owners live only in one HVP context; nothing is attached to
normal nodes. A complete recorded primal graph is required before using the
three reverse traversals. Unsupported existing graphs retain their old path.
"""
import numpy as np

from . import _autograd, _errors, _graph, _layout, _ops, _reductions, _state, _tensor


def current():
    return _state._tensor_reverse_context


def unbroadcast(gradient, shape):
    while gradient.ndim > len(shape):
        gradient = gradient.sum(0)
    axes = tuple(axis for axis, (actual, wanted) in enumerate(zip(gradient.shape, shape)) if wanted == 1 and actual != 1)
    if axes:
        gradient = gradient.sum(axes, True)
    return gradient.reshape(shape) if gradient.shape != shape else gradient


class RecordedTensor(_tensor.Tensor):
    __slots__ = ()

    def _binary_record(self, value, other, operation, reverse=False):
        context = current()
        if context is None or not value.requires_grad:
            return value
        right = other if isinstance(other, _tensor.Tensor) else context.constant(other, self.dtype)
        def backward(g):
            if operation == 'add':
                left_grad, right_grad = g, g
            elif operation == 'sub':
                left_grad, right_grad = (-g, g) if reverse else (g, -g)
            elif operation == 'mul':
                left_grad, right_grad = g * right, g * self
            else:
                left_grad, right_grad = g / right, -g * ((self / right) / right)
            return unbroadcast(left_grad, self.shape), unbroadcast(right_grad, right.shape)
        return context.record(value, (self, right), backward)

    def __add__(self, other): return self._binary_record(super().__add__(other), other, 'add')
    __radd__ = __add__
    def __sub__(self, other): return self._binary_record(super().__sub__(other), other, 'sub')
    def __rsub__(self, other): return self._binary_record(super().__rsub__(other), other, 'sub', True)
    def __mul__(self, other): return self._binary_record(super().__mul__(other), other, 'mul')
    __rmul__ = __mul__
    def __truediv__(self, other): return self._binary_record(super().__truediv__(other), other, 'div')
    def __rtruediv__(self, other):
        context = current()
        if context is None:
            return super().__rtruediv__(other)
        return context.constant(other, self.dtype) / self
    def __neg__(self):
        value = super().__neg__()
        return current().record(value, (self,), lambda g: (-g,)) if current() else value
    def __pow__(self, exponent):
        value = super().__pow__(exponent)
        context = current()
        if context is None:
            return value
        def backward(g):
            if exponent == 0:
                return (context.constant(np.zeros(self.shape, dtype=self.dtype)),)
            return (g * (exponent * self ** (exponent - 1)),)
        return context.record(value, (self,), backward)
    def reciprocal(self):
        value = super().reciprocal()
        return current().record(value, (self,), lambda g: (g * -(value * value),)) if current() else value
    def exp(self):
        value = super().exp()
        return current().record(value, (self,), lambda g: (g * value,)) if current() else value
    def sin(self):
        value = super().sin()
        return current().record(value, (self,), lambda g: (g * self.cos(),)) if current() else value
    def cos(self):
        value = super().cos()
        return current().record(value, (self,), lambda g: (-g * self.sin(),)) if current() else value
    def tanh(self):
        value = super().tanh()
        return current().record(value, (self,), lambda g: (g * (1 - value ** 2),)) if current() else value
    def abs(self):
        value = super().abs()
        context = current()
        return context.record(value, (self,), lambda g: (g * context.sign(self),)) if context else value
    def relu(self):
        value = super().relu()
        context = current()
        return context.record(value, (self,), lambda g: (context.where(self._data <= 0, 0., g),)) if context else value
    def reshape(self, *shape):
        value = super().reshape(*shape)
        return current().record(value, (self,), lambda g: (g.reshape(self.shape),)) if current() else value
    def view(self, *shape, **kwargs):
        value = super().view(*shape, **kwargs)
        return current().record(value, (self,), lambda g: (g.reshape(self.shape),)) if current() else value
    def unsqueeze(self, dim):
        value = super().unsqueeze(dim)
        return current().record(value, (self,), lambda g: (g.reshape(self.shape),)) if current() else value
    def squeeze(self, dim=_layout.UNSPECIFIED_DIM):
        value = super().squeeze(dim)
        return current().record(value, (self,), lambda g: (g.reshape(self.shape),)) if current() else value
    def sum(self, dim=_layout.UNSPECIFIED_DIM, keepdim=_layout.UNSPECIFIED_DIM, *, dtype=None):
        value = super().sum(dim, keepdim, dtype=dtype)
        context = current()
        if context is None:
            return value
        requested, keep = _reductions.arguments(dim, keepdim, name='sum')
        axes = _reductions.axes(requested, self.ndim)
        def backward(g):
            if not keep:
                for axis in sorted(axes):
                    g = g.unsqueeze(axis)
            return (g * context.constant(np.ones(self.shape, dtype=self.dtype)),)
        return context.record(value, (self,), backward)
    def to(self, *args, **kwargs):
        value = super().to(*args, **kwargs)
        context = current()
        return context.record(value, (self,), lambda g: (g.to(dtype=self.dtype),)) if context else value


class Context:
    def __init__(self):
        self.rules = {}
        self.owners = []
        self.has_norm = False
        self.used = False
        self.reverse_active = False

    def clear(self):
        self.rules.clear()
        self.owners.clear()

    def wrap(self, value):
        if type(value) is _tensor.Tensor:
            value.__class__ = RecordedTensor
        return value

    def constant(self, value, dtype=None):
        return self.wrap(_tensor.Tensor(np.asarray(value, dtype=dtype), _metadata_owned=True))

    def record(self, value, inputs, backward):
        value = self.wrap(value)
        if not value.requires_grad or any(value is item for item in inputs):
            return value
        node = _graph.node_for(value)
        active = tuple(item.requires_grad for item in inputs)
        wanted = [id(_graph.node_for(item)) for item, needed in zip(inputs, active) if needed]
        if [id(parent) for parent, _, _ in node._parents] != wanted:
            # A composite primitive must record its actual internal operations.
            return value
        self.rules[id(node)] = (inputs, active, backward)
        self.owners.append(value)
        return value

    def where(self, mask, left, right):
        if not isinstance(left, _tensor.Tensor):
            left = self.constant(left, right.dtype)
        if not isinstance(right, _tensor.Tensor):
            right = self.constant(right, left.dtype)
        mask = np.asarray(mask, dtype=bool)
        condition = self.constant(mask)
        value = _ops.where(condition, left, right)
        return self.record(value, (left, right), lambda g: (
            unbroadcast(self.where(mask, g, 0.), left.shape),
            unbroadcast(self.where(mask, 0., g), right.shape)))

    def sign(self, source):
        data = (source._data > 0).astype(source.dtype) - (source._data < 0).astype(source.dtype)
        value = _autograd.result(data, (source, lambda g: _graph.LOGICAL_ZERO), saves_tensors=False, zero_passthrough=True)
        return self.record(value, (source,), lambda g: (_graph.LOGICAL_ZERO,))

    def norm(self, value, source, p, axes, keepdim):
        self.has_norm = True
        def restore(item):
            if not keepdim:
                for axis in sorted(axes):
                    item = item.unsqueeze(axis)
            return item
        def backward(g):
            if p == 0:
                return (None,)
            grad, length = restore(g), restore(value)
            if p == 1:
                result = self.sign(source) * grad
            elif p == 2:
                result = grad * self.where(length._data == 0, 0., source / length)
            elif np.isinf(p):
                selected = (np.abs(source._data) == length._data) | np.isnan(source._data)
                mask = self.constant(selected.astype(source.dtype))
                count = self.constant(selected.sum(axis=axes, keepdims=True).astype(source.dtype))
                result = self.sign(source) * ((grad / count) * mask)
            elif p < 1:
                scaled = self.sign(source) * self.where(source._data == 0, 0., source.abs() ** (p - 1))
                result = scaled * grad * length ** (1 - p)
            else:
                scaled = self.sign(source) * source.abs() ** (p - 1) if p < 2 else source * source.abs() ** (p - 2)
                scale = self.where(length._data == 0, 0., grad / length ** (p - 1))
                result = scaled * scale
            return (result,)
        return self.record(value, (source,), backward)

    def order(self, outputs):
        result, seen, stack = [], set(), [(_graph.node_for(value), False) for value in outputs if value is not None and value.requires_grad]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                result.append(node)
            elif id(node) not in seen:
                seen.add(id(node));stack.append((node, True))
                stack.extend((parent, False) for parent, _, _ in node._parents)
        return result

    def can_reverse(self, output):
        if not self.has_norm:
            return False
        for node in self.order((output,)):
            if node._parents:
                if id(node) not in self.rules or node.callbacks is not None:
                    return False
            elif getattr(node, '_backward_hooks', None) or getattr(node, '_post_accumulate_grad_hooks', None):
                return False
        return True

    def grad(self, outputs, inputs, seeds=None):
        targets = {id(_graph.node_for(value)) for value in inputs}
        order = self.order(outputs)
        needed = set(targets)
        for node in order:
            if any(id(parent) in needed for parent, _, _ in node._parents):
                needed.add(id(node))
        pending, returned = {}, {}
        def add(node, gradient):
            if gradient is None or id(node) not in needed:
                return
            key = id(node)
            if gradient is _graph.LOGICAL_ZERO:
                pending.setdefault(key, gradient)
            elif key not in pending or pending[key] is _graph.LOGICAL_ZERO:
                pending[key] = gradient
            else:
                pending[key] = pending[key] + gradient
        if seeds is None:
            seeds = tuple(None if value is None else self.constant(np.ones(value.shape, dtype=value.dtype)) for value in outputs)
        for value, seed in zip(outputs, seeds):
            if value is not None and value.requires_grad:
                add(_graph.node_for(value), seed)
        for node in reversed(order):
            key = id(node)
            gradient = pending.pop(key, None)
            if gradient is None or key not in needed:
                continue
            if key in targets:
                returned[key] = self.constant(np.zeros(node.shape, dtype=node.dtype)) if gradient is _graph.LOGICAL_ZERO else gradient
            if not node._parents:
                continue
            if node.released:
                raise RuntimeError('Saved tensors were freed before the higher-order traversal')
            if node.versions and any(counter.version != version for counter, version in node.versions):
                raise _errors.BrowserUnsupported('A tensor saved for backward was modified in-place')
            if gradient is _graph.LOGICAL_ZERO:
                if node._logical_zero_passthrough:
                    for parent, _, _ in node._parents:
                        add(parent, gradient)
                    continue
                gradient = self.constant(np.zeros(node.shape, dtype=node.dtype))
            if key not in self.rules:
                raise _errors.BrowserUnsupported('A higher-order pullback has no validated Tensor rule')
            parents, active, backward = self.rules[key]
            gradients = backward(gradient)
            assert len(gradients) == len(parents)
            for parent, enabled, value in zip(parents, active, gradients):
                if enabled and value is not None:
                    if value is not _graph.LOGICAL_ZERO:
                        if value.shape != parent.shape:
                            raise RuntimeError('A higher-order pullback changed its input gradient shape')
                        if value.dtype != parent.dtype:
                            value = value.to(dtype=parent.dtype)
                    add(_graph.node_for(parent), value)
        return tuple(returned.get(id(_graph.node_for(value))) for value in inputs)

    def hvp(self, output, inputs, vectors, strict):
        previous = self.reverse_active
        self.reverse_active = True
        try:
            jacobian = self.grad((output,), inputs)
            if strict and any(value is None or not value.requires_grad for value in jacobian):
                raise RuntimeError('An HVP Jacobian is independent of an input in strict mode')
            dummy = tuple(self.constant(np.zeros(value.shape, dtype=value.dtype)).requires_grad_() for value in inputs)
            double_backward = self.grad(jacobian, inputs, dummy)
            result = self.grad(double_backward, dummy, vectors)
            self.used = True
            return result
        finally:
            self.reverse_active = previous
