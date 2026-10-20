"""Scoped tensor pullbacks for native-order vector-norm Hessian products.

Ordinary graphs retain their existing numerical engine and metadata. Primitive
Tensor methods append scoped descriptors directly; no subclass wrapping is used. Rules and
temporary Tensor owners live only in one HVP context; nothing is attached to
normal nodes. A complete recorded primal graph is required before using the
three reverse traversals. Unsupported existing graphs retain their old path.
"""
import numpy as np

from . import _autograd, _errors, _graph, _layout, _ops, _promotion, _reductions, _state, _tensor


def current():
    return _state._tensor_reverse_context


def unbroadcast(gradient, shape):
    while gradient.ndim > len(shape):
        gradient = gradient.sum(0)
    axes = tuple(axis for axis, (actual, wanted) in enumerate(zip(gradient.shape, shape)) if wanted == 1 and actual != 1)
    if axes:
        gradient = gradient.sum(axes, True)
    return gradient.reshape(shape) if gradient.shape != shape else gradient


def binary_pullback(context, inputs, active, description):
    """Build a Tensor pullback only after a norm activates the private tape.

    The descriptor retains the original operands and construction-time flags.
    Pure numerical HVPs need neither this closure nor its captured cell objects.
    """
    source = inputs[0]
    if description[0] == 'scalar':
        _, other, operation, reverse = description
        def backward(g):
            if operation == 'add':
                gradient = g
            elif operation == 'sub':
                gradient = -g if reverse else g
            else:
                right = context.constant(other, source.dtype)
                gradient = g * right if operation == 'mul' else g / right
            return (unbroadcast(gradient, source.shape),)
        return backward
    _, operation, reverse = description
    right = inputs[1]
    left_active, right_active = active
    def backward(g):
        if operation == 'add':
            left_grad = g if left_active else None
            right_grad = g if right_active else None
        elif operation == 'sub':
            left_grad = (-g if reverse else g) if left_active else None
            right_grad = (g if reverse else -g) if right_active else None
        elif operation == 'mul':
            left_grad = g * right if left_active else None
            right_grad = g * source if right_active else None
        else:
            left_grad = g / right if left_active else None
            right_grad = -g * ((source / right) / right) if right_active else None
        return (unbroadcast(left_grad, source.shape) if left_grad is not None else None,
                unbroadcast(right_grad, right.shape) if right_grad is not None else None)
    return backward


def unary_pullback(context, inputs, description):
    source = inputs[0]
    _, operation, argument = description
    if operation == 'neg':return lambda g: (-g,)
    if operation == 'pow':
        def backward(g):
            if argument == 0:
                return (context.constant(np.zeros(source.shape, dtype=source.dtype)),)
            return (g * (argument * source ** (argument - 1)),)
        return backward
    if operation == 'reciprocal':return lambda g: (g * -(argument * argument),)
    if operation == 'exp':return lambda g: (g * argument,)
    if operation == 'sin':return lambda g: (g * source.cos(),)
    if operation == 'cos':return lambda g: (-g * source.sin(),)
    if operation == 'tanh':return lambda g: (g * (1 - argument ** 2),)
    if operation == 'abs':return lambda g: (g * context.sign(source),)
    if operation == 'relu':return lambda g: (context.where(source._data <= 0, 0., g),)
    if operation == 'reshape':return lambda g: (g.reshape(source.shape),)
    if operation == 'to':return lambda g: (g.to(dtype=source.dtype),)
    if operation == 'cumsum':
        from . import _reductions
        axis, reverse = argument
        return lambda g: (_reductions._scan_result(g.to(dtype=source.dtype), axis, source.dtype, reverse=not reverse),)
    if operation == 'sum':
        axes, keepdim = argument
        def backward(g):
            if not keepdim:
                for axis in sorted(axes):g = g.unsqueeze(axis)
            return (g * context.constant(np.ones(source.shape, dtype=source.dtype)),)
        return backward
    raise RuntimeError('Unknown Tensor pullback operation')


_BINARY_OPERATIONS = {np.add:'add', np.subtract:'sub', np.multiply:'mul', np.divide:'div'}


class Context:
    tensor_type = _tensor.Tensor
    binary_operations = _BINARY_OPERATIONS

    def __init__(self):
        self.rules = {}
        self.owners = []
        self.pending = []
        self.recording_primal = True
        self.has_norm = False
        self.used = False
        self.reverse_active = False

    def clear(self):
        self.rules.clear()
        self.pending.clear()
        self.owners.clear()

    def wrap(self, value):
        # Recording metadata never changes a Tensor's Python class.
        return value

    def constant(self, value, dtype=None):
        return _tensor.Tensor(np.asarray(value, dtype=dtype), _metadata_owned=True)

    def unary(self, value, source, operation, argument=None):
        if operation in ('reciprocal','exp','tanh'):
            argument = value
        description = ('unary', operation, argument)
        if self.recording_primal and type(value) is _tensor.Tensor:
            node = value._grad_node
            if node is not None and value is not source:
                # The primitive has just constructed this nonleaf. Preserve
                # the same operand flag snapshot and original node without
                # another generic record dispatch. Validate parents on use.
                self.pending.append((value, (source,), description, (source.requires_grad,), node))
                return value
        return self.record(value, (source,), description)


    def record(self, value, inputs, backward, active=None):
        value_type = type(value)
        node = value._grad_node if value_type is _tensor.Tensor else None
        if node is None and not value.requires_grad:
            return value
        count = len(inputs)
        if count == 1:
            if value is inputs[0]:
                return value
            if active is None:
                active = (inputs[0].requires_grad,)
        elif count == 2:
            if value is inputs[0] or value is inputs[1]:
                return value
            if active is None:
                active = (inputs[0].requires_grad, inputs[1].requires_grad)
        else:
            if any(value is item for item in inputs):
                return value
            if active is None:
                active = tuple(item.requires_grad for item in inputs)
        if node is None:
            node = _graph.node_for(value)
        if self.recording_primal:
            # Preserve construction-time flags and the original node. A
            # callback may change captured leaves before it returns. Pure
            # legacy HVPs never need a Tensor-rule map or parent validation.
            self.pending.append((value, inputs, backward, active, node))
            # The pending tuple already owns the result throughout primal AD.
            return value
        return self._store_rule(value, inputs, backward, active, node)

    def _store_rule(self, value, inputs, backward, active, node, retain_owner=True):
        count = len(inputs)
        parents = node._parents
        # Keep exact parent identities, order and duplicate edges. The common
        # unary/binary rules need neither temporary id lists nor generators.
        if count == 1:
            if active[0]:
                if len(parents) != 1 or parents[0][0] is not _graph.node_for(inputs[0]):
                    return value
            elif parents:
                return value
        elif count == 2:
            if active[0] and active[1]:
                if (len(parents) != 2 or parents[0][0] is not _graph.node_for(inputs[0])
                        or parents[1][0] is not _graph.node_for(inputs[1])):
                    return value
            elif active[0] or active[1]:
                parent = inputs[0] if active[0] else inputs[1]
                if len(parents) != 1 or parents[0][0] is not _graph.node_for(parent):
                    return value
            elif parents:
                return value
        else:
            wanted = [id(_graph.node_for(item)) for item, needed in zip(inputs, active) if needed]
            if [id(parent) for parent, _, _ in parents] != wanted:
                # A composite primitive must record its actual internal operations.
                return value
        if isinstance(backward, tuple):
            backward = (unary_pullback(self, inputs, backward) if backward[0] == 'unary'
                        else binary_pullback(self, inputs, active, backward))
        self.rules[id(node)] = (inputs, active, backward)
        if retain_owner:
            self.owners.append(value)
        return value

    def _activate(self, reachable=None):
        if not self.recording_primal:
            return
        self.recording_primal = False
        pending, self.pending = self.pending, []
        for value, inputs, backward, active, node in pending:
            # Only the primal output's ancestors participate in the first
            # reverse traversal. Tangent-only and discarded branches retain
            # their numerical graphs, but need no Tensor pullback or owner.
            if reachable is not None and id(node) not in reachable:
                continue
            self.owners.append(value)
            self._store_rule(value, inputs, backward, active, node, retain_owner=False)

    def where(self, mask, left, right):
        if not isinstance(left, _tensor.Tensor):
            left = self.constant(left, right.dtype)
        if not isinstance(right, _tensor.Tensor):
            right = self.constant(right, left.dtype)
        mask = np.asarray(mask, dtype=bool)
        condition = self.constant(mask)
        value = _ops.where(condition, left, right)
        active = (left.requires_grad, right.requires_grad)
        def backward(g):
            return (unbroadcast(self.where(mask, g, 0.), left.shape) if active[0] else None,
                    unbroadcast(self.where(mask, 0., g), right.shape) if active[1] else None)
        return self.record(value, (left, right), backward, active=active)

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
        order = self.order((output,))
        self._activate({id(node) for node in order})
        for node in order:
            if node._parents:
                if id(node) not in self.rules or node.callbacks is not None:
                    return False
            elif getattr(node, '_backward_hooks', None) or getattr(node, '_post_accumulate_grad_hooks', None):
                return False
        return order

    def grad(self, outputs, inputs, seeds=None, *, order=None):
        self._activate()
        targets = {id(_graph.node_for(value)) for value in inputs}
        # The first traversal can reuse the exact graph order validated just
        # before entering HVP. Later reverse graphs still need fresh orders.
        if order is None:
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

    def hvp(self, output, inputs, vectors, strict, *, order=None):
        previous = self.reverse_active
        self.reverse_active = True
        try:
            jacobian = self.grad((output,), inputs, order=order)
            if strict and any(value is None or not value.requires_grad for value in jacobian):
                raise RuntimeError('An HVP Jacobian is independent of an input in strict mode')
            dummy = tuple(self.constant(np.zeros(value.shape, dtype=value.dtype)).requires_grad_() for value in inputs)
            double_backward = self.grad(jacobian, inputs, dummy)
            result = self.grad(double_backward, dummy, vectors)
            self.used = True
            return result
        finally:
            self.reverse_active = previous
