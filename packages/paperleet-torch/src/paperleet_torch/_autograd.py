"""NumPy-backed autograd implementation; extracted from PaperLeet."""
import contextlib
import math
import numpy as np


from . import _tensor

def result(data, *edges, saved=()):
    if any(isinstance(node, ForwardTensor) for node in (*[node for node, _ in edges], *saved)):
        raise _errors.BrowserUnsupported('This operation has no validated forward-mode rule; use the server')
    value = _tensor.Tensor(data)
    if value.dtype.kind == 'c' and _state._grad_enabled and any(node.requires_grad for node, _ in edges):
        raise _errors.BrowserUnsupported('Complex autograd requires the server')
    value._parents = tuple((node, vjp, node.dtype) for node, vjp in edges if node.requires_grad and _state._grad_enabled)
    value.requires_grad = bool(value._parents)
    if value.requires_grad:
        # Conservatively guard every operand, including frozen operands and output
        # arrays captured by VJP closures. Unsupported graph reuse falls back rather
        # than returning derivatives calculated from mutated storage.
        counters = {node._storage for node in (value, *(node for node, _ in edges), *saved)}
        value._saved_versions = tuple((counter, counter.version) for counter in counters)
    return value


def joint_result(data, inputs, backward, *, saved=()):
    """Internal multi-input VJP, evaluated once per backward traversal.

    ``backward(seed, needs)`` returns one entry per input, using None only for
    frozen inputs. Needs are captured at graph construction, not recomputed
    from mutable leaf flags. All operands retain the normal version guards.
    No gradients or seeds are cached across traversals.
    """
    inputs = tuple(inputs)
    value = result(data, *((node, None) for node in inputs), saved=saved)
    if value.requires_grad:
        needs = tuple(node.requires_grad and _state._grad_enabled for node in inputs)
        value._joint_vjp = (backward, needs)
    return value


def backprop(output, seed):
    if not output.requires_grad:
        raise RuntimeError('Tensor does not require grad and has no grad_fn')
    order, visited, stack = [], set(), [(output, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
        elif id(node) not in visited:
            visited.add(id(node))
            stack.append((node, True))
            stack.extend((parent, False) for parent, _, _ in node._parents)
    for node in order:
        if any(counter.version != version for counter, version in node._saved_versions):
            raise _errors.BrowserUnsupported('A tensor saved for backward was modified in-place; this graph requires the server')
    gradients = {id(output): np.asarray(seed, dtype=output.dtype)}
    for node in reversed(order):
        grad = gradients.get(id(node))
        if grad is None:
            continue
        if node._joint_vjp is None:
            # Keep ordinary edges on their original hot path: no joint dispatch
            # or shape-validation branch per edge in the WASM interpreter.
            for parent, vjp, dtype in node._parents:
                incoming = np.asarray(vjp(grad), dtype=dtype)
                key = id(parent)
                gradients[key] = incoming if key not in gradients else gradients[key] + incoming
            continue
        backward, needs = node._joint_vjp
        derivatives = backward(grad, needs)
        if not isinstance(derivatives, (tuple, list)) or len(derivatives) != len(needs):
            raise RuntimeError('A joint VJP must return one gradient per input')
        if any(needed and derivative is None for needed, derivative in zip(needs, derivatives)):
            raise RuntimeError('A joint VJP omitted a required input gradient')
        active = (derivative for needed, derivative in zip(needs, derivatives) if needed)
        for (parent, _, dtype), derivative in zip(node._parents, active):
            incoming = np.asarray(derivative, dtype=dtype)
            if incoming.shape != parent.shape:
                raise RuntimeError('A joint VJP gradient shape must match its input')
            key = id(parent)
            gradients[key] = incoming if key not in gradients else gradients[key] + incoming
    return order, gradients


class ForwardTensor(_tensor.Tensor):
    """One forward tangent; both channels retain ordinary reverse-mode graphs."""
    def __init__(self, primal, tangent):
        if primal.dtype not in (np.dtype('float32'), np.dtype('float64')) or tangent.shape != primal.shape:
            raise _errors.BrowserUnsupported('Forward mode requires matching real floating tensors')
        self.__dict__ = primal.__dict__.copy()
        self._primal_value = primal
        self._tangent_value = tangent.to(dtype=primal.dtype) if tangent.dtype != primal.dtype else tangent
        self._forward_level = _state._jvp_level

    @property
    def _primal(self):
        if not _state._jvp_active or self._forward_level != _state._jvp_level:
            raise _errors.BrowserUnsupported('An escaped forward tensor requires the server')
        return self._primal_value

    @property
    def _tangent(self):
        if not _state._jvp_active or self._forward_level != _state._jvp_level:
            raise _errors.BrowserUnsupported('An escaped forward tensor requires the server')
        return self._tangent_value

    def __add__(self, other):
        p, t = forward_parts(other)
        return ForwardTensor(self._primal + p, self._tangent + t)
    def __radd__(self, other): return self + other
    def __neg__(self): return ForwardTensor(-self._primal, -self._tangent)
    def __sub__(self, other): return self + -other
    def __rsub__(self, other): return -self + other
    def __mul__(self, other):
        p, t = forward_parts(other)
        return ForwardTensor(self._primal * p, self._tangent * p + self._primal * t)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):
        p, t = forward_parts(other)
        return ForwardTensor(self._primal / p, (self._tangent - (self._primal / p) * t) / p)
    def __rtruediv__(self, other):
        p, t = forward_parts(other)
        return ForwardTensor(p / self._primal, (t - (p / self._primal) * self._tangent) / self._primal)
    def __pow__(self, exponent):
        p = self._primal ** exponent
        t = self._tangent * (exponent * self._primal ** (exponent - 1)) if exponent else self._tangent * 0
        return ForwardTensor(p, t)
    def __rpow__(self, base):
        p = self._primal.__rpow__(base)
        return ForwardTensor(p, self._tangent * p * math.log(base))
    def __matmul__(self, other):
        p, t = forward_parts(other)
        return ForwardTensor(self._primal @ p, self._tangent @ p + self._primal @ t)
    def __rmatmul__(self, other):
        p, t = forward_parts(other)
        return ForwardTensor(p @ self._primal, t @ self._primal + p @ self._tangent)
    def __getitem__(self, index): return ForwardTensor(self._primal[index], self._tangent[index])
    def detach(self): return self._primal.detach()
    def numpy(self): raise _errors.BrowserUnsupported('NumPy escape during JVP requires the server')
    def __array__(self, *args, **kwargs): return self.numpy()
    def requires_grad_(self, *args): raise _errors.BrowserUnsupported('Changing gradient flags inside JVP requires the server')
    def backward(self, *args, **kwargs): raise _errors.BrowserUnsupported('Backward inside JVP requires the server')
    def mapped(self, name, *args, **kwargs):
        return ForwardTensor(getattr(self._primal, name)(*args, **kwargs), getattr(self._tangent, name)(*args, **kwargs))
    def to(self, *args, **kwargs):
        primal = self._primal.to(*args, **kwargs)
        if primal.dtype.kind != 'f': raise _errors.BrowserUnsupported('Nonfloating casts inside JVP require the server')
        tangent = self._tangent.to(*args, **kwargs)
        return self if primal is self._primal and tangent is self._tangent else ForwardTensor(primal, tangent)
    def exp(self):
        p = self._primal.exp()
        return ForwardTensor(p, self._tangent * p)
    def log(self): return ForwardTensor(self._primal.log(), self._tangent / self._primal)
    def sin(self): return ForwardTensor(self._primal.sin(), self._tangent * self._primal.cos())
    def cos(self): return ForwardTensor(self._primal.cos(), -self._tangent * self._primal.sin())
    def tanh(self):
        p = self._primal.tanh()
        return ForwardTensor(p, self._tangent * (1 - p.square()))
    def sigmoid(self):
        p = self._primal.sigmoid()
        return ForwardTensor(p, self._tangent * p * (1 - p))
    def logsigmoid(self):
        primal = self._primal
        value = primal.logsigmoid()
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            slope = np.exp(-np.logaddexp(np.zeros((),dtype=primal.dtype),primal._data))
            probability = 1 / (1 + np.exp(-primal._data))
            curvature = probability * (probability - 1)
        derivative = result(slope,(primal,lambda g:g*curvature))
        return ForwardTensor(value,self._tangent*derivative)
    def relu(self): return ForwardTensor(self._primal.relu(), _ops.where(self._primal <= 0, 0., self._tangent))
    def softmax(self, dim, dtype=None):
        p = self._primal.softmax(dim, dtype)
        t = self._tangent.to(dtype=p.dtype)
        return ForwardTensor(p, p * (t - (t * p).sum(dim, True)))
    def log_softmax(self, dim, dtype=None):
        p = self._primal.log_softmax(dim, dtype)
        t = self._tangent.to(dtype=p.dtype)
        return ForwardTensor(p, t - (t * self._primal.softmax(dim, dtype)).sum(dim, True))


for _forward_name in ('clone', 'reshape', 'flatten', 'unsqueeze', 'squeeze', 'permute', 'transpose',
                      'expand', 'repeat', 'flip', 'roll', 'diag', 'diagonal', 'sum', 'mean'):
    def _forward_map(self, *args, _name=_forward_name, **kwargs): return self.mapped(_name, *args, **kwargs)
    setattr(ForwardTensor, _forward_name, _forward_map)


ForwardTensor.view = ForwardTensor.reshape


def forward_parts(value):
    if isinstance(value, ForwardTensor): return value._primal, value._tangent
    value = _ops.wrap(value)
    return value, _tensor.Tensor(np.zeros_like(value._data))


def jvp(func, primals, tangents, *, strict=False, has_aux=False):
    """Exact, single-call forward AD for the validated teaching subset."""

    if _state._jvp_active or has_aux: raise _errors.BrowserUnsupported('Nested JVP and auxiliary output mode require the server')
    if not isinstance(primals, tuple) or not isinstance(tangents, tuple) or not primals or len(primals) != len(tangents):
        raise RuntimeError('JVP requires matching nonempty input/tangent tuples')
    for p, t in zip(primals, tangents):
        if not isinstance(p, _tensor.Tensor) or not isinstance(t, _tensor.Tensor): raise _errors.BrowserUnsupported('Nested/non-tensor JVP inputs require the server')
        if isinstance(p, ForwardTensor) or isinstance(t, ForwardTensor): raise _errors.BrowserUnsupported('Escaped/nested JVP inputs require the server')
        if p.shape != t.shape or p.dtype != t.dtype or p.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise _errors.BrowserUnsupported('JVP inputs and tangents must have matching real shapes and dtypes')
    def unpack(output):
        if isinstance(output, ForwardTensor): return output._primal, output._tangent
        if isinstance(output, _tensor.Tensor):
            if strict: raise RuntimeError('JVP output is independent of its inputs')
            return output, _tensor.Tensor(np.zeros_like(output._data))
        if isinstance(output, (tuple, list)) and output:
            pairs = [unpack(v) for v in output]
            return type(output)(v[0] for v in pairs), type(output)(v[1] for v in pairs)
        if isinstance(output, dict) and output:
            pairs = {key: unpack(v) for key, v in output.items()}
            return {key: v[0] for key, v in pairs.items()}, {key: v[1] for key, v in pairs.items()}
        raise _errors.BrowserUnsupported('This JVP output structure requires the server')
    _state._jvp_active = True
    _state._jvp_level += 1
    try:
        return unpack(func(*(ForwardTensor(p, t) for p, t in zip(primals, tangents))))
    finally:
        _state._jvp_active = False


def autograd_grad(output, inputs, grad_outputs=None, *, retain_graph=False, create_graph=False, allow_unused=False):
    if create_graph: raise _errors.BrowserUnsupported('Higher-order autograd requires the server')
    if isinstance(output, (tuple, list)):
        if len(output) != 1: raise _errors.BrowserUnsupported('Multiple-output autograd requires the server')
        output = output[0]
    if grad_outputs is None and output.numel() != 1: raise RuntimeError('Non-scalar outputs require grad_outputs')
    _, gradients = backprop(output, np.ones_like(output._data) if grad_outputs is None else _ops.wrap(grad_outputs)._data)
    values = []
    for node in ([inputs] if isinstance(inputs, _tensor.Tensor) else inputs):
        if id(node) not in gradients:
            if not allow_unused: raise RuntimeError('A differentiated tensor was not used in the graph')
            values.append(None)
        else: values.append(_tensor.Tensor(gradients[id(node)]))
    return tuple(values)


def gradcheck(func, inputs, *, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True, **options):
    """Bounded full first-order Jacobian check; never substitutes training VJPs."""
    if options: raise _errors.BrowserUnsupported('Advanced gradcheck options require the server')
    inputs = tuple(inputs) if isinstance(inputs, (list, tuple)) else (inputs,)
    if any(not isinstance(x, _tensor.Tensor) or x.dtype != np.dtype('float64') for x in inputs):
        raise _errors.BrowserUnsupported('gradcheck requires real float64 tensor inputs')
    active = [i for i,x in enumerate(inputs) if x.requires_grad]
    if not active: raise ValueError('gradcheck requires a differentiable input')
    if sum(inputs[i].numel() for i in active) > 256:
        raise _errors.BrowserUnsupported('Large gradcheck inputs require the server')
    for i in active:
        x = inputs[i]
        offsets = {sum(index * stride for index,stride in zip(position, x._data.strides)) for position in np.ndindex(x.shape)}
        if len(offsets) != x.numel(): raise _errors.BrowserUnsupported('Overlapping gradcheck storage requires the server')
    if not np.isfinite([eps, atol, rtol]).all() or eps <= 0 or min(atol, rtol) < 0:
        raise ValueError('Invalid gradcheck tolerances')
    def evaluate(values):
        output = func(*values)
        output = tuple(output) if isinstance(output, (list, tuple)) else (output,)
        if not output or any(not isinstance(x, _tensor.Tensor) or x.dtype != np.dtype('float64') for x in output):
            raise _errors.BrowserUnsupported('gradcheck requires real float64 tensor outputs')
        if sum(x.numel() for x in output) > 256: raise _errors.BrowserUnsupported('Large gradcheck outputs require the server')
        return output
    # Cloned independent leaves prevent perturbations or .grad accumulation in
    # the caller. Reject shared input storage rather than mis-check aliases.
    if any(np.shares_memory(a._data, b._data) for i,a in enumerate(inputs) for b in inputs[i+1:]):
        raise _errors.BrowserUnsupported('Overlapping gradcheck inputs require the server')
    leaves = [_tensor.Tensor(x._data.copy(), requires_grad=x.requires_grad) for x in inputs]
    outputs = evaluate(leaves)
    size = sum(x.numel() for x in outputs)
    analytical = [np.zeros((size, leaves[i].numel())) for i in active]
    row = 0
    for output in outputs:
        for index in range(output.numel()):
            if output.requires_grad:
                seed = np.zeros(output.shape); seed.flat[index] = 1
                derivatives = autograd_grad(output, [leaves[i] for i in active], _tensor.Tensor(seed), allow_unused=True)
                for matrix, derivative in zip(analytical, derivatives):
                    if derivative is not None: matrix[row] = derivative._data.reshape(-1)
            row += 1
    def numerical_output(values):
        with no_grad(): evaluated = evaluate(values)
        if len(evaluated) != len(outputs) or any(a.shape != b.shape for a,b in zip(evaluated, outputs)):
            raise RuntimeError('gradcheck output shape changed during perturbation')
        return np.concatenate([x._data.reshape(-1) for x in evaluated])
    baseline = np.concatenate([x._data.reshape(-1) for x in outputs])
    success = np.allclose(numerical_output(leaves), baseline, atol=atol, rtol=rtol)
    for matrix, index in zip(analytical, active):
        numerical = np.empty_like(matrix)
        for column in range(leaves[index].numel()):
            plus = [_tensor.Tensor(x._data.copy(), requires_grad=x.requires_grad) for x in leaves]
            minus = [_tensor.Tensor(x._data.copy(), requires_grad=x.requires_grad) for x in leaves]
            plus[index]._data.flat[column] += eps
            minus[index]._data.flat[column] -= eps
            numerical[:,column] = (numerical_output(plus) - numerical_output(minus)) / (2 * eps)
        success = bool(success and np.allclose(matrix, numerical, atol=atol, rtol=rtol))
    if not success and raise_exception: raise RuntimeError('gradcheck Jacobian mismatch between analytical and numerical derivatives')
    return bool(success)


from ._grad_mode import no_grad, enable_grad, set_grad_enabled, is_grad_enabled


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _errors, _ops, _state
