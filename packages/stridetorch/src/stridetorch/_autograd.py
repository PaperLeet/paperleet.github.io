"""NumPy-backed autograd implementation; extracted from PaperLeet."""
import contextlib
import math
import warnings
import numpy as np


from . import _arithmetic, _graph, _layout, _tensor

def result(data, *edges, saved=(), saves_tensors=True, versioned=None):
    single = len(edges) == 1
    if single:
        node, vjp = edges[0]
        forward = isinstance(node, ForwardTensor)
    else:forward = any(isinstance(node, ForwardTensor) for node, _ in edges)
    if forward or saved and any(isinstance(node, ForwardTensor) for node in saved):
        raise _errors.BrowserUnsupported('This operation has no validated forward-mode rule; use the server')
    value = _tensor.Tensor(data, _metadata_owned=True)
    if not _state._grad_enabled:return value
    if value.dtype.kind == 'c' and any(node.requires_grad for node, _ in edges):
        raise _errors.BrowserUnsupported('Complex autograd requires the server')
    if single:
        parents = ((_graph.node_for(node), vjp, node.dtype),) if node.requires_grad else ()
    else:parents = tuple((_graph.node_for(node), vjp, node.dtype) for node, vjp in edges if node.requires_grad)
    if parents:
        value._requires_grad = True
        # Default to conservative guards for arrays captured by VJP closures.
        # Proven selective rules can name only the inputs their pullbacks save.
        if saves_tensors:
            operands = (value, *(node for node, _ in edges), *saved) if versioned is None else versioned
            counters = {node._storage for node in operands}
        else:
            counters = ()
        value._grad_node = _graph.Node(value, parents, saved=saves_tensors,
                                      versions=tuple((counter, counter.version) for counter in counters))
    return value



def view_result(data, parent, backward):
    """One-input metadata pullback, including reshape's possible storage copy.

    These rules save only shape/index metadata. They never retain operand
    values or require storage-version guards; value-saving operations use result.
    """
    if isinstance(parent,ForwardTensor):
        raise _errors.BrowserUnsupported('This operation has no validated forward-mode rule; use the server')
    value=_tensor.Tensor(data,_metadata_owned=True)
    shared=value._storage is parent._storage
    if shared:
        base=parent._view_base
        value._view_base=parent if base is None else base
    if _state._grad_enabled and parent.requires_grad:
        if value.dtype.kind=='c':
            raise _errors.BrowserUnsupported('Complex autograd requires the server')
        value._requires_grad=True
        value._grad_node=_graph.Node(value,((_graph.node_for(parent),backward,parent.dtype),),saved=False)
    elif shared:
        value._requires_grad=None
        value._no_grad_view=True
    return value


def joint_result(data, inputs, backward, *, saved=(), versioned=None, prune_inputs=False):
    """Internal multi-input VJP, evaluated once per backward traversal.

    ``backward(seed, needs)`` returns one entry per input, using None only for
    frozen inputs. Needs are captured at graph construction, not recomputed
    from mutable leaf flags. Normal guards cover all operands unless the caller
    supplies an explicit, validated ``versioned`` selection.
    Validated rules may opt into traversal-specific input pruning; other rules
    continue receiving construction-time needs. No gradients or seeds are cached.
    """
    inputs = tuple(inputs)
    if any(isinstance(node, ForwardTensor) for node in (*inputs, *saved)):
        raise _errors.BrowserUnsupported('This operation has no validated forward-mode rule; use the server')
    value = _tensor.Tensor(data, _metadata_owned=True)
    if not _state._grad_enabled:return value
    needs = tuple(node.requires_grad for node in inputs)
    if not any(needs):return value
    if value.dtype.kind == 'c':
        raise _errors.BrowserUnsupported('Complex autograd requires the server')
    # Build active edges and shared metadata once instead of constructing a
    # temporary per-edge VJP list and then replacing its backward descriptor.
    parents = tuple((_graph.node_for(node), None, node.dtype)
                    for node, active in zip(inputs, needs) if active)
    operands = (value, *inputs, *saved) if versioned is None else versioned
    counters = {node._storage for node in operands}
    value._requires_grad = True
    value._grad_node = _graph.Node(value, parents, saved=True,
        versions=tuple((counter, counter.version) for counter in counters),
        joint=(backward, needs, True) if prune_inputs else (backward, needs))
    return value


def backprop(output, seed, *, retain_graph=False, inputs=None, accumulate=False):
    outputs = (output,) if isinstance(output, _tensor.Tensor) else tuple(output)
    seeds = (seed,) if isinstance(output, _tensor.Tensor) else tuple(seed)
    roots = tuple(_graph.node_for(value) for value in outputs)
    targets = None if inputs is None else {id(_graph.node_for(value)) for value in inputs}
    return _graph.run(roots, seeds, retain_graph=retain_graph, targets=targets, accumulate=accumulate)


def match_tangent(tangent, primal):
    if tangent.dtype != primal.dtype: tangent = tangent.to(dtype=primal.dtype)
    return tangent if tangent.shape == primal.shape else tangent.expand(primal.shape)


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
        raw = other._primal if isinstance(other, ForwardTensor) else other
        primal = self._primal + raw
        tangent = self._tangent + other._tangent if isinstance(other, ForwardTensor) else match_tangent(self._tangent, primal)
        return ForwardTensor(primal, tangent)
    def __radd__(self, other): return self + other
    def __neg__(self): return ForwardTensor(-self._primal, -self._tangent)
    def __sub__(self, other):
        raw = other._primal if isinstance(other, ForwardTensor) else other
        primal = self._primal - raw
        tangent = self._tangent - other._tangent if isinstance(other, ForwardTensor) else match_tangent(self._tangent, primal)
        return ForwardTensor(primal, tangent)
    def __rsub__(self, other):
        raw = other._primal if isinstance(other, ForwardTensor) else other
        primal = raw - self._primal
        tangent = other._tangent - self._tangent if isinstance(other, ForwardTensor) else match_tangent(-self._tangent, primal)
        return ForwardTensor(primal, tangent)
    def __mul__(self, other):
        p = _arithmetic.forward_primal(self._primal, other)
        primal = self._primal * p
        tangent = self._tangent * p
        if isinstance(other, ForwardTensor): tangent = tangent + self._primal * other._tangent
        return ForwardTensor(primal, tangent)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):
        p = _arithmetic.forward_primal(self._primal, other)
        primal = self._primal / p
        tangent = (self._tangent - primal * other._tangent) / p if isinstance(other, ForwardTensor) else self._tangent / p
        return ForwardTensor(primal, tangent)
    def __rtruediv__(self, other):
        if not isinstance(other, _tensor.Tensor): return self.reciprocal() * other
        p = other._primal if isinstance(other, ForwardTensor) else other
        primal = p / self._primal
        numerator = other._tangent.to(dtype=primal.dtype) - primal * self._tangent if isinstance(other, ForwardTensor) else -primal * self._tangent
        return ForwardTensor(primal, numerator / self._primal)
    def reciprocal(self):
        with np.errstate(all='ignore'):
            primal = self._primal.reciprocal()
            tangent = self._tangent * -(primal * primal)
        return ForwardTensor(primal, tangent)
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
        primal=getattr(self._primal,name)(*args,**kwargs)
        tangent=self._tangent
        if name in ('flatten','unsqueeze','squeeze','permute','transpose','expand'):
            tangent=_layout.matching_tangent(tangent,self._primal)
        return ForwardTensor(primal,getattr(tangent,name)(*args,**kwargs))
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
    def var(self, *args, dim=_layout.UNSPECIFIED_DIM, unbiased=_layout.UNSPECIFIED_DIM,
            keepdim=_layout.UNSPECIFIED_DIM, correction=_layout.UNSPECIFIED_DIM):
        from . import _statistics
        return _statistics.reduce(self, *args, dim=dim, unbiased=unbiased,
                                  keepdim=keepdim, correction=correction)
    def std(self, *args, dim=_layout.UNSPECIFIED_DIM, unbiased=_layout.UNSPECIFIED_DIM,
            keepdim=_layout.UNSPECIFIED_DIM, correction=_layout.UNSPECIFIED_DIM):
        from . import _statistics
        return _statistics.reduce(self, *args, dim=dim, unbiased=unbiased,
                                  keepdim=keepdim, correction=correction, kind='std')
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


def _forward_view(self, *shape, **kwargs):
    primal = self._primal.view(*shape, **kwargs)
    tangent = _layout.matching_tangent(self._tangent, self._primal).view(primal.shape)
    return ForwardTensor(primal, tangent)


def _forward_view_as(self, other):
    if not isinstance(other, _tensor.Tensor): raise TypeError('view_as requires a Tensor')
    return self.view(other.shape)


ForwardTensor.view = _forward_view
ForwardTensor.view_as = _forward_view_as


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
            if isinstance(output,tuple) and hasattr(output,'_fields'):
                return type(output)(*(v[0] for v in pairs)), type(output)(*(v[1] for v in pairs))
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


def _backward_arguments(outputs, gradients, retain_graph, create_graph):
    if create_graph:raise _errors.BrowserUnsupported('Higher-order autograd requires the server')
    if _state._jvp_active:raise _errors.BrowserUnsupported('Backward inside a forward AD transform is unsupported')
    if retain_graph is not None and not isinstance(retain_graph,bool):raise TypeError('retain_graph must be bool or None')
    outputs = (outputs,) if isinstance(outputs,_tensor.Tensor) else tuple(outputs)
    if gradients is None:gradients = (None,)*len(outputs)
    elif isinstance(gradients,_tensor.Tensor):gradients = (gradients,)
    else:gradients = tuple(gradients)
    if len(outputs)!=len(gradients):raise RuntimeError('Expected one gradient per output')
    seeds=[]
    for output,gradient in zip(outputs,gradients):
        if not isinstance(output,_tensor.Tensor):raise TypeError('Outputs must be Tensors')
        if not output.requires_grad or output._no_grad_view:raise RuntimeError('Tensor does not require grad and has no grad_fn')
        if gradient is None:
            if output.numel()!=1:raise RuntimeError('Non-scalar outputs require grad_outputs')
            seed=np.ones_like(output._data)
        else:
            if not isinstance(gradient,_tensor.Tensor):raise TypeError('Gradient must be a Tensor or None')
            if gradient.shape!=output.shape:raise RuntimeError('Gradient shape must match its output')
            if gradient.dtype.kind=='c':raise _errors.BrowserUnsupported('Complex backward seeds are unsupported')
            if gradient.dtype == output.dtype: seed=gradient._data
            else:
                with np.errstate(all='ignore'): seed=np.asarray(gradient._data,dtype=output.dtype)
        seeds.append(seed)
    return outputs,tuple(seeds),bool(retain_graph)


def _backward_inputs(inputs):
    inputs=(inputs,) if isinstance(inputs,_tensor.Tensor) else tuple(inputs)
    if not inputs:raise RuntimeError('Expected nonempty differentiation inputs')
    for value in inputs:
        if not isinstance(value,_tensor.Tensor):raise TypeError('Differentiation inputs must be Tensors')
        if not value.requires_grad:raise RuntimeError('A differentiated Tensor does not require grad')
    return inputs


def autograd_backward(tensors,grad_tensors=None,retain_graph=None,create_graph=False,
                      grad_variables=None,inputs=None):
    if grad_variables is not None:raise _errors.BrowserUnsupported('Deprecated grad_variables is unsupported')
    outputs,seeds,retain_graph=_backward_arguments(tensors,grad_tensors,retain_graph,create_graph)
    inputs=None if inputs is None else _backward_inputs(inputs)
    if inputs is not None:
        for value in inputs:value.retain_grad()
    backprop(outputs,seeds,retain_graph=retain_graph,inputs=inputs,accumulate=True)


def autograd_grad(output, inputs, grad_outputs=None, retain_graph=None, create_graph=False,
                  only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False):
    if is_grads_batched:raise _errors.BrowserUnsupported('Batched backward seeds are unsupported')
    if not only_inputs:warnings.warn('only_inputs is deprecated and ignored',FutureWarning,stacklevel=2)
    if materialize_grads and allow_unused is False:raise ValueError('materialize_grads requires allow_unused=True')
    allow_unused=materialize_grads if allow_unused is None else allow_unused
    outputs,seeds,retain_graph=_backward_arguments(output,grad_outputs,retain_graph,create_graph)
    keys=tuple(inputs) if type(inputs) is dict else None
    inputs=_backward_inputs(tuple(inputs.values()) if keys is not None else inputs)
    order,gradients=backprop(outputs,seeds,retain_graph=retain_graph,inputs=inputs)
    values=[]
    for value in inputs:
        key=id(_graph.node_for(value))
        if key not in gradients:
            if not allow_unused:raise RuntimeError('A differentiated tensor was not used in the graph')
            values.append(_tensor.Tensor(np.zeros_like(value._data)) if materialize_grads else None)
        else:
            value=gradients[key]
            values.append(value if isinstance(value,_tensor.Tensor) else _tensor.Tensor(value))
    return tuple(values) if keys is None else dict(zip(keys,values))


def functional_vjp(func, inputs, v=None, create_graph=False, strict=False):
    """First-order functional VJP with isolated input objects and no .grad writes."""
    if create_graph:
        raise _errors.BrowserUnsupported('Higher-order functional VJP requires the server')
    if _state._jvp_active:
        raise _errors.BrowserUnsupported('Functional VJP inside forward AD requires the server')

    def tensors(value, name):
        was_tuple = isinstance(value, tuple)
        values = value if was_tuple else (value,)
        if any(not isinstance(item, _tensor.Tensor) for item in values):
            raise TypeError(name + ' must be a Tensor or a tuple of Tensors')
        return values, was_tuple

    with enable_grad():
        original, inputs_tuple = tensors(inputs, 'inputs')
        if any(isinstance(item, ForwardTensor) or item.dtype.kind == 'c' for item in original):
            raise _errors.BrowserUnsupported('Complex or forward-AD functional VJP inputs require the server')
        # Distinct leaves separate explicit arguments from captured originals,
        # including repeated arguments that happen to share the same storage.
        arguments = tuple(item.detach().requires_grad_(True) for item in original)
        previous_transform = _state._functional_vjp_active
        _state._functional_vjp_active = True
        try:
            returned = func(*arguments)
        finally:
            _state._functional_vjp_active = previous_transform
        outputs, outputs_tuple = tensors(returned, 'outputs')
        if strict and any(not output.requires_grad for output in outputs):
            raise RuntimeError('A functional VJP output does not require gradients in strict mode')
        if v is None:
            if len(outputs) != 1 or outputs[0].numel() != 1:
                raise RuntimeError('Implicit VJP seeds require one output with one element')
            seeds = (None,)
        else:
            seeds, _ = tensors(v, 'v')
            if len(seeds) != len(outputs):
                raise RuntimeError('Expected one VJP seed per output')
            if any(seed.shape != output.shape for seed, output in zip(seeds, outputs)):
                raise RuntimeError('VJP seed shapes must match their outputs')
        active = tuple((output, seed) for output, seed in zip(outputs, seeds) if output.requires_grad)
        if active and arguments:
            gradients = autograd_grad(tuple(item[0] for item in active), arguments,
                                      tuple(item[1] for item in active), allow_unused=True)
        else:
            gradients = (None,) * len(arguments)
        if strict and any(value is None for value in gradients):
            raise RuntimeError('A functional VJP input is independent of all outputs in strict mode')
        # Native functional VJP rejects empty Tensor tuples during postprocessing.
        if not outputs or not arguments:
            raise IndexError('Empty functional VJP Tensor tuple')
        gradients = tuple(_tensor.Tensor(np.zeros_like(argument._data)) if gradient is None
                          else gradient.detach() for argument, gradient in zip(arguments, gradients))
        outputs = tuple(output.detach() for output in outputs)
        return (outputs if outputs_tuple else outputs[0],
                gradients if inputs_tuple else gradients[0])


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
                derivatives = autograd_grad(output, [leaves[i] for i in active], _tensor.Tensor(seed), allow_unused=True, retain_graph=True)
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
