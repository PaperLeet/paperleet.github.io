"""Reverse graph metadata, independent of intermediate Tensor storage."""
import numpy as np

from . import _errors, _tensor_hooks


class NumericalScope:
    """Batch kernel error handling; user hooks retain their mutable policy."""
    def __enter__(self):
        self.initial = self.policy = np.geterr()
        self.initial_handler = self.handler = np.geterrcall()
        self.used = False
        self.context = np.errstate(all='ignore')
        self.context.__enter__()
        return self

    def callback(self, gradient, hooks, owner=None):
        with np.errstate(**self.policy):
            if np.geterrcall() is not self.handler: np.seterrcall(self.handler)
            try:
                return _tensor_hooks.apply(gradient, hooks, owner)
            finally:
                self.policy, self.handler = np.geterr(), np.geterrcall()
                self.used = True

    def __exit__(self, *error):
        self.context.__exit__(*error)
        if self.used:
            if self.policy != self.initial: np.seterr(**self.policy)
            if self.handler is not self.initial_handler: np.seterrcall(self.handler)
        return False


class Node:
    __slots__ = ('_parents', 'joint', 'versions', 'shape', 'dtype',
                 'callbacks', 'saved', 'released')

    def __init__(self, tensor, parents, *, versions=(), saved=False, joint=None):
        self._parents, self.joint, self.versions = parents, joint, versions
        self.shape, self.dtype = tensor.shape, tensor.dtype
        self.saved, self.released, self.callbacks = saved, False, None

    def release(self):
        if self.saved:
            self._parents = tuple((parent, None, dtype) for parent, _, dtype in self._parents)
            self.joint, self.versions, self.released = None, (), True


class _StoppedView:
    """A missing gradient edge for a view created while recording was disabled."""
    __slots__ = ('shape','dtype')
    _parents = ()
    requires_grad = False
    _backward_hooks = None
    _post_accumulate_grad_hooks = None
    def __init__(self,tensor):self.shape,self.dtype=tensor.shape,tensor.dtype


def node_for(tensor):
    if tensor._no_grad_view:return _StoppedView(tensor)
    node = tensor._grad_node
    # A leaf already owns exactly the storage/gradient information needed by
    # the engine. Use it directly, preserving graph ownership without a cached
    # leaf-node object or a reference cycle. Intermediates use lightweight nodes.
    return tensor if node is None else node


def run(roots, seeds, *, retain_graph=False, targets=None, accumulate=False):
    order, visited, stack = [], set(), [(node, False) for node in roots]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
        elif id(node) not in visited:
            visited.add(id(node))
            stack.append((node, True))
            stack.extend((parent, False) for parent, _, _ in node._parents)
    needed = None
    if targets is not None:
        needed = set(targets)
        for node in order:
            if any(id(parent) in needed for parent, _, _ in node._parents):
                needed.add(id(node))
    if len(order) == len(roots) == 1 and not order[0]._parents:
        return _execute(order, roots, seeds, needed, retain_graph, targets, accumulate, None)
    with NumericalScope() as scope:
        return _execute(order, roots, seeds, needed, retain_graph, targets, accumulate, scope)


def _execute(order, roots, seeds, needed, retain_graph, targets, accumulate, scope):
    pending = {}
    if len(roots) == 1:
        pending[id(roots[0])] = seeds[0]
    else:
        for root, seed in zip(roots, seeds):
            key = id(root)
            pending[key] = seed if key not in pending else pending[key] + seed
    derivatives = {}
    for node in reversed(order):
        key = id(node)
        gradient = pending.pop(key, None)
        if gradient is None or needed is not None and key not in needed:
            if needed is not None and key not in needed:
                continue
            if not accumulate and targets is not None and key in targets:
                continue
            # An unused edge still consumes the saved ancestor graph. This
            # matters for shared outputs such as real slogdet's sign/log pair.
            if isinstance(node,Node) and node.saved:
                if node.released:
                    raise RuntimeError('Trying to backward through the graph a second time: saved tensors were freed; use retain_graph=True')
                if not retain_graph:node.release()
            continue
        parents = node._parents
        leaf = not parents
        if leaf:
            owner = node
            hook_gradient = None
            update = accumulate and owner.requires_grad and (targets is None or key in targets)
            if node._backward_hooks or update and node._post_accumulate_grad_hooks:
                apply = _tensor_hooks.apply if scope is None else scope.callback
                gradient, hook_gradient = apply(gradient, node._backward_hooks, owner if update else None)
            elif update:
                owner._accumulate_gradient(gradient)
        elif node.callbacks is not None:
            callbacks = node.callbacks
            owner = callbacks()
            retained = owner is not None and owner.requires_grad and owner.retains_grad
            hook_gradient = None
            if callbacks.hooks:
                apply = _tensor_hooks.apply if scope is None else scope.callback
                gradient, hook_gradient = apply(gradient, callbacks.hooks, owner if retained else None)
            elif retained:
                owner._accumulate_gradient(gradient)
        else:
            owner = None
        if not accumulate and (leaf or targets is not None and key in targets or owner is not None and owner.retains_grad):
            derivatives[key] = hook_gradient if (leaf or node.callbacks is not None) and hook_gradient is not None else gradient
        execute_target = accumulate and targets is not None and key in targets and not leaf
        edges = parents if needed is None or execute_target else tuple(edge for edge in parents if id(edge[0]) in needed)
        if not edges:
            continue
        if node.released:
            raise RuntimeError('Trying to backward through the graph a second time: saved tensors were freed; use retain_graph=True')
        if any(counter.version != version for counter, version in node.versions):
            raise _errors.BrowserUnsupported('A tensor saved for backward was modified in-place; this graph requires the server')
        if node.joint is None:
            for parent, vjp, dtype in edges:
                value = vjp(gradient)
                # None is an unused edge; a zero array is still a real gradient.
                if value is None:
                    continue
                incoming = np.asarray(value, dtype=dtype)
                if needed is not None and id(parent) not in needed:
                    continue
                parent_key = id(parent)
                pending[parent_key] = incoming if parent_key not in pending else pending[parent_key] + incoming
        else:
            backward, active = node.joint
            # The public rule receives construction-time active inputs. The
            # engine can discard unrelated branches after the joint evaluation.
            values = backward(gradient, active)
            if not isinstance(values, (tuple, list)) or len(values) != len(active):
                raise RuntimeError('A joint VJP must return one gradient per input')
            if any(flag and value is None for flag, value in zip(active, values)):
                raise RuntimeError('A joint VJP omitted a required input gradient')
            values = (value for flag, value in zip(active, values) if flag)
            for (parent, _, dtype), value in zip(parents, values):
                if needed is not None and id(parent) not in needed:
                    continue
                incoming = np.asarray(value, dtype=dtype)
                if incoming.shape != parent.shape:
                    raise RuntimeError('A joint VJP gradient shape must match its input')
                parent_key = id(parent)
                pending[parent_key] = incoming if parent_key not in pending else pending[parent_key] + incoming
        if not retain_graph and node.saved:
            node.release()
    return order, derivatives
