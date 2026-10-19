"""Explicit experimental GPU matmul offload with ordinary CPU Tensor results.

GPU kernels and first-order pullbacks use the existing bounded MatrixProgram.
There is no resident Torch GPU storage, new device, or automatic monkeypatch.
The injected submitter is also a deterministic transport test seam.
"""
import numpy as np
from stridetorch import torch, _autograd, _matmul, _state, _tensor

_FLOAT32 = np.dtype('float32')
_MACS = 268435456
_busy = False
_MODE_NAMES = ('_grad_enabled', '_jvp_active', '_functional_ad_active',
               '_functional_jvp_active', '_hvp_active', '_full_forward_graph_active',
               '_tensor_reverse_context')


def _mode():
    return tuple(getattr(_state, name) for name in _MODE_NAMES)


def _ordinary_mode():
    return _state._grad_enabled and not any(_mode()[1:])


def _supported_pair(a, b):
    if (a.ndim != 2 or b.ndim != 2 or a.dtype != _FLOAT32 or b.dtype != _FLOAT32
            or a.shape[1] != b.shape[0]):
        return False
    m, k = a.shape
    n = b.shape[1]
    if min(m, k, n) < 1 or max(m, k, n) > 1024 or m*k*n > _MACS:
        return False
    limit = np.finfo(np.float32)
    lower, upper = np.sqrt(limit.tiny), np.sqrt(limit.max/k)*.25
    for value in (a, b):
        if value.nbytes > 4*1024*1024:
            return False
        magnitude = np.abs(value)
        if not magnitude.max() <= upper or np.min(magnitude, where=magnitude != 0, initial=upper) < lower:
            return False
    return True


class _Snapshot:
    def __init__(self, value):
        self.tensor = value
        self.data = value._data
        self.storage, self.version = value._storage, value._storage.version
        self.shape, self.dtype = value.shape, value.dtype
        self.activity = value.requires_grad
        self.node, self.no_grad_view = value._grad_node, value._no_grad_view
        self.owned = np.array(value._data, dtype=np.float32, order='C', copy=True)

    def check(self, *, metadata=False, content=False):
        value = self.tensor
        changed = (value._storage is not self.storage or self.storage.version != self.version
                   or value._data is not self.data or value.shape != self.shape or value.dtype != self.dtype)
        if metadata:
            changed = changed or (value.requires_grad != self.activity or value._grad_node is not self.node
                                  or value._no_grad_view != self.no_grad_view)
        if content and not changed:
            changed = not np.array_equal(value._data.view(np.uint32), self.owned.view(np.uint32))
        if changed:
            raise RuntimeError('A Tensor input changed during or after GPU submission')


def _cpu_product(a, b, transpose_a, transpose_b):
    a = a.T if transpose_a else a
    b = b.T if transpose_b else b
    return _matmul.product(a, b) if _tensor._WASM_MATMUL else a @ b


class MatmulOffload:
    """Opt-in host GPU forward and actual first-order autograd for finite f32 2D.

    Inference under no_grad, AD transforms and unsupported inputs use the
    existing CPU operation without suspending. GPU failures are propagated;
    they are never silently retried as CPU work. Results remain CPU Tensors.
    """
    def __init__(self, submit=None):
        self.submit = self._rpc_submit if submit is None else submit

    @staticmethod
    def _rpc_submit(jobs):
        from matrix_client import MatrixProgram
        program = MatrixProgram()
        references = {}
        for a, b, _, _ in jobs:
            for value in (a, b):
                if id(value) not in references:
                    references[id(value)] = program.input(value)
        outputs = [program.matmul(references[id(a)], references[id(b)], ta, tb)
                   for a, b, ta, tb in jobs]
        with program.run(outputs) as result:
            return list(result.arrays)

    def _submit_jobs(self, jobs, retain_outputs):
        return self.submit(jobs)

    def _execute(self, jobs, checks, *, metadata=False, seed=None, seed_owned=None,
                 retain_outputs=None):
        global _busy
        if _busy:
            raise RuntimeError('A GPU Tensor submission is already active in this Python runtime')
        mode = _mode()
        seed_storage = _tensor.storage_version(seed) if seed is not None else None
        seed_version = seed_storage.version if seed_storage is not None else None
        groups, current, cost = [], [], 0
        for job in jobs:
            a, b, ta, tb = job
            work = (a.shape[1] if ta else a.shape[0]) * (a.shape[0] if ta else a.shape[1]) * (b.shape[0] if tb else b.shape[1])
            if work > _MACS:
                raise RuntimeError('GPU product exceeds the published work limit')
            if current and cost + work > _MACS:
                groups.append(current);current=[];cost=0
            current.append(job);cost+=work
        if current:groups.append(current)
        values=[]
        _busy = True
        try:
            offset = 0
            for group in groups:
                retain = tuple(retain_outputs[offset:offset+len(group)]) if retain_outputs is not None else (False,)*len(group)
                offset += len(group)
                result = self._submit_jobs(tuple(group), retain)
                if not isinstance(result, (tuple, list)) or len(result) != len(group):
                    raise RuntimeError('GPU transport returned an invalid output count')
                for value, (a, b, ta, tb) in zip(result, group):
                    shape = (a.shape[1] if ta else a.shape[0], b.shape[0] if tb else b.shape[1])
                    if not isinstance(value, np.ndarray) or value.dtype != _FLOAT32 or value.shape != shape:
                        raise RuntimeError('GPU transport returned an invalid matrix')
                    # Transport buffers cannot become public Tensor storage.
                    values.append(np.array(value, order='C', copy=True))
                for check in checks:check.check(metadata=metadata, content=True)
                if seed_storage is not None and (seed_storage.version != seed_version
                        or not np.array_equal(seed.view(np.uint32), seed_owned.view(np.uint32))):
                    raise RuntimeError('A backward seed changed during GPU submission')
                if _mode() != mode:
                    raise RuntimeError('Autograd mode changed during GPU submission')
        finally:
            _busy = False
        return values

    def __call__(self, a, b):
        if _busy:
            raise RuntimeError('A GPU Tensor submission is already active in this Python runtime')
        if (not isinstance(a, _tensor.Tensor) or not isinstance(b, _tensor.Tensor)
                or not _ordinary_mode() or not _supported_pair(a._data, b._data)):
            return a @ b
        left = _Snapshot(a)
        right = left if a is b else _Snapshot(b)
        output, = self._execute([(left.owned, right.owned, False, False)], (left, right), metadata=True, retain_outputs=(True,))
        # Keep original version/identity guards, not obsolete forward copies.
        left.owned = right.owned = None
        if not left.activity and not right.activity:
            return _tensor.Tensor(output, _metadata_owned=True)
        def backward(g, needs):
            # Preserve CPU numerical behavior for unsupported seed ranges and
            # callers explicitly evaluating backward with recording disabled.
            values = [None, None]
            originals = ((right,) if needs[0] else ()) + ((left,) if needs[1] else ())
            for original in originals:original.check()
            # Native CPU matmul reads current shared storage at backward time.
            # NumPy writes between calls bypass version counters; snapshot the
            # current required factors, then detect byte changes during waits.
            fresh = {id(original): _Snapshot(original.tensor) for original in originals}
            seed_owned = np.array(g, copy=True, order='C')
            jobs, checks, targets = [], [], []
            if needs[0]:jobs.append((seed_owned, fresh[id(right)].owned, False, True));checks.append(fresh[id(right)]);targets.append(0)
            if needs[1]:jobs.append((fresh[id(left)].owned, seed_owned, True, False));checks.append(fresh[id(left)]);targets.append(1)
            eligible = _ordinary_mode() and all(_supported_pair(x.T if ta else x, y.T if tb else y) for x, y, ta, tb in jobs)
            def upstream(index):
                node = (a if index == 0 else b)._grad_node
                joint = getattr(node, 'joint', None)
                return joint is not None and getattr(joint[0], '_offload_owner', None) is self
            keep = tuple(upstream(index) for index in targets)
            computed = self._execute(jobs, checks, seed=g, seed_owned=seed_owned, retain_outputs=keep) if eligible else [_cpu_product(*job) for job in jobs]
            for index, value in zip(targets, computed):values[index]=value
            return tuple(values)
        backward._offload_owner = self
        versioned = ((b,) if left.activity else ()) + ((a,) if right.activity else ())
        return _autograd.joint_result(output, (a, b), backward, versioned=versioned, prune_inputs=True)
