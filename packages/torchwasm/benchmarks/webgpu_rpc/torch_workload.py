"""Matched finite-float32 matrix forward/VJP workloads for later GPU measurement.

TorchMatrixStep uses actual autograd and returns independent CPU snapshots.
GpuMatrixStep uses explicit analytical VJPs through the experimental matrix
service. It is not a Torch GPU storage or autograd backend. No timing starts at
import; the browser must enter Python through a JSPI-compatible async entrypoint.
"""
from contextlib import ExitStack
import hashlib

import numpy as np

_FLOAT32 = np.dtype('float32')
_PROGRAM_MACS = 268435456
_MATRIX_BYTES = 4 * 1024 * 1024
_DIMENSION = 1024
_OPERATIONS = ('forward', 'left_vjp', 'right_vjp')
_INPUTS = dict(forward=('a', 'b'), left_vjp=('b', 'g'), right_vjp=('a', 'g'))


def specifications():
    return ((37, 53, 29), (64, 64, 64), (256, 256, 256), (512, 512, 512))


def make_inputs(shape):
    m, k, n = shape
    values = []
    for rows, cols, modulus, center in ((m, k, 17, 8), (k, n, 13, 6), (m, n, 11, 5)):
        data = ((np.arange(rows * cols) % modulus - center) / 8).astype(_FLOAT32).reshape(rows, cols)
        data.flags.writeable = False
        values.append(data)
    return tuple(values)


def fingerprint(values):
    return tuple((str(value.dtype), tuple(value.shape), value.strides,
                  hashlib.sha256(value.tobytes()).hexdigest(), bool(value.flags.writeable))
                 for value in values)


def _validate(values):
    values = tuple(values)
    if len(values) != 3 or any(not isinstance(value, np.ndarray) or value.ndim != 2 or
                               value.dtype != _FLOAT32 or not value.flags.c_contiguous for value in values):
        raise TypeError('Three C-contiguous float32 matrices are required')
    a, b, g = values
    m, k = a.shape
    if b.shape[0] != k or g.shape != (m, b.shape[1]):
        raise ValueError('Matrices must have shapes (m,k), (k,n), (m,n)')
    n = b.shape[1]
    if min(m, k, n) < 1 or max(m, k, n) > _DIMENSION or any(value.nbytes > _MATRIX_BYTES for value in values):
        raise ValueError('Workload exceeds the published matrix bounds')
    if m * k * n > _PROGRAM_MACS or not all(np.isfinite(value).all() for value in values):
        raise ValueError('One product exceeds the published work bound, or input is nonfinite')
    return values


def program_groups(shape):
    cost = int(shape[0]) * int(shape[1]) * int(shape[2])
    if cost < 1 or cost > _PROGRAM_MACS:
        raise ValueError('A single product exceeds the published program work limit')
    groups = []
    for start in range(0, 3, _PROGRAM_MACS // cost):
        groups.append(_OPERATIONS[start:start + _PROGRAM_MACS // cost])
    return tuple(groups)


def _needed(group):
    return tuple(name for name in ('a', 'b', 'g') if any(name in _INPUTS[operation] for operation in group))


def description(values, resident):
    a, b, g = values
    shape = (a.shape[0], a.shape[1], b.shape[1])
    groups = program_groups(shape)
    arrays = dict(zip(('a', 'b', 'g'), values))
    return dict(shape=shape, logical_macs=3 * int(np.prod(shape)), program_groups=groups,
                setup_upload_bytes=sum(value.nbytes for value in values) if resident else 0,
                per_call_upload_bytes=0 if resident else sum(arrays[name].nbytes for group in groups for name in _needed(group)),
                per_call_download_bytes=4 * (shape[0] * shape[2] + shape[0] * shape[1] + shape[1] * shape[2]),
                all_outputs_are_cpu_snapshots=True, gpu_vjps_are_explicit=True,
                limits_changed=False)


class TorchMatrixStep:
    """One actual Torch-compatible forward and autograd.grad invocation."""
    def __init__(self, api, values, *, resident):
        self.api = api
        self.values = _validate(values)
        self.resident = bool(resident)
        self.closed = False
        self.inputs = self._tensors() if resident else None

    def _tensors(self):
        tensors = tuple(self.api.from_numpy(value).requires_grad_(index < 2) for index, value in enumerate(self.values))
        for tensor, value in zip(tensors, self.values):
            if tensor.data_ptr() != value.__array_interface__['data'][0]:
                raise RuntimeError('The CPU baseline copied an input')
        return tensors

    def __call__(self):
        if self.closed:
            raise RuntimeError('Workload closed')
        a, b, g = self.inputs if self.resident else self._tensors()
        output = a @ b
        left, right = self.api.autograd.grad(output, (a, b), g)
        if a.grad is not None or b.grad is not None:
            raise RuntimeError('autograd.grad accumulated a caller gradient')
        return tuple(value.detach().numpy().copy() for value in (output, left, right))

    def close(self):
        self.inputs = None
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Workload closed')
        return self

    def __exit__(self, *error):
        self.close()


class GpuMatrixStep:
    """Explicit GPU matmul/VJP programs, including readback on every call.

    Cold mode submits raw arrays and counts repeated uploads when a quota forces
    partitioning. Resident mode uploads three inputs at construction, then reuses
    those same handles. Setup and final release are outside steady-state calls.
    """
    def __init__(self, matrix_api, values, *, resident, client=None):
        self.matrix_api = matrix_api
        self.values = _validate(values)
        self.resident = bool(resident)
        self.client = client
        self.closed = False
        self.scope = ExitStack()
        self.metadata = description(self.values, resident)
        try:
            inputs = tuple(self.scope.enter_context(matrix_api.Matrix.upload(value, client=client))
                           for value in self.values) if resident else self.values
            self.inputs = dict(zip(('a', 'b', 'g'), inputs))
        except BaseException:
            self.scope.close()
            raise

    def __call__(self):
        if self.closed:
            raise RuntimeError('Workload closed')
        arrays = {}
        for group in self.metadata['program_groups']:
            program = self.matrix_api.MatrixProgram(client=self.client)
            refs = {name: program.input(self.inputs[name]) for name in _needed(group)}
            outputs = []
            for operation in group:
                if operation == 'forward':
                    output = program.matmul(refs['a'], refs['b'])
                elif operation == 'left_vjp':
                    output = program.matmul(refs['g'], refs['b'], transpose_b=True)
                else:
                    output = program.matmul(refs['a'], refs['g'], transpose_a=True)
                outputs.append(output)
            with program.run(outputs, read=True, retain=False) as result:
                arrays.update(zip(group, result.arrays))
        return tuple(arrays[name] for name in _OPERATIONS)

    def close(self):
        if not self.closed:
            try:
                self.scope.close()
            finally:
                self.closed = True
                self.inputs = None

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Workload closed')
        return self

    def __exit__(self, *error):
        self.close()
