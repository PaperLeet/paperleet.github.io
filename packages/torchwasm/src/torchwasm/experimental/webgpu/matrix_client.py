"""Experimental explicit resident matrices, separate from torch device policy."""
import numpy as np
from ._pyodide import run_sync, to_js, dict_to_js, default_client


class Matrix:
    def __init__(self, handle, shape, client=None):
        self.handle = int(handle)
        self.shape = tuple(shape)
        self.client = default_client() if client is None else client
        self.closed = False

    def _live(self):
        if self.closed:
            raise RuntimeError('Matrix released')

    @classmethod
    def upload(cls, value, client=None):
        value = np.asarray(value)
        if value.ndim != 2 or value.dtype != np.float32:
            raise TypeError('A 2D float32 array is required')
        value = np.ascontiguousarray(value)
        client = default_client() if client is None else client
        result = run_sync(client.upload(value.reshape(-1), *value.shape))
        return cls(result.handle, value.shape, client)

    def matmul(self, other, transpose_a=False, transpose_b=False):
        self._live()
        other._live()
        if self.client != other.client:
            raise ValueError('Matrices belong to different sessions')
        result = run_sync(self.client.matmul(self.handle, other.handle, transpose_a, transpose_b))
        return Matrix(result.handle, (result.rows, result.cols), self.client)

    def numpy(self):
        self._live()
        # to_py explicitly copies JS data into a Python-owned ndarray.
        result = run_sync(self.client.read(self.handle))
        return np.array(result.to_py(), dtype=np.float32, copy=True).reshape(self.shape)

    def close(self):
        if not self.closed:
            if not self.client.closed:
                run_sync(self.client.release(self.handle))
            self.closed = True

    def __enter__(self):
        self._live()
        return self

    def __exit__(self, *args):
        self.close()


class ProgramResult:
    def __init__(self, matrices, arrays, client, input_matrices=None):
        self.matrices = tuple(matrices)
        self.arrays = tuple(arrays)
        self.client = client
        self.input_matrices = {} if input_matrices is None else input_matrices

    def close(self):
        live = [value for value in (*self.matrices, *self.input_matrices.values()) if value is not None and not value.closed]
        if live and not self.client.closed:
            run_sync(self.client.releaseMany(to_js([value.handle for value in live])))
        for value in live:
            value.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MatrixProgram:
    """A bounded matmul DAG. Input arrays are copied when run() submits them."""
    def __init__(self, client=None):
        self.client = default_client() if client is None else client
        self.inputs = []
        self.nodes = []

    def input(self, value):
        if self.nodes:
            raise ValueError('Add inputs before program nodes')
        self.inputs.append(value)
        return len(self.inputs) - 1

    def matmul(self, a, b, transpose_a=False, transpose_b=False):
        self.nodes.append(dict(a=a, b=b, transposeA=transpose_a, transposeB=transpose_b))
        return len(self.inputs) + len(self.nodes) - 1

    def run(self, outputs, *, read=True, retain=False, retain_inputs=(), retain_outputs=None):
        outputs = tuple(outputs)
        retained = (retain,)*len(outputs) if retain_outputs is None else tuple(retain_outputs)
        if len(retained) != len(outputs):
            raise ValueError('Expected one retention flag per computed output')
        retained_inputs = tuple(retain_inputs)
        if any(type(ref) is not int or not 0 <= ref < len(self.inputs) or isinstance(self.inputs[ref], Matrix) for ref in retained_inputs):
            raise ValueError('Only newly uploaded inputs can be retained')
        if len(set(retained_inputs)) != len(retained_inputs):
            raise ValueError('Retained inputs must be unique')
        if retained_inputs:
            check = getattr(self.client, 'requireFeatures', None)
            if not callable(check):
                raise RuntimeError('Retained inputs require capability negotiation; reload the client and GPU service together')
            check('program-matmul-f32', 'program-retain-inputs')
        inputs = []
        for value in self.inputs:
            if isinstance(value, Matrix):
                value._live()
                if value.client != self.client:
                    raise ValueError('Matrix belongs to another session')
                inputs.append(dict(handle=value.handle))
            else:
                value = np.asarray(value)
                if value.ndim != 2 or value.dtype != np.float32:
                    raise TypeError('A 2D float32 array is required')
                inputs.append(dict(rows=value.shape[0], cols=value.shape[1], data=np.ascontiguousarray(value).reshape(-1)))
        descriptors = [dict(ref=ref, read=read, retain=flag) for ref,flag in zip(outputs,retained)]
        descriptors.extend(dict(ref=ref,read=False,retain=True) for ref in retained_inputs)
        convert = dict_to_js
        result = run_sync(self.client.program(convert(inputs), convert(self.nodes), convert(descriptors)))
        matrices, arrays, input_matrices = [], [], {}
        try:
            for descriptor, output in zip(descriptors, result.outputs):
                matrix = Matrix(output.handle, (output.rows, output.cols), self.client) if descriptor['retain'] else None
                array = np.array(output.data.to_py(), dtype=np.float32, copy=True).reshape((output.rows, output.cols)) if descriptor['read'] else None
                if descriptor['ref'] in retained_inputs:
                    input_matrices[descriptor['ref']] = matrix
                else:
                    matrices.append(matrix);arrays.append(array)
        except BaseException:
            handles = [output.handle for descriptor,output in zip(descriptors,result.outputs) if descriptor['retain']]
            if handles and not self.client.closed:
                run_sync(self.client.releaseMany(to_js(handles)))
            raise
        return ProgramResult(matrices, arrays, self.client, input_matrices)
