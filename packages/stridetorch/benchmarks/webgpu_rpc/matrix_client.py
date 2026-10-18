"""Experimental explicit resident matrices, separate from torch device policy."""
import numpy as np
from pyodide.ffi import run_sync
from pyodide.ffi import to_js
from js import Object
from js import matrix_rpc


class Matrix:
    def __init__(self, handle, shape, client=None):
        self.handle = int(handle)
        self.shape = tuple(shape)
        self.client = matrix_rpc if client is None else client
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
        client = matrix_rpc if client is None else client
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
    def __init__(self, matrices, arrays, client):
        self.matrices = tuple(matrices)
        self.arrays = tuple(arrays)
        self.client = client

    def close(self):
        live = [value for value in self.matrices if value is not None and not value.closed]
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
        self.client = matrix_rpc if client is None else client
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

    def run(self, outputs, *, read=True, retain=False):
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
        descriptors = [dict(ref=ref, read=read, retain=retain) for ref in outputs]
        convert = lambda value: to_js(value, dict_converter=Object.fromEntries)
        result = run_sync(self.client.program(convert(inputs), convert(self.nodes), convert(descriptors)))
        matrices, arrays = [], []
        try:
            for output in result.outputs:
                matrices.append(Matrix(output.handle, (output.rows, output.cols), self.client) if retain else None)
                arrays.append(np.array(output.data.to_py(), dtype=np.float32, copy=True).reshape((output.rows, output.cols)) if read else None)
        except BaseException:
            # Host handles already exist if conversion fails after the reply.
            handles = [output.handle for output in result.outputs if retain]
            if handles and not self.client.closed:
                run_sync(self.client.releaseMany(to_js(handles)))
            raise
        return ProgramResult(matrices, arrays, self.client)
