"""Experimental explicit resident matrices, separate from torch device policy."""
import numpy as np
from pyodide.ffi import run_sync
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
