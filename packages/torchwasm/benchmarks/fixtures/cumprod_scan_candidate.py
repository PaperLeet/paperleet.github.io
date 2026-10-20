"""Development-only product scan; no TorchWasm source or aliases are mutated."""
import numpy as np

ACCUMULATORS = {np.dtype('float16'): np.dtype('float32'),
                np.dtype('float32'): np.dtype('float64'),
                np.dtype('complex64'): np.dtype('complex128')}
WORK_BYTES = 256 * 1024


def product_scan(data, axis, dtype):
    dtype = np.dtype(dtype)
    if not data.ndim:
        return np.array(data, dtype=dtype, copy=True)
    result = np.empty(data.shape, dtype=dtype)
    if not result.size:
        return result
    accumulator = ACCUMULATORS.get(dtype, dtype)
    length = data.shape[axis]
    outer = data.size // length
    block = max(1, min(length, WORK_BYTES // accumulator.itemsize // outer))
    work_shape = list(data.shape)
    work_shape[axis] = block
    workspace = np.empty(work_shape, dtype=accumulator)
    carry = np.ones(data.shape[:axis]+data.shape[axis+1:], dtype=accumulator)
    expanded = np.expand_dims(carry, axis)
    first_index = [slice(None)] * data.ndim
    first_index[axis] = slice(0, 1)
    last_index = [slice(None)] * data.ndim
    last_index[axis] = -1
    with np.errstate(all='ignore'):
        for start in range(0, length, block):
            count = min(block, length-start)
            source_index = [slice(None)] * data.ndim
            source_index[axis] = slice(start, start+count)
            work_index = [slice(None)] * data.ndim
            work_index[axis] = slice(0, count)
            work = workspace[tuple(work_index)]
            incoming = data[tuple(source_index)]
            # Requested output conversion precedes the wider accumulation.
            if incoming.dtype != dtype:
                incoming = incoming.astype(dtype)
            np.copyto(work, incoming, casting='unsafe')
            # Native's initial identity multiply is observable for complex Inf.
            np.multiply(work[tuple(first_index)], expanded,
                        out=work[tuple(first_index)])
            if count > 1:
                np.multiply.accumulate(work, axis=axis, out=work)
            np.copyto(result[tuple(source_index)], work, casting='unsafe')
            if start+count < length:
                np.copyto(carry, work[tuple(last_index)])
    return result
