"""Bounded content-coherent GPU buffer residency for explicit Tensor offload.

A context owns its immutable buffers. CPU Tensor values remain public, while
resident buffers can bridge forward nodes and directly adjacent GPU pullbacks.
No computed numerical result is memoized: every eligible call runs its program.
"""
from collections import OrderedDict
import operator
import weakref
import numpy as np
import torch_bridge as core


class _RpcBackend:
    def program(self, jobs, resolved, retain_outputs, retain_inputs):
        from matrix_client import MatrixProgram
        from types import SimpleNamespace
        program = MatrixProgram()
        refs, seen = {}, {}
        for a, b, _, _ in jobs:
            for value in (a, b):
                actual = resolved[id(value)]
                if id(actual) not in seen:
                    seen[id(actual)] = program.input(actual)
                refs[id(value)] = seen[id(actual)]
        outputs = [program.matmul(refs[id(a)], refs[id(b)], ta, tb) for a, b, ta, tb in jobs]
        retained = [refs[key] for key in retain_inputs]
        result = program.run(outputs, read=True, retain_inputs=retained, retain_outputs=retain_outputs)
        return SimpleNamespace(arrays=result.arrays, matrices=result.matrices,
                               inputs={key:result.input_matrices[refs[key]] for key in retain_inputs})

    def release(self, matrices):
        from pyodide.ffi import run_sync, to_js
        live = [matrix for matrix in matrices if not matrix.closed]
        if not live:
            return
        client = live[0].client
        if any(matrix.client != client for matrix in live):
            raise RuntimeError('Resident buffers belong to different sessions')
        if not client.closed:
            run_sync(client.releaseMany(to_js([matrix.handle for matrix in live])))
        for matrix in live:
            matrix.closed = True


class _Entry:
    def __init__(self, serial, data, matrix, used):
        self.serial, self.data, self.matrix, self.used = serial, data, matrix, used
        self.signature = _signature(data)


def _signature(value):
    bits = value.view(np.uint32)
    return (value.shape, value.dtype.str, int(bits[0,0]), int(bits[-1,-1]),
            int(bits[value.shape[0]//2,value.shape[1]//2]))


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint32), b.view(np.uint32))


class ResidentMatmulOffload(core.MatmulOffload):
    """Explicit bounded session; no GPU device or stable package API is added.

    close() frees resident buffers and rejects new forward calls. Existing
    graphs can finish through CPU pullbacks after an intentional close. Failed
    GPU calls propagate errors; they are never retried as CPU work.
    """
    def __init__(self, *, max_entries=8, max_bytes=8*1024*1024, backend=None):
        for name, value, maximum in [('max_entries', max_entries, 8), ('max_bytes', max_bytes, 8*1024*1024)]:
            if isinstance(value, bool):
                raise TypeError(name+' must be an integer')
            number = operator.index(value)
            if not 0 <= number <= maximum:
                raise ValueError(name+' exceeds the bounded cache limit')
        self.max_entries, self.max_bytes = operator.index(max_entries), operator.index(max_bytes)
        self.backend = _RpcBackend() if backend is None else backend
        self.closed = False
        self.failed = False
        self._entries, self._leases = [], {}
        self._operations = OrderedDict()
        self._serial = self._clock = 0
        self._counters = dict(uploads=0, upload_bytes=0, cache_hits=0, programs=0,
                              inline_upload_bytes=0, retained_results=0, evictions=0)
        super().__init__()

    def __enter__(self):
        self._live()
        if core._busy:
            raise RuntimeError('A GPU Tensor submission is already active')
        return self

    def __exit__(self, typ, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is not None:
                raise RuntimeError('Resident cleanup failed after an operation error: '+str(cleanup)) from error
            raise
        return False

    def _live(self):
        if self.closed:
            raise RuntimeError('Resident GPU session is closed')
        if self.failed:
            raise RuntimeError('Resident GPU cleanup failed; close the session before reuse')

    def __call__(self, a, b):
        self._live()
        return super().__call__(a, b)

    def _touch(self, entry):
        self._clock += 1
        entry.used = self._clock

    def _find(self, value):
        signature = _signature(value)
        for entry in self._entries:
            if entry.signature == signature and _bits_equal(entry.data, value):
                self._touch(entry)
                return entry
        return None

    def _release(self, matrices):
        live = list({id(matrix): matrix for matrix in matrices if id(matrix) in self._leases}.values())
        if not live:
            return
        self.backend.release(live)
        removed = {id(matrix) for matrix in live}
        for key in removed:
            self._leases.pop(key, None)
        self._entries[:] = [entry for entry in self._entries if id(entry.matrix) not in removed]

    def _add(self, value, matrix):
        self._serial += 1
        self._clock += 1
        entry = _Entry(self._serial, value, matrix, self._clock)
        self._entries.append(entry)
        return entry

    def _known_output(self, key):
        reference = self._operations.get(key) if key is not None else None
        entry = reference() if reference is not None else None
        return entry is not None and entry in self._entries

    def _submit_jobs(self, jobs, retain_outputs):
        if self.closed:
            return [core._cpu_product(*job) for job in jobs]
        self._live()
        unique = {id(value): value for a, b, _, _ in jobs for value in (a, b)}
        resolved, entries, pinned = {}, {}, set()
        for key, value in unique.items():
            entry = self._find(value)
            if entry is not None:
                resolved[key], entries[key] = entry.matrix, entry
                pinned.add(id(entry.matrix))
                self._counters['cache_hits'] += 1
        pool = list(self._entries)
        victims, reserved_count, reserved_bytes = [], 0, 0
        def reserve(size):
            nonlocal pool, reserved_count, reserved_bytes
            if not self.max_entries or size > self.max_bytes:
                return False
            trial = list(pool)
            removed = []
            while len(trial)+reserved_count >= self.max_entries or sum(entry.data.nbytes for entry in trial)+reserved_bytes+size > self.max_bytes:
                available = [entry for entry in trial if id(entry.matrix) not in pinned]
                if not available:
                    return False
                victim = min(available, key=lambda entry:entry.used)
                trial.remove(victim);removed.append(victim)
            pool = trial;victims.extend(removed)
            reserved_count += 1;reserved_bytes += size
            return True
        pending, canonical, retained_inputs = {}, {}, []
        for key, value in unique.items():
            if key in resolved:
                continue
            previous = next((old for old,other in pending.items() if _signature(other)==_signature(value) and _bits_equal(other,value)),None)
            if previous is not None:
                resolved[key] = pending[previous];canonical[key]=previous
                continue
            pending[key]=value;resolved[key]=value;canonical[key]=key
            if reserve(value.nbytes):retained_inputs.append(key)
        wanted=[]
        for (a,b,ta,tb),retain in zip(jobs,retain_outputs):
            key=(entries[id(a)].serial,entries[id(b)].serial,ta,tb) if id(a) in entries and id(b) in entries else None
            shape=(a.shape[1] if ta else a.shape[0],b.shape[0] if tb else b.shape[1])
            wanted.append(bool(retain and not self._known_output(key) and reserve(4*shape[0]*shape[1])))
        # Evictions, uploads and retention are planned before execution. One
        # release request removes every victim; the program owns all new bytes.
        self._release([entry.matrix for entry in victims])
        self._counters['evictions'] += len(victims)
        self._counters['programs'] += 1
        self._counters['uploads'] += len(retained_inputs)
        self._counters['upload_bytes'] += sum(pending[key].nbytes for key in retained_inputs)
        self._counters['inline_upload_bytes'] += sum(value.nbytes for key,value in pending.items() if key not in retained_inputs)
        result = self.backend.program(jobs, resolved, tuple(wanted), tuple(retained_inputs))
        matrices = list(result.matrices)
        incoming = dict(result.inputs)
        for matrix in [*matrices,*incoming.values()]:
            if matrix is not None:self._leases[id(matrix)]=matrix
        arrays=list(result.arrays)
        if len(arrays)!=len(jobs) or len(matrices)!=len(jobs) or set(incoming)!=set(retained_inputs):
            raise RuntimeError('Resident transport returned an invalid output count')
        for key,matrix in incoming.items():
            if matrix is None:raise RuntimeError('Resident transport omitted an input lease')
            entries[key]=self._add(pending[key],matrix)
        for key,source in canonical.items():
            if source in entries:entries[key]=entries[source]
        discard=[]
        for array,matrix,job,keep in zip(arrays,matrices,jobs,wanted):
            a,b,ta,tb=job;shape=(a.shape[1] if ta else a.shape[0],b.shape[0] if tb else b.shape[1])
            if not isinstance(array,np.ndarray) or array.dtype!=np.float32 or array.shape!=shape:
                raise RuntimeError('Resident transport returned an invalid matrix')
            if matrix is None:
                if keep:raise RuntimeError('Resident transport omitted an output lease')
                continue
            existing=self._find(array)
            if keep and existing is None:
                existing=self._add(array,matrix);self._counters['retained_results']+=1
            else:discard.append(matrix)
            key=(entries[id(a)].serial,entries[id(b)].serial,ta,tb) if id(a) in entries and id(b) in entries else None
            if keep and existing is not None and key is not None:
                self._operations[key]=weakref.ref(existing);self._operations.move_to_end(key)
                while len(self._operations)>64:self._operations.popitem(last=False)
        self._release(discard)
        return arrays

    def _execute(self, *args, **kwargs):
        if core._busy:
            raise RuntimeError('A GPU Tensor submission is already active')
        if self.closed:
            return super()._execute(*args, **kwargs)
        self._live()
        before = dict(self._leases)
        try:
            return super()._execute(*args, **kwargs)
        except BaseException as error:
            # The base has reset the global gate after validating original
            # Tensor/seed state. Rollback itself can suspend too.
            core._busy = True
            try:
                self._release([matrix for key, matrix in self._leases.items() if key not in before])
            except BaseException as cleanup:
                self.failed = True
                raise RuntimeError('Resident rollback failed: '+str(cleanup)) from error
            finally:
                core._busy = False
            raise

    def cache_info(self):
        if core._busy:
            raise RuntimeError('A GPU Tensor submission is already active')
        return dict(self._counters, entries=len(self._entries), resident_bytes=sum(entry.data.nbytes for entry in self._entries),
                    owned_leases=len(self._leases), closed=self.closed, failed=self.failed,
                    max_entries=self.max_entries, max_bytes=self.max_bytes)

    def clear(self):
        self._live()
        if core._busy:
            raise RuntimeError('A GPU Tensor submission is already active')
        core._busy = True
        try:
            self._release(list(self._leases.values()))
            self._operations.clear()
        except BaseException:
            self.failed = True
            raise
        finally:
            core._busy = False

    def close(self):
        if self.closed and not self._leases:
            return
        if core._busy:
            raise RuntimeError('A GPU Tensor submission is already active')
        core._busy = True
        self.closed = True
        try:
            self._release(list(self._leases.values()))
            self._operations.clear()
            self.failed = False
        except BaseException:
            self.failed = True
            raise
        finally:
            core._busy = False
