"""Run a CPU benchmark with BLAS controls set before imports and recorded.

Other arguments pass through to the chosen bench_*.py runner. Apple Accelerate's
per-thread threading model is selected/query-checked when its API is available;
environment settings alone are reported as configuration, not measured threads.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys


def source_hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob('*.py'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['runtime', 'tensor_hooks', 'tensor_factories', 'arithmetic', 'selection', 'module_hooks', 'nn', 'schedulers'], default='runtime')
    parser.add_argument('--threads', type=int, default=1, help='Requested maximum for backend environment controls')
    parser.add_argument('--output')
    options, remaining = parser.parse_known_args()
    if options.threads < 1: parser.error('threads must be positive')
    names = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']
    for name in names: os.environ[name] = str(options.threads)
    import numpy as np
    build = np.__config__.show(mode='dicts')
    blas = build.get('Build Dependencies', {}).get('blas', {})
    controls = dict(environment={name: os.environ[name] for name in names}, numpy_blas=blas)
    accelerate = None
    if sys.platform == 'darwin' and blas.get('name') == 'accelerate':
        import ctypes
        try:
            accelerate = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
            accelerate.BLASSetThreading.argtypes = [ctypes.c_uint]
            accelerate.BLASSetThreading.restype = ctypes.c_int
            accelerate.BLASGetThreading.argtypes = []
            accelerate.BLASGetThreading.restype = ctypes.c_uint
        except (OSError, AttributeError) as error:
            accelerate = None
            controls['accelerate'] = dict(verified=False, reason=str(error))
        if accelerate is not None:
            # Public thread_api.h: enum 0=framework-selected, 1=single-threaded.
            requested = 1 if options.threads == 1 else 0
            before = accelerate.BLASGetThreading()
            status = accelerate.BLASSetThreading(requested)
            selected = accelerate.BLASGetThreading()
            if status != 0 or selected != requested:
                raise RuntimeError('Accelerate rejected the requested BLAS threading model')
            controls['accelerate'] = dict(before=before, set_status=status, selected=selected, verified=True)
    package = Path(importlib.util.find_spec('paperleet_torch').origin).parent
    before = source_hashes(package)
    path = Path(__file__).with_name('bench_' + options.benchmark + '.py')
    output = io.StringIO(); previous = sys.argv
    try:
        sys.argv = [str(path), *remaining]
        with contextlib.redirect_stdout(output): runpy.run_path(str(path), run_name='__main__')
    finally:
        sys.argv = previous
    report = json.loads(output.getvalue())
    actual = Path(report['implementation']).parent
    report['source_hashes'] = source_hashes(actual)
    if actual == package and before != report['source_hashes']:
        raise RuntimeError('Benchmark sources changed during execution')
    if accelerate is not None:
        after = accelerate.BLASGetThreading()
        controls['accelerate']['after'] = after
        if after != controls['accelerate']['selected']:
            raise RuntimeError('Accelerate BLAS threading policy changed during execution')
    report['cpu_controls'] = controls
    raw = json.dumps(report, indent=2) + '\n'
    if options.output: Path(options.output).write_text(raw, encoding='utf8')
    print(raw, end='')


if __name__ == '__main__': main()
