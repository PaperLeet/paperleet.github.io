"""CPython tracemalloc allocations, not RSS or browser heap, for delayed Adam."""
import argparse
import gc
import json
import platform
import sys
import tracemalloc

import numpy as np
from paperleet_torch import __version__, torch


def measure(dtype='float32', count=16, width=256):
    params = [torch.nn.Parameter(torch.ones(width, width, dtype=np.dtype(dtype))) for _ in range(count)]
    gc.collect()
    tracemalloc.start()
    optimizer = torch.optim.Adam(params)
    constructor, constructor_peak = tracemalloc.get_traced_memory()
    def moment_bytes():
        state = optimizer.__dict__.get('state')
        if state is None:
            return sum(value.nbytes for name in ('m', 'v') for value in optimizer.__dict__[name])
        return sum(value._data.nbytes for item in state.values() for value in item.values())
    constructor_moments = moment_bytes()
    params[-1].grad = torch.ones(width, width, dtype=np.dtype(dtype))
    optimizer.step()
    one_gradient, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    state = optimizer.__dict__.get('state')
    return dict(library_version=__version__, python=platform.python_version(), numpy=np.__version__,
                implementation=sys.modules[torch.Tensor.__module__].__file__, dtype=dtype,
                parameters=count, width=width, constructor_retained_bytes=constructor,
                constructor_peak_bytes=constructor_peak, one_gradient_retained_bytes=one_gradient,
                constructor_moment_bytes=constructor_moments, one_gradient_moment_bytes=moment_bytes(),
                peak_bytes=peak, state_entries=None if state is None else len(state),
                note='NumPy allocations registered with tracemalloc; excludes pre-created parameters; not RSS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = json.dumps(measure(args.dtype), indent=2) + '\n'
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as stream:stream.write(report)
    print(report)
