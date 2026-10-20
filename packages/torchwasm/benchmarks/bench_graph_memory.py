"""CPython tracemalloc for a retained loss and a chain of saved activations.

NumPy input allocation precedes tracing. The scalar loss stays alive, while all
user-held intermediate variables are dropped. These are traced Python/NumPy
allocations, not process RSS, browser heap size or GPU memory.
"""
import argparse
import gc
import json
import platform
import sys
import tracemalloc

import numpy as np
from torchwasm import torch,__version__


def run_benchmarks(dtype='float32',depth=16,shape=(64,128)):
    if dtype not in ('float32','float64') or depth<1 or not shape or min(shape)<1:
        raise ValueError('Use float32/float64 and positive depth/shape')
    results=[]
    for retain in [False,True]:
        x=torch.tensor(np.full(shape,.2,dtype=dtype),requires_grad=True)
        gc.collect();tracemalloc.start()
        try:
            hidden=x
            for _ in range(depth):hidden=(hidden+.02).tanh()
            loss=hidden.square().mean();del hidden;gc.collect()
            forward,forward_peak=tracemalloc.get_traced_memory()
            loss.backward(retain_graph=retain);gc.collect()
            backward,backward_peak=tracemalloc.get_traced_memory()
            final=None
            if retain:
                loss.backward();gc.collect();final=tracemalloc.get_traced_memory()[0]
            results.append(dict(dtype=dtype,retain_graph=retain,loss=loss.item(),
                gradient_max=float(x.grad.numpy().max()),forward_retained_bytes=forward,
                forward_peak_bytes=forward_peak,after_backward_retained_bytes=backward,
                backward_peak_bytes=backward_peak,after_final_release_bytes=final))
        finally:
            tracemalloc.stop()
        del x,loss;gc.collect()
    return dict(library_version=__version__,python=platform.python_version(),numpy=np.__version__,
                implementation=sys.modules[torch.Tensor.__module__].__file__,depth=depth,
                shape=shape,inputs_allocated_before_trace=True,results=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype',choices=['float32','float64'],default='float32')
    parser.add_argument('--depth',type=int,default=16)
    parser.add_argument('--output')
    options=parser.parse_args();report=json.dumps(run_benchmarks(options.dtype,options.depth),indent=2)+'\n'
    if options.output:
        with open(options.output,'w',encoding='utf8') as stream:stream.write(report)
    print(report)
