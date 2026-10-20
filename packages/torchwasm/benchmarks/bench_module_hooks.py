"""Measure module dispatch, construction and checkpoint callback overhead.

Run directly, or exec in Pyodide with argv. Baselines without hook registration
still execute all common cases. Inputs/model construction are outside hot loops
except for the explicitly named construction/registration workloads.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torchwasm
from torchwasm import torch


def run_benchmarks(repeat=11,iterations=1200,warmup=3,dtype='float32',cases=None):
    if dtype not in ('float32','float64'):raise ValueError('Use float32/float64')
    if min(repeat,iterations)<1 or warmup<0:raise ValueError('Invalid sample counts')
    options=SimpleNamespace(repeat=repeat,iterations=iterations,warmup=warmup,dtype=dtype)
    torch.manual_seed(283)
    dtype=getattr(torch,options.dtype)
    x=torch.ones(16,32,dtype=dtype)
    identity=torch.nn.Identity()
    chain=torch.nn.Sequential(*(torch.nn.Identity() for _ in range(16)))
    model=torch.nn.Sequential(torch.nn.Linear(32,64,dtype=dtype),torch.nn.Tanh(),
                              torch.nn.Linear(64,4,dtype=dtype))
    frozen=torch.nn.Linear(32,32,dtype=dtype).requires_grad_(False)
    state=model.state_dict()
    available={
        'identity_call':lambda:identity(x),
        'sequential_identity_16':lambda:chain(x),
        'linear_inference':lambda:frozen(x),
        'construct_identity':torch.nn.Identity,
        'state_dict':model.state_dict,
        'load_state_dict':lambda:model.load_state_dict(state),
    }
    if hasattr(type(identity),'register_forward_hook'):
        passive=torch.nn.Identity()
        passive.register_forward_pre_hook(lambda *a:None)
        passive.register_forward_hook(lambda *a:None)
        kwargs=torch.nn.Identity()
        kwargs.register_forward_pre_hook(lambda m,a,k:(a,k),with_kwargs=True)
        kwargs.register_forward_hook(lambda m,a,k,o:o,with_kwargs=True,always_call=True)
        cleared=torch.nn.Identity();cleared.register_forward_hook(lambda *a:None).remove()
        cleared(x)  # First post-removal call re-establishes the no-hook fast path.
        temporary=torch.nn.Identity()
        callback=lambda *a:None
        def register_remove():
            temporary.register_forward_hook(callback,with_kwargs=True,always_call=True).remove()
        checkpoint=torch.nn.Sequential(torch.nn.Linear(32,64,dtype=dtype),torch.nn.Tanh(),
                                       torch.nn.Linear(64,4,dtype=dtype))
        checkpoint.register_state_dict_pre_hook(lambda *a:None)
        checkpoint.register_state_dict_post_hook(lambda *a:None)
        checkpoint.register_load_state_dict_pre_hook(lambda *a:None)
        checkpoint.register_load_state_dict_post_hook(lambda *a:None)
        checkpoint_state=checkpoint.state_dict()
        available.update({
            'identity_pre_post':lambda:passive(x),
            'identity_kwargs_pre_post':lambda:kwargs(input=x),
            'identity_after_remove':lambda:cleared(x),
            'register_remove':register_remove,
            'state_dict_with_hooks':checkpoint.state_dict,
            'load_state_dict_with_hooks':lambda:checkpoint.load_state_dict(checkpoint_state),
        })
    names=list(available) if cases is None else list(cases)
    if not names or len(set(names))!=len(names) or any(name not in available for name in names):
        raise ValueError('Select unique available cases: '+', '.join(available))
    results={}
    for name in names:
        run=available[name]
        inner=options.iterations*(10 if name in ('identity_call','sequential_identity_16',
            'construct_identity','identity_pre_post','identity_kwargs_pre_post','identity_after_remove',
            'register_remove') else 1)
        for _ in range(options.warmup):run()
        samples=[]
        for _ in range(options.repeat):
            start=time.perf_counter()
            for _ in range(inner):run()
            samples.append((time.perf_counter()-start)*1000/inner)
        results[name]={'median_ms':statistics.median(samples),'samples_ms':samples,'iterations':inner}
    return {'library_version':torchwasm.__version__,'python':platform.python_version(),
        'implementation':sys.modules[torch.Tensor.__module__].__file__,
        'numpy':np.__version__,'platform':platform.platform(),'dtype':options.dtype,'seed':283,
        'repeat':options.repeat,'iterations':options.iterations,'warmup':options.warmup,'results':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output');parser.add_argument('--repeat',type=int,default=11)
    parser.add_argument('--iterations',type=int,default=1200)
    parser.add_argument('--dtype',choices=['float32','float64'],default='float32')
    parser.add_argument('--case',action='append',dest='cases')
    options=parser.parse_args()
    report=json.dumps(run_benchmarks(repeat=options.repeat,iterations=options.iterations,
                                    dtype=options.dtype,cases=options.cases),indent=2)+'\n'
    if options.output:
        with open(options.output,'w',encoding='utf8') as stream:stream.write(report)
    print(report)
