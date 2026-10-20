"""Compare warm public matmul calls in two source trees on CPU or Pyodide.

Run fresh processes with distinct --cycle values. A0/A1 are independent imports
of the same baseline. B is the candidate. Input setup and value checks are untimed.
"""
import argparse
import importlib.util
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import warnings
import numpy as np
def load_library(root, label):
    name = '_matmul_benchmark_' + label
    spec = importlib.util.spec_from_file_location(name, str(Path(root)/'__init__.py'),
                                                submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.torch


def layout(value, name):
    if name == 'C': return value.copy(order='C')
    if name == 'F': return value.copy(order='F')
    if name == 'stride':
        storage=np.empty((value.shape[0]*2,value.shape[1]*2),dtype=value.dtype)
        storage[::2,::2]=value
        return storage[::2,::2]
    raise ValueError(name)


def specs():
    rows=[]
    for dtype in ['float32','float64']:
        for name,shape,order,modes in [
            ('tiny',(4,5,6),'C',['forward']),
            ('small',(32,64,32),'C',['forward','backward']),
            ('below_threshold',(64,64,128),'C',['forward']),
            ('threshold',(128,64,128),'C',['forward','backward']),
            ('dense128',(128,128,128),'C',['forward','backward']),
            ('dense256',(256,256,256),'C',['forward','backward']),
            ('dense512',(512,512,512),'C',['forward']),
            ('transpose256',(256,256,256),'F',['forward']),
            ('stride256',(256,256,256),'stride',['forward']),
            ('skinny',(1024,64,1),'C',['forward']),
            ('batch',(2,64,64),'C',['forward']),
            ('nan256',(256,256,256),'C',['forward']),
            ('subnormal256',(256,256,256),'C',['forward']),
        ]:
            for mode in modes: rows.append(dict(key=f'{dtype}_{name}_{mode}',dtype=dtype,name=name,shape=shape,layout=order,mode=mode))
    return rows


def inputs(spec):
    rng=np.random.default_rng(76086)
    m,k,n=spec['shape'];dtype=spec['dtype']
    if spec['name']=='batch':
        a=rng.uniform(-1,1,size=(2,64,64)).astype(dtype);b=rng.uniform(-1,1,size=(64,64)).astype(dtype)
    else:
        a=layout(rng.uniform(-1,1,size=(m,k)).astype(dtype),spec['layout'])
        b=layout(rng.uniform(-1,1,size=(k,n)).astype(dtype),spec['layout'])
    if spec['name']=='nan256':a[0,0]=np.nan
    if spec['name']=='subnormal256':a[0,0]=np.finfo(dtype).smallest_subnormal
    return a,b,np.ones(np.matmul(a,b).shape,dtype=dtype)


def function(torch,raw,mode):
    a,b=[torch.from_numpy(x).requires_grad_(mode=='backward') for x in raw[:2]]
    seed=torch.from_numpy(raw[2])
    if mode=='forward':return lambda:a@b
    return lambda:torch.autograd.grad(a@b,(a,b),seed)


def duration(fn,n):
    start=time.perf_counter()
    for _ in range(n):fn()
    return time.perf_counter()-start


def iterations(fn):
    n=1
    while True:
        elapsed=duration(fn,n)
        if elapsed>=.02:return max(1,math.ceil(n*.05/elapsed))
        n*=max(2,min(32,math.ceil(.025/max(elapsed,1e-9))))


def run_benchmark(baseline_source,candidate_source,cycle=0,sample_count=3):
    roots={'A0':Path(baseline_source),'A1':Path(baseline_source),'B':Path(candidate_source)}
    libraries={label:load_library(folder,label) for label,folder in roots.items()}
    source_hashes={label:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*')) if p.is_file()} for label,folder in [('baseline',roots['A0']),('candidate',roots['B'])]}
    rows=[];case_order=specs();random.Random(76076+cycle).shuffle(case_order)
    for spec in case_order:
        raw=inputs(spec)
        functions={name:function(torch,raw,spec['mode']) for name,torch in libraries.items()}
        checked={name:fn() for name,fn in functions.items()}
        for name,value in checked.items():
            actual=value if isinstance(value,tuple) else (value,)
            expected=checked['A0'] if isinstance(checked['A0'],tuple) else (checked['A0'],)
            for x,y in zip(actual,expected):
                np.testing.assert_allclose(x.detach().numpy(),y.detach().numpy(),rtol=3e-4,atol=3e-4,equal_nan=True)
        del checked,actual,expected,value,x,y
        order=['A0','A1','B'];order=order[cycle%3:]+order[:cycle%3]
        counts={name:iterations(functions[name]) for name in order}
        gc.collect();enabled=gc.isenabled();gc.disable()
        try:
            samples={name:[] for name in order}
            for repeat in range(sample_count):
                rotation=repeat%3
                permutation=order[rotation:]+order[:rotation]
                for name in permutation:
                    elapsed=duration(functions[name],counts[name])
                    samples[name].append(dict(iterations=counts[name],elapsed_seconds=elapsed,seconds_per_call=elapsed/counts[name]))
        finally:
            if enabled:gc.enable()
        median={name:statistics.median(x['seconds_per_call'] for x in samples[name]) for name in order}
        rows.append(dict(**spec,samples=samples,median_seconds=median,candidate_over_baseline=median['B']/median['A0'],control_over_baseline=median['A1']/median['A0']))
    return dict(cycle=cycle,platform=sys.platform,python=sys.version,numpy=np.__version__,source_hashes=source_hashes,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source',required=True,type=Path)
    parser.add_argument('--candidate-source',required=True,type=Path)
    parser.add_argument('--cycle',type=int,default=0)
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.samples<3:parser.error('--samples must be at least 3')
    result=run_benchmark(args.baseline_source,args.candidate_source,args.cycle,args.samples)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='completed',cycle=args.cycle,rows=len(result['rows']))),flush=True)
